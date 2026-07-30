"""
Token Aligner for handling different tokenizers between SLM and LLM models.
Token 对齐器 —— 处理 SLM（小语言模型）与 LLM（大语言模型）之间的不同 tokenizer。

This module provides functionality to align tokens between two different tokenizers,
handling cases where the same text is tokenized differently.
本模块提供在两种不同 tokenizer 之间对齐 token 的功能，
处理同一段文本被不同 tokenizer 分词为不同 token 序列的情况。

在 C2C 框架中的作用:
    当 Sharer 和 Receiver 使用不同的 tokenizer 时（如 Qwen2.5 vs Qwen3），
    相同的文本会被切分为不同长度、不同 ID 的 token 序列。
    TokenAligner 负责：
    1. 将 SLM token 序列解码为字符串片段
    2. 用 LLM tokenizer 重新编码这些字符串
    3. 处理 1-to-1 / 1-to-many / many-to-1 的映射关系
    4. 在 chat template 级别对齐消息（template 部分 + message 部分分别处理）

核心类:
    - AlignmentStrategy: 对齐策略枚举（FIRST=取第一个、LONGEST=取最长的）
    - TokenAligner: 主类，负责 token 对齐、chat message 对齐、可视化等

与其他模块的关系:
    - 被 rosetta/train/dataset_adapters.py 中的 AlignedChatDataset 调用
    - 被 rosetta/baseline/multi_stage.py 中的 TwoStageRosetta 使用
    - 不直接依赖 Projector/Fuser，属于数据预处理层
"""

from typing import List, Tuple, Optional, Dict, Literal, Union
import torch
from transformers import PreTrainedTokenizerBase
from enum import Enum


class AlignmentStrategy(Enum):
    """
    Strategies for handling 1-to-many token alignments.
    处理 SLM 到 LLM 的 1-to-many token 映射时的选择策略枚举。

    在 C2C 框架中的作用:
        当 SLM（Sharer 侧模型）的一个 token 被解码为字符串后，
        用 LLM（Receiver 侧模型）的 tokenizer 重新编码可能产生多个 token。
        此枚举定义了从多个候选 LLM token 中选取哪一个的规则。

    策略说明:
        - FIRST:   始终选取第一个 LLM token。速度快，适合大多数场景。
        - LONGEST: 选取解码后字符串最长的那个 LLM token。
                   可以保留更多信息，但计算开销稍大。
    """
    FIRST = "first"      # Always take the first LLM token / 始终取第一个 LLM token
    LONGEST = "longest"  # Take the LLM token with the longest string / 取解码字符串最长的 LLM token


class TokenAligner:
    """
    Aligns tokens between SLM (Small Language Model) and LLM (Large Language Model) tokenizers.

    在 C2C 框架中的核心作用:
        当 Sharer 模型（SLM）和 Receiver 模型（LLM）使用不同的 tokenizer 时，
        相同的文本会被切分为不同长度、不同 ID 的 token 序列。
        TokenAligner 负责建立 SLM token 到 LLM token 的映射关系，使得：
        1. SLM 产生的 KV-Cache 能够正确对应到 LLM 的 token 位置
        2. 在训练数据准备阶段，能够生成对齐的 (SLM序列, LLM序列) 对
        3. 支持 chat template 级别的精细对齐（区分 template 部分和 message 部分）

    核心算法流程:
        对于每个 SLM token:
        1. 解码为字符串（保留特殊 token）
        2. 用 LLM tokenizer 重新编码
        3. 处理映射关系：
           - 1-to-1: 直接使用
           - 1-to-many: 根据策略（FIRST/LONGEST）选择一个
           - 0-to-1: 使用 unk_token 兜底
        4. 缓存结果以提升性能

    与其他模块的关系:
        - 被 rosetta/train/dataset_adapters.py 中的 AlignedChatDataset 调用，
          用于生成训练数据时对 input_ids 进行对齐
        - 被 rosetta/baseline/multi_stage.py 中的 TwoStageRosetta 使用，
          用于两阶段推理时对齐 prompt 的 token 序列
        - 不直接依赖 Projector/Fuser，属于数据预处理层

    Attributes:
        slm_tokenizer: SLM（Sharer 侧）的 tokenizer
        llm_tokenizer: LLM（Receiver 侧）的 tokenizer
        strategy: 1-to-many 映射时的选择策略
        verbose: 是否打印调试信息
        _alignment_cache: 对齐结果缓存，避免重复计算
    """
    
    def __init__(
        self,
        slm_tokenizer: PreTrainedTokenizerBase,
        llm_tokenizer: PreTrainedTokenizerBase,
        strategy: Union[AlignmentStrategy, str] = AlignmentStrategy.FIRST,
        verbose: bool = False
    ):
        """
        Initialize the TokenAligner.
        初始化 TokenAligner 实例。

        Args / 参数:
            slm_tokenizer: The tokenizer for the Small Language Model (base)
                SLM（Sharer 侧小语言模型）的 tokenizer，作为对齐的基准
            llm_tokenizer: The tokenizer for the Large Language Model
                LLM（Receiver 侧大语言模型）的 tokenizer，目标对齐对象
            strategy: Strategy for handling 1-to-many token mappings
                     Either AlignmentStrategy enum or string ('first' or 'longest')
                处理 1-to-many token 映射时的选择策略，
                可以是 AlignmentStrategy 枚举或字符串 ('first' 或 'longest')
            verbose: Whether to print debug information during alignment
                是否在对齐过程中打印调试信息

        关键逻辑:
            1. pad_token 处理：如果 tokenizer 没有设置 pad_token，
               则使用 eos_token 作为替代，确保后续 padding 操作正常
            2. 策略转换：支持字符串形式的策略输入，自动转换为枚举类型
            3. 缓存初始化：使用字典缓存已计算的对齐结果，
               key 为 SLM token ID 元组，value 为对齐后的 LLM token ID 列表
        """
        self.slm_tokenizer = slm_tokenizer
        self.llm_tokenizer = llm_tokenizer

        # 确保两个 tokenizer 都有 pad_token，否则用 eos_token 替代
        # 这在后续 padding 对齐序列时是必需的
        if self.slm_tokenizer.pad_token is None:
            self.slm_tokenizer.pad_token = self.slm_tokenizer.eos_token
            self.slm_tokenizer.pad_token_id = self.slm_tokenizer.eos_token_id
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id

        # Handle string strategy input / 处理字符串形式的策略输入
        if isinstance(strategy, str):
            strategy = AlignmentStrategy(strategy.lower())
        self.strategy = strategy
        self.verbose = verbose

        # Cache for token mappings to improve performance
        # 对齐结果缓存：key=(slm_token_id_1, slm_token_id_2, ...), value=[llm_token_id_1, ...]
        # 避免对相同输入重复计算，提升训练/推理时的性能
        self._alignment_cache: Dict[Tuple[int, ...], List[int]] = {}
    
    def align_tokens(
        self,
        slm_token_ids: Union[List[int], torch.Tensor],
        return_mapping: bool = False
    ) -> Union[List[int], Tuple[List[int], List[Tuple[int, List[int]]]]]:
        """
        Align SLM tokens to LLM tokens.
        将 SLM token 序列对齐为 LLM token 序列（核心方法）。

        算法流程:
            对每个 SLM token 执行以下操作:
            1. 解码: 用 slm_tokenizer.decode() 将 token ID 还原为字符串
            2. 特殊 token 处理: 如果是特殊 token (pad/eos/bos/unk)，
               调用 _map_special_token() 直接映射
            3. 重新编码: 用 llm_tokenizer.encode() 将字符串编码为 LLM token
            4. 根据编码结果长度分别处理:
               - 0 个 token: 使用 unk_token 兜底（不应发生）
               - 1 个 token: 完美的 1-to-1 映射，直接使用
               - 多个 token: 1-to-many 映射，调用 _apply_strategy() 选择一个
            5. 缓存结果

        Args / 参数:
            slm_token_ids: Token IDs from the SLM tokenizer
                SLM tokenizer 产出的 token ID 列表或 Tensor
            return_mapping: If True, also return the detailed mapping
                是否同时返回详细的映射关系（用于调试和分析）

        Returns / 返回值:
            If return_mapping is False:
                List of aligned LLM token IDs (与 SLM 等长的 LLM token ID 列表)
            If return_mapping is True:
                Tuple of (aligned_llm_token_ids, mapping_details)
                其中 mapping_details 是 [(slm_token_id, [候选 llm_token_ids]), ...]
        """
        # Convert to list if tensor / 如果输入是 Tensor，转为 Python list
        if isinstance(slm_token_ids, torch.Tensor):
            slm_token_ids = slm_token_ids.tolist()

        # Check cache / 查缓存：如果同样的 SLM 序列已对齐过，直接返回
        cache_key = tuple(slm_token_ids)
        if cache_key in self._alignment_cache and not return_mapping:
            return self._alignment_cache[cache_key]

        aligned_llm_tokens = []      # 对齐后的 LLM token ID 列表
        mapping_details = []         # 详细映射关系（用于调试/分析）

        # 逐 token 对齐
        for slm_token_id in slm_token_ids:
            # Decode SLM token to string (without special token processing)
            # 将 SLM token ID 解码为字符串（不跳过特殊 token，不清理空格）
            slm_token_str = self.slm_tokenizer.decode(
                [slm_token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )

            # Handle special tokens / 处理特殊 token（pad/eos/bos/unk 等）
            if slm_token_id in self.slm_tokenizer.all_special_ids:
                # Try to find corresponding special token in LLM tokenizer
                # 尝试在 LLM tokenizer 中找到对应的特殊 token
                llm_token_id = self._map_special_token(slm_token_id, slm_token_str)
                aligned_llm_tokens.append(llm_token_id)
                mapping_details.append((slm_token_id, [llm_token_id]))
                continue

            # Tokenize the string with LLM tokenizer
            # 用 LLM tokenizer 对该字符串重新编码
            llm_token_ids = self.llm_tokenizer.encode(
                slm_token_str,
                add_special_tokens=False,  # 不添加特殊 token，保持纯净
                return_tensors=None
            )

            if len(llm_token_ids) == 0:
                # Handle empty tokenization (shouldn't normally happen)
                # 处理空编码结果（通常不应该发生，属于异常情况）
                if self.verbose:
                    print(f"Warning: SLM token {slm_token_id} ('{slm_token_str}') "
                          f"resulted in empty LLM tokenization")
                # Use unknown token as fallback / 使用 unk_token 作为兜底
                llm_token_id = self.llm_tokenizer.unk_token_id or 0
                aligned_llm_tokens.append(llm_token_id)
                mapping_details.append((slm_token_id, [llm_token_id]))

            elif len(llm_token_ids) == 1:
                # Perfect 1-to-1 mapping / 完美的 1-to-1 映射，直接使用
                aligned_llm_tokens.append(llm_token_ids[0])
                mapping_details.append((slm_token_id, llm_token_ids))

            else:
                # 1-to-many mapping, apply strategy
                # 1-to-many 映射：一个 SLM token 对应多个 LLM token，需要策略选择
                selected_token = self._apply_strategy(llm_token_ids, slm_token_str)
                aligned_llm_tokens.append(selected_token)
                mapping_details.append((slm_token_id, llm_token_ids))

                if self.verbose:
                    # 打印调试信息：显示所有候选 token 和最终选择
                    selected_str = self.llm_tokenizer.decode(
                        [selected_token],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False
                    )
                    print(f"SLM token {slm_token_id} ('{slm_token_str}') -> "
                          f"LLM tokens {llm_token_ids}, selected {selected_token} ('{selected_str}')")

        # Cache the result / 缓存本次对齐结果
        self._alignment_cache[cache_key] = aligned_llm_tokens

        if return_mapping:
            return aligned_llm_tokens, mapping_details
        return aligned_llm_tokens
    
    def _map_special_token(self, slm_token_id: int, slm_token_str: str) -> int:
        """
        Map special tokens between tokenizers.
        在两个 tokenizer 之间映射特殊 token（pad/eos/bos/unk）。

        三级回退策略:
            1. 直接映射: 通过预定义的 pad->pad, eos->eos 等映射表查找
            2. 字符串匹配: 用 SLM 特殊 token 的字符串表示在 LLM tokenizer 中查找
            3. 兜底: 返回 LLM 的 unk_token_id

        Args / 参数:
            slm_token_id: The SLM special token ID
                SLM 侧的特殊 token ID
            slm_token_str: The string representation of the special token
                该特殊 token 的字符串表示（如 "<pad>", "</s>" 等）

        Returns / 返回值:
            The corresponding LLM token ID
            对应的 LLM token ID
        """
        # Common special token mappings / 预定义的常见特殊 token 映射表
        # 将 SLM 的 pad/eos/bos/unk 映射到 LLM 的对应 token
        special_token_map = {
            self.slm_tokenizer.pad_token_id: self.llm_tokenizer.pad_token_id,
            self.slm_tokenizer.eos_token_id: self.llm_tokenizer.eos_token_id,
            self.slm_tokenizer.bos_token_id: self.llm_tokenizer.bos_token_id,
            self.slm_tokenizer.unk_token_id: self.llm_tokenizer.unk_token_id,
        }

        # Direct mapping if available / 第一级回退：直接查表
        if slm_token_id in special_token_map and special_token_map[slm_token_id] is not None:
            return special_token_map[slm_token_id]

        # Try to find by string representation
        # 第二级回退：通过字符串表示在 LLM tokenizer 中查找
        try:
            llm_token_id = self.llm_tokenizer.convert_tokens_to_ids(slm_token_str)
            if llm_token_id != self.llm_tokenizer.unk_token_id:
                return llm_token_id
        except:
            pass

        # Fallback to unknown token / 第三级回退：使用 unk_token
        return self.llm_tokenizer.unk_token_id or 0
    
    def _apply_strategy(self, llm_token_ids: List[int], original_str: str) -> int:
        """
        Apply the selected strategy to choose one LLM token from multiple candidates.
        当 1 个 SLM token 映射到多个 LLM token 时，根据策略选择一个。

        策略详解:
            - FIRST:   直接返回候选列表的第一个 token。
                       优点：速度快，无需额外计算。
                       缺点：可能丢失信息（后续 token 被忽略）。
            - LONGEST: 遍历所有候选 token，选择解码后字符串最长的那个。
                       优点：保留更多语义信息。
                       缺点：需要多次 decode 调用，计算开销更大。

        Args / 参数:
            llm_token_ids: List of candidate LLM token IDs
                候选 LLM token ID 列表（长度 >= 2）
            original_str: The original string from SLM token
                SLM token 解码后的原始字符串（目前未使用，保留供扩展）

        Returns / 返回值:
            The selected LLM token ID
            选中的 LLM token ID
        """
        if self.strategy == AlignmentStrategy.FIRST:
            # FIRST 策略：直接取第一个候选 token
            return llm_token_ids[0]

        elif self.strategy == AlignmentStrategy.LONGEST:
            # LONGEST 策略：找解码后字符串最长的 token
            # Find the token with the longest string representation
            longest_token = llm_token_ids[0]
            longest_length = 0

            for token_id in llm_token_ids:
                # 解码每个候选 token，比较字符串长度
                token_str = self.llm_tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False
                )
                if len(token_str) > longest_length:
                    longest_length = len(token_str)
                    longest_token = token_id

            return longest_token

        else:
            # Default to first token if unknown strategy / 未知策略默认取第一个
            return llm_token_ids[0]
    
    def align_sequence(
        self,
        text: str,
        return_details: bool = False
    ) -> Union[Tuple[List[int], List[int]], Dict[str, any]]:
        """
        Tokenize text with both tokenizers and return aligned sequences.
        用两种 tokenizer 对文本进行编码，并返回对齐后的序列（便捷封装方法）。

        与 align_tokens() 的区别:
            - align_tokens(): 输入已有的 SLM token ID 列表，直接对齐
            - align_sequence(): 输入原始文本，自动用 SLM tokenizer 编码后再对齐
              适合快速测试和调试

        Args / 参数:
            text: The input text to tokenize and align
                待对齐的原始文本
            return_details: If True, return detailed alignment information
                是否返回详细的对齐信息（用于分析和调试）

        Returns / 返回值:
            If return_details is False:
                Tuple of (slm_token_ids, aligned_llm_token_ids)
                (SLM token ID 列表, 对齐后的 LLM token ID 列表)
            If return_details is True:
                Dictionary with detailed alignment information including:
                - text: 原始文本
                - slm_token_ids / slm_decoded: SLM 编码结果及解码
                - aligned_llm_token_ids / aligned_llm_decoded: 对齐后的 LLM 编码结果
                - original_llm_token_ids: LLM tokenizer 直接编码的结果（用于对比）
                - mapping: 详细映射关系
                - one_to_one_rate: 1-to-1 映射的比例（越高说明 tokenizer 越兼容）
        """
        # Tokenize with SLM / 用 SLM tokenizer 编码文本
        slm_tokens = self.slm_tokenizer.encode(
            text,
            add_special_tokens=True,  # 添加特殊 token（如 bos/eos）
            return_tensors=None
        )

        # Get aligned LLM tokens / 获取对齐后的 LLM token
        if return_details:
            aligned_llm_tokens, mapping = self.align_tokens(slm_tokens, return_mapping=True)

            # Decode tokens for inspection / 解码 token 以便人工检查
            slm_decoded = [
                self.slm_tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for tid in slm_tokens
            ]
            llm_decoded = [
                self.llm_tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for tid in aligned_llm_tokens
            ]

            # Original LLM tokenization for comparison
            # 用 LLM tokenizer 直接编码原始文本（用于与对齐结果对比）
            original_llm_tokens = self.llm_tokenizer.encode(
                text,
                add_special_tokens=True,
                return_tensors=None
            )

            # One-to-one mapping statistics / 统计 1-to-1 映射的比例
            # 比例越高说明两个 tokenizer 的切分越兼容
            num_tokens = len(slm_tokens)
            one_to_one_count = sum(1 for _slm_id, candidates in mapping if len(candidates) == 1)
            one_to_one_rate = (one_to_one_count / num_tokens) if num_tokens > 0 else 0.0

            return {
                'text': text,
                'slm_token_ids': slm_tokens,
                'slm_decoded': slm_decoded,
                'aligned_llm_token_ids': aligned_llm_tokens,
                'aligned_llm_decoded': llm_decoded,
                'original_llm_token_ids': original_llm_tokens,
                'mapping': mapping,
                'strategy': self.strategy.value,
                'num_tokens': num_tokens,
                'one_to_one_count': one_to_one_count,
                'one_to_one_rate': one_to_one_rate
            }
        else:
            aligned_llm_tokens = self.align_tokens(slm_tokens)
            return slm_tokens, aligned_llm_tokens
    
    def visualize_alignment(self, text: str):
        """
        Print a visual representation of the token alignment.
        打印 token 对齐的可视化表示（调试工具）。

        输出格式:
            1. 头部信息：文本、策略
            2. 三行 token ID 总览：SLM / 对齐后 LLM / 原始 LLM
            3. 逐 token 对齐详情：
               - 普通映射：[位置] SLM ID ('解码') -> LLM ID ('解码')
               - 1-to-many 映射：额外显示 [candidates: ...] 列出所有候选

        Args / 参数:
            text: The text to analyze
                待分析的文本
        """
        details = self.align_sequence(text, return_details=True)
        
        print("=" * 80)
        print(f"Text: {text}")
        print(f"Strategy: {details['strategy']}")
        print("=" * 80)
        print(f"SLM tokens ({len(details['slm_token_ids'])}): {details['slm_token_ids']}")
        print(f"Aligned LLM tokens ({len(details['aligned_llm_token_ids'])}): {details['aligned_llm_token_ids']}")
        print(f"Original LLM tokens ({len(details['original_llm_token_ids'])}): {details['original_llm_token_ids']}")
        print("-" * 80)
        print("Token-by-token alignment:")
        
        for i, (slm_id, llm_id) in enumerate(zip(details['slm_token_ids'], details['aligned_llm_token_ids'])):
            slm_str = details['slm_decoded'][i]
            llm_str = details['aligned_llm_decoded'][i]
            mapping_info = details['mapping'][i]

            if len(mapping_info[1]) > 1:
                # 1-to-many 映射：显示所有候选 token 及其解码字符串
                candidates_str = ', '.join([
                    f"{tid}:'{self.llm_tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)}'"
                    for tid in mapping_info[1]
                ])
                print(f"  [{i:3d}] SLM {slm_id:6d} ('{slm_str}') -> "
                      f"LLM {llm_id:6d} ('{llm_str}') "
                      f"[candidates: {candidates_str}]")
            else:
                # 1-to-1 映射：简洁显示
                print(f"  [{i:3d}] SLM {slm_id:6d} ('{slm_str}') -> "
                      f"LLM {llm_id:6d} ('{llm_str}')")
        print("=" * 80)
    
    def clear_cache(self):
        """Clear the alignment cache. / 清除对齐结果缓存。"""
        self._alignment_cache.clear()

    # ================================================================
    # Chat messages alignment / Chat 消息对齐
    # ================================================================
    # 以下方法用于对齐 chat 格式的消息序列。
    # 在 C2C 中，训练数据和推理输入都是 chat 格式的消息列表（如
    # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]），
    # 每个 tokenizer 会使用自己的 chat template 将消息列表转换为 token 序列。
    # 这些方法负责在 chat template 级别进行精细对齐。
    def _apply_chat_template_to_ids(
        self,
        tokenizer: PreTrainedTokenizerBase,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool,
        enable_thinking: bool,
        remove_last_surfix: bool
    ) -> Tuple[str, List[int], Optional[List[Tuple[int, int]]]]:
        """
        Apply chat template (no tokenization) then tokenize to ids with optional offsets.
        应用 chat template 并编码为 token ID，同时获取字符偏移量。

        处理流程:
            1. 调用 tokenizer.apply_chat_template() 将消息列表格式化为模板字符串
               （如 "<|user|>...<|assistant|>..."）
            2. 调用 tokenizer() 对模板字符串进行编码，获取 input_ids 和 offset_mapping
            3. offset_mapping 记录了每个 token 在原始字符串中的字符位置范围

        特殊参数 remove_last_surfix:
            当为 True 时，移除最后一条 assistant 消息后的模板后缀
            （如 generation prompt "<|assistant|>\\n"）。
            主要用于训练时，只保留 assistant 的实际回复内容。

        Args / 参数:
            tokenizer: 使用的 tokenizer（SLM 或 LLM 的）
            messages: chat 消息列表，格式为 [{"role": "...", "content": "..."}, ...]
            add_generation_prompt: 是否在末尾添加生成提示（如 "<|assistant|>\\n"）
            enable_thinking: 是否启用思考模式（部分模型支持）
            remove_last_surfix: 是否移除最后一条消息后的模板后缀

        Returns / 返回值:
            Tuple of (templated_text, input_ids, offsets):
            - templated_text: 应用模板后的完整字符串
            - input_ids: 编码后的 token ID 列表
            - offsets: 每个 token 对应的字符位置范围 [(start, end), ...]，
              用于后续确定哪些 token 属于 message 内容、哪些属于 template
        """
        if remove_last_surfix:
            # 特殊处理：移除最后一条 assistant 消息后的模板后缀
            assert messages[-1]["role"] == "assistant", "Last message must be an assistant message"
            # 先对除最后一条外的消息应用模板（带 generation prompt）
            templated_text = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            # 然后手动追加最后一条 assistant 的 content（不带后缀）
            templated_text += messages[-1]["content"]
        else:
            # 正常处理：对所有消息应用模板
            templated_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking
            )
        # 对模板字符串进行编码，获取 token ID 和字符偏移映射
        encoded = tokenizer(
            templated_text,
            add_special_tokens=False,  # 模板已包含特殊 token，不重复添加
            return_offsets_mapping=True  # 返回每个 token 的字符位置范围
        )
        input_ids: List[int] = encoded["input_ids"]
        offsets = encoded.get("offset_mapping")
        return templated_text, input_ids, offsets

    @staticmethod
    def _first_non_empty_content(messages: List[Dict[str, str]]) -> Optional[str]:
        """
        从消息列表中找到第一个非空的 content 字段。
        用于确定 chat 模板中消息内容的起始位置。

        Args / 参数:
            messages: chat 消息列表

        Returns / 返回值:
            第一个非空 content 字符串，如果全部为空则返回 None
        """
        for m in messages:
            content = m.get("content")
            if isinstance(content, str) and len(content.strip()) > 0:
                return content
        return None

    def _find_boundary_token_index(
        self,
        tokenizer: PreTrainedTokenizerBase,
        templated_text: str,
        offsets: Optional[List[Tuple[int, int]]],
        content_text: Optional[str]
    ) -> int:
        """
        Find token index where the first non-empty message content starts.
        Falls back to 0 if not found.
        找到第一条非空消息内容在 token 序列中的起始索引。

        算法逻辑:
            1. 在模板字符串中查找 content_text 的字符位置 (char_idx)
            2. 如果有 offset_mapping，遍历找到第一个 start >= char_idx 的 token 索引
            3. 如果没有 offset_mapping（回退方案），对 char_idx 前的子串编码并计数

        用途:
            确定 template 部分和 message 部分的分界点，
            用于区分哪些 token 属于模板标记、哪些属于实际消息内容。

        Args / 参数:
            tokenizer: 使用的 tokenizer
            templated_text: 应用模板后的完整字符串
            offsets: 每个 token 的字符位置偏移量列表，可能为 None
            content_text: 要查找的消息内容文本

        Returns / 返回值:
            消息内容起始位置对应的 token 索引（从 0 开始），
            如果找不到则返回 0
        """
        if not content_text:
            return 0
        # 在模板字符串中查找消息内容的字符位置
        char_idx = templated_text.find(content_text)
        if char_idx < 0:
            # Try a shorter probe to improve chances
            # 尝试用更短的前缀进行匹配（处理部分内容被截断的情况）
            probe = content_text[: min(32, len(content_text))]
            if len(probe) > 0:
                char_idx = templated_text.find(probe)
        if char_idx < 0:
            return 0

        if offsets:
            # 有 offset_mapping 时：遍历找到第一个起始位置 >= char_idx 的 token
            for idx, (start, _end) in enumerate(offsets):
                if start >= char_idx:
                    return idx
            return len(offsets)

        # Fallback without offsets: tokenize prefix and count tokens
        # 回退方案：对 char_idx 前的子串进行编码，token 数量即为索引
        prefix = templated_text[:char_idx]
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        return len(prefix_ids)

    @staticmethod
    def _compute_content_spans(templated_text: str, messages: List[Dict[str, str]]) -> List[Tuple[int, int]]:
        """
        Compute character spans in templated_text that correspond to message contents.
        Searches sequentially to reduce ambiguity when contents repeat.
        Enhanced matching: ensures the found content is followed by '<' (special token start)
        to avoid matching content inside special tokens like <begin_of_text>.

        计算模板字符串中每条消息内容对应的字符范围（span）。

        算法核心思想:
            1. 顺序遍历每条消息的 content
            2. 在模板字符串中从上次匹配结束位置开始搜索
            3. 增强匹配：验证匹配位置后是否紧跟 '<'（特殊 token 起始符），
               避免在 <begin_of_text> 等特殊 token 内部误匹配
            4. 如果增强匹配失败，回退到普通匹配并做额外校验

        用途:
            确定哪些字符范围对应消息内容，进而通过 offset_mapping
            确定哪些 token 属于 message（需要 token 对齐），
            哪些属于 template（只需 padding 对齐）。

        Args / 参数:
            templated_text: 应用 chat template 后的完整字符串
            messages: chat 消息列表

        Returns / 返回值:
            List of (start_char_idx, end_char_idx) tuples
            每个消息内容在模板字符串中的字符范围列表
        """
        spans: List[Tuple[int, int]] = []
        search_from = 0  # 从模板字符串的这个位置开始搜索（避免重复匹配）
        for m in messages:
            content = m.get("content")
            if not isinstance(content, str) or len(content) == 0:
                continue

            # Find all possible matches starting from search_from
            # 从 search_from 位置开始搜索 content
            idx = search_from
            found_valid_match = False

            while idx < len(templated_text):
                idx = templated_text.find(content, idx)
                if idx < 0:
                    break

                # Check if this match is valid (followed by '<' indicating a special token)
                # 增强匹配：检查匹配位置后是否紧跟 '<'（特殊 token 起始符）
                # 这可以避免在 <begin_of_text> 等特殊 token 内部误匹配
                end_pos = idx + len(content)
                if end_pos < len(templated_text) and templated_text[end_pos] == '<':
                    # Valid match: content is followed by a special token
                    # 有效匹配：content 后面紧跟特殊 token
                    spans.append((idx, end_pos))
                    search_from = end_pos
                    found_valid_match = True
                    break
                else:
                    # Check if this is the end of the text (also valid for last message)
                    # 如果是文本末尾，也算有效匹配（最后一条消息）
                    if end_pos == len(templated_text):
                        spans.append((idx, end_pos))
                        search_from = end_pos
                        found_valid_match = True
                        break

                # Invalid match, try next occurrence
                # 无效匹配，尝试下一个出现位置
                idx += 1

            # Fallback: if no valid match found with '<' requirement, use the old method
            # but only as a last resort and with additional validation
            # 回退方案：如果增强匹配失败，使用普通匹配并做额外校验
            if not found_valid_match:
                idx = templated_text.find(content, search_from)
                if idx < 0:
                    # Try searching from start as last resort
                    # 最后手段：从头开始搜索
                    idx = templated_text.find(content)

                if idx >= 0:
                    end_pos = idx + len(content)
                    # Additional check: avoid matching inside obvious special tokens
                    # 额外校验：避免在特殊 token 内部匹配
                    # Check if we're inside a special token (preceded by '<' and not followed by '>')
                    start_context = templated_text[max(0, idx-10):idx]
                    end_context = templated_text[end_pos:min(len(templated_text), end_pos+10)]

                    # Skip if we're clearly inside a special token
                    # 如果明显在特殊 token 内部（如 <begin_of_text>），跳过
                    if ('<' in start_context and '>' not in start_context and
                        'begin_of_text' in templated_text[max(0, idx-20):idx+20]):
                        # This looks like we're matching inside <begin_of_text> or similar
                        continue

                    spans.append((idx, end_pos))
                    search_from = end_pos

        return spans

    @staticmethod
    def _build_token_mask_from_spans(
        offsets: Optional[List[Tuple[int, int]]],
        num_tokens: int,
        spans: List[Tuple[int, int]]
    ) -> List[bool]:
        """
        Build a boolean mask for tokens whose offset range overlaps any span.
        If offsets are missing, default to all False.

        根据字符范围（spans）构建 token 级别的布尔掩码。
        掩码为 True 表示该 token 落在某个消息内容范围内。

        算法逻辑:
            对每个 token，检查其字符范围 [start, end) 是否与任一 span 重叠。
            重叠条件: token.start < span.end AND token.end > span.start

        用途:
            区分 template token 和 message token，
            后续对齐时对两者采用不同策略（padding vs token 映射）。

        Args / 参数:
            offsets: 每个 token 的字符位置偏移量 [(start, end), ...]
            num_tokens: token 总数
            spans: 消息内容的字符范围列表 [(start, end), ...]

        Returns / 返回值:
            长度为 num_tokens 的布尔列表，True 表示该 token 属于消息内容
        """
        if not offsets or len(offsets) != num_tokens:
            return [False] * num_tokens
        mask: List[bool] = []
        for (start, end) in offsets:
            if end <= start:
                # 空 token 或特殊 token，不属于任何内容
                mask.append(False)
                continue
            is_msg = False
            for s, e in spans:
                # overlap check / 检查 token 范围 [start, end) 是否与 span [s, e) 重叠
                if start < e and end > s:
                    is_msg = True
                    break
            mask.append(is_msg)
        return mask

    @staticmethod
    def _spans_to_token_ranges(
        offsets: List[Tuple[int, int]],
        spans: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Convert character spans to token index ranges using offsets.
        start token = first token with end > span_start
        end token = first token with start >= span_end

        将字符级别的 span 转换为 token 索引范围。

        转换规则:
            - start_idx: 第一个满足 offset.end > span.start 的 token 索引
              （即该 token 的结束位置超过了 span 的起始，说明它部分或完全在 span 内）
            - end_idx: 第一个满足 offset.start >= span.end 的 token 索引
              （即该 token 的起始位置已经到达或超过 span 的结束）
            - 最终范围: [start_idx, end_idx)

        Args / 参数:
            offsets: 每个 token 的字符位置偏移量 [(char_start, char_end), ...]
            spans: 字符级别的范围列表 [(char_start, char_end), ...]

        Returns / 返回值:
            List of (start_token_idx, end_token_idx) tuples
            token 索引范围列表，每个对应一个 span
        """
        ranges: List[Tuple[int, int]] = []
        n = len(offsets)
        for s, e in spans:
            # find start index / 找到第一个跨越 span 起始位置的 token
            start_idx = 0
            while start_idx < n and offsets[start_idx][1] <= s:
                start_idx += 1
            # find end index / 找到第一个越过 span 结束位置的 token
            end_idx = start_idx
            while end_idx < n and offsets[end_idx][0] < e:
                end_idx += 1
            ranges.append((start_idx, end_idx))
        return ranges

    def align_chat_messages(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        return_details: bool = False,
        remove_last_surfix: bool = False
    ) -> Dict[str, any]:
        """
        Align chat-templated sequences by sections (template/message/template...):
        - Preserve all template tokens (pad the shorter template section)
        - For each message section, map SLM tokens to LLM tokens 1:1 via strategy
        - If remove_last_surfix is True, remove the last suffix from the LLM text
        Returns essentials: slm_ids_padded, llm_ids_padded, message_mask (shared),
        slm_padding_mask, llm_padding_mask (True where token is padding inserted).
        When return_details=True, also returns 'sections' with aligned ranges.

        对齐 chat 格式消息序列（核心方法，用于训练数据准备和推理）。

        分段对齐策略（核心算法）:
            1. 将 chat 消息应用模板后，识别出交替出现的 section：
               [template, message, template, message, ..., template]
               - template section: 模板标记（如 <|user|>, <|assistant|> 等）
               - message section: 实际消息内容

            2. 对 SLM 和 LLM 分别执行上述识别，得到对应的 section 列表

            3. 逐 section 对齐：
               - template section: 取两者中较长的长度，短的用 pad_token 补齐
                 （因为不同 tokenizer 的模板标记数量可能不同）
               - message section: 调用 align_tokens() 将 SLM token 逐一映射为 LLM token
                 （保证 1:1 映射，长度一致）

            4. 同时生成多个掩码：
               - message_mask: 标记哪些位置是消息内容（用于计算 loss 时区分）
               - slm_padding_mask: SLM 序列中哪些是填充的 pad token
               - llm_padding_mask: LLM 序列中哪些是填充的 pad token

        Args / 参数:
            messages: chat 消息列表
            add_generation_prompt: 是否添加生成提示
            enable_thinking: 是否启用思考模式
            return_details: 是否返回详细的 section 信息
            remove_last_surfix: 是否移除最后一条消息的模板后缀

        Returns / 返回值:
            Dictionary with keys:
            - slm_ids_padded: 对齐并 padding 后的 SLM token ID 列表
            - llm_ids_padded: 对齐并 padding 后的 LLM token ID 列表
            - message_mask: 布尔列表，True 表示该位置属于消息内容
            - slm_padding_mask: 布尔列表，True 表示 SLM 侧是 padding
            - llm_padding_mask: 布尔列表，True 表示 LLM 侧是 padding
            - sections (可选): 每个 section 的类型和范围信息
        """
        assert not (add_generation_prompt and remove_last_surfix), "add_generation_prompt and remove_last_surfix cannot be True at the same time"

        # Build templated sequences with offsets
        # 步骤1：对 SLM 和 LLM 分别应用 chat template，获取模板文本、token ID 和偏移量
        slm_text, slm_ids, slm_offsets = self._apply_chat_template_to_ids(
            self.slm_tokenizer, messages, add_generation_prompt, enable_thinking, remove_last_surfix
        )
        llm_text, llm_ids, llm_offsets = self._apply_chat_template_to_ids(
            self.llm_tokenizer, messages, add_generation_prompt, enable_thinking, remove_last_surfix
        )

        # Required pad tokens / 确保 pad_token 已设置
        assert self.slm_tokenizer.pad_token_id is not None, "SLM pad_token_id required"
        assert self.llm_tokenizer.pad_token_id is not None, "LLM pad_token_id required"
        slm_pad_id = self.slm_tokenizer.pad_token_id
        llm_pad_id = self.llm_tokenizer.pad_token_id

        # Content spans (char) and token ranges
        # 步骤2：计算消息内容在模板字符串中的字符范围，并转换为 token 索引范围
        content_spans_slm = self._compute_content_spans(slm_text, messages)
        content_spans_llm = self._compute_content_spans(llm_text, messages)
        assert slm_offsets is not None and llm_offsets is not None, "offset_mapping required"
        # slm_msg_ranges / llm_msg_ranges: 每条消息内容对应的 token 索引范围 [(start, end), ...]
        slm_msg_ranges = self._spans_to_token_ranges(slm_offsets, content_spans_slm)
        llm_msg_ranges = self._spans_to_token_ranges(llm_offsets, content_spans_llm)
        # Build section ranges (template/message alternating)
        # 步骤3：将 token 序列划分为交替的 template/message section
        def build_sections(total_len: int, msg_ranges: List[Tuple[int,int]]):
            """
            根据消息范围列表，构建 section 列表。
            返回 [(section_type, start_idx, end_idx), ...] 其中 section_type 为 "template" 或 "message"
            """
            sections: List[Tuple[str,int,int]] = []
            prev = 0
            for (s, e) in msg_ranges:
                if prev < s:
                    # 消息之间的部分属于 template
                    sections.append(("template", prev, s))
                sections.append(("message", s, e))
                prev = e
            if prev < total_len:
                # 最后一条消息之后的部分也属于 template
                sections.append(("template", prev, total_len))
            return sections
        slm_sections = build_sections(len(slm_ids), slm_msg_ranges)
        llm_sections = build_sections(len(llm_ids), llm_msg_ranges)
        assert len(slm_sections) == len(llm_sections), "Section count mismatch"

        # 步骤4：逐 section 对齐，构建输出序列
        slm_out: List[int] = []          # SLM 侧输出序列
        llm_out: List[int] = []          # LLM 侧输出序列
        mask_out: List[bool] = []        # message_mask: 标记消息内容位置
        slm_pad_mask_out: List[bool] = []  # SLM 侧 padding 掩码
        llm_pad_mask_out: List[bool] = []  # LLM 侧 padding 掩码
        detailed_sections: List[Dict[str, Union[str, Tuple[int,int]]]] = []

        for (stype_s, s_s, e_s), (stype_l, s_l, e_l) in zip(slm_sections, llm_sections):
            assert stype_s == stype_l, "Section type mismatch"
            slm_start_out = len(slm_out)
            llm_start_out = len(llm_out)
            if stype_s == "template":
                # template section: 两侧可能长度不同，用 padding 补齐到较长的一方
                slm_seg_len = e_s - s_s
                llm_seg_len = e_l - s_l
                target_len = slm_seg_len if slm_seg_len >= llm_seg_len else llm_seg_len
                slm_pad_needed = target_len - slm_seg_len  # SLM 侧需要补的 pad 数量
                llm_pad_needed = target_len - llm_seg_len  # LLM 侧需要补的 pad 数量
                # 原始 token + padding
                slm_seg = slm_ids[s_s:e_s] + [slm_pad_id] * slm_pad_needed
                llm_seg = llm_ids[s_l:e_l] + [llm_pad_id] * llm_pad_needed
                slm_out.extend(slm_seg)
                llm_out.extend(llm_seg)
                # template 部分 message_mask = False（不属于消息内容）
                mask_out.extend([False] * target_len)
                # 记录哪些位置是 padding（原始 token 为 False，填充的为 True）
                slm_pad_mask_out.extend([False] * slm_seg_len + [True] * slm_pad_needed)
                llm_pad_mask_out.extend([False] * llm_seg_len + [True] * llm_pad_needed)
            else:  # message section / 消息内容部分
                # 用 align_tokens 将 SLM token 逐一映射为 LLM token
                slm_msg = slm_ids[s_s:e_s]
                llm_msg = self.align_tokens(slm_msg)
                assert len(llm_msg) == len(slm_msg)  # 对齐后长度应一致
                slm_out.extend(slm_msg)
                llm_out.extend(llm_msg)
                # message 部分 message_mask = True（属于消息内容，需要参与 loss 计算）
                mask_out.extend([True] * len(slm_msg))
                # no padding in message sections / 消息内容部分无 padding
                slm_pad_mask_out.extend([False] * len(slm_msg))
                llm_pad_mask_out.extend([False] * len(slm_msg))
            slm_end_out = len(slm_out)
            llm_end_out = len(llm_out)
            detailed_sections.append({
                'type': stype_s,
                'slm_range': (slm_start_out, slm_end_out),
                'llm_range': (llm_start_out, llm_end_out)
            })

        # 步骤5：组装返回结果
        result_min = {
            'slm_ids_padded': slm_out,
            'llm_ids_padded': llm_out,
            'message_mask': mask_out,
            'slm_padding_mask': slm_pad_mask_out,
            'llm_padding_mask': llm_pad_mask_out
        }
        if return_details:
            result_min['sections'] = detailed_sections
            result_min['slm_text'] = slm_text
            result_min['llm_text'] = llm_text
        return result_min
