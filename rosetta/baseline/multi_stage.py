"""
Multi-stage evaluation utilities for VLM+LLM and LLM+LLM pipelines.
多阶段推理与评估工具模块，支持 VLM+LLM 和 LLM+LLM 两种流水线架构。

This module provides utilities for multi-stage evaluation where:
本模块提供多阶段推理的核心工具类，主要包括以下三种流水线：

1. VLM describes/analyzes images + LLM performs reasoning
   多模态流水线 (MultiModalInference)：VLM 先描述/分析图像，LLM 基于图像描述进行推理
2. LLM provides background context + LLM performs reasoning
   双LLM流水线 (TwoStageInference)：第一个 LLM 提供背景知识，第二个 LLM 基于背景知识回答问题

Core Classes / 核心类:
    - TwoStageInference: LLM+LLM 两阶段推理基类，stage1 由 context model 生成背景知识，
      stage2 由 answer model 结合背景知识回答问题。对应 C2C 框架中 sharer→receiver 的文本通信模式。
    - TwoStageRosetta(TwoStageInference): 继承自 TwoStageInference，但 stage2 使用 Rosetta 模型
      （即经过 C2C KV-Cache 投影/融合训练的 receiver 模型）替代普通 LLM。Rosetta 模型内部集成了
      Projector（KV 维度投影）和 Fuser（KV 融合），可直接利用 sharer 的 KV-Cache 进行推理。
    - MultiModalInference: VLM+LLM 多模态流水线，stage1 由 VLM (如 Qwen2.5-VL) 描述图像，
      stage2 由 LLM 基于图像描述回答问题。

Relationships / 与其他模块的关系:
    - rosetta.utils.evaluate: 提供 set_default_chat_template、apply_generation_config、
      load_rosetta_model 等工具函数
    - rosetta.baseline 下其他模块: 本文件作为 baseline 评测的基础设施，被上层评测脚本调用
    - transformers: 使用 HuggingFace 的 AutoModel / AutoTokenizer 加载模型
    - qwen_vl_utils: 可选依赖，用于处理 Qwen2.5-VL 的视觉输入
"""

# ===== 标准库与第三方依赖导入 =====
from typing import Dict, Optional, Any
import torch
from transformers import (
    # Qwen2_5_VLForConditionalGeneration,  # VLM 模型类（在下方 try/except 中延迟导入）
    AutoProcessor,       # 多模态处理器自动加载器（用于 VLM 的图像+文本预处理）
    AutoTokenizer,       # Tokenizer 自动加载器（用于文本编码/解码）
    AutoModelForCausalLM,  # 因果语言模型自动加载器（用于加载 LLM）
)
# 从 rosetta 工具模块导入：设置默认聊天模板、应用生成配置
from rosetta.utils.evaluate import set_default_chat_template, apply_generation_config

# 可选依赖：Qwen2.5-VL 视觉语言模型相关组件
# 如果未安装 qwen_vl_utils，仅会打印警告，不会阻断程序启动（因为仅 MultiModalInference 需要）
try:
    from qwen_vl_utils import process_vision_info           # 从消息列表中提取图像/视频输入
    from transformers import Qwen2_5_VLForConditionalGeneration  # Qwen2.5-VL 条件生成模型
except ImportError:
    print("Please install qwen-vl-utils to use VLM models")  # 提示用户安装可选依赖

class TwoStageInference:
    """
    Two-stage LLM+LLM inference pipeline for question answering.
    两阶段 LLM+LLM 问答推理流水线。

    工作原理（两阶段推理流程）：
        Stage 1 (Context Generation / 背景知识生成):
            - 使用 context model（sharer 角色）接收不含选项的原始问题
            - 模型生成与问题相关的背景知识文本（background context）
            - 对应 C2C 框架中 sharer 处理 prompt 产生 KV-Cache 的文本等价操作

        Stage 2 (Answer Generation / 答案生成):
            - 使用 answer model（receiver 角色）接收完整问题（含选项）
            - 将 stage1 生成的背景知识以对话上下文形式注入
            - 模型基于背景知识生成最终答案

    与 C2C 的关系：
        本类是 baseline 实现，两个 LLM 之间通过纯文本（background context）通信。
        相比之下，TwoStageRosetta 子类通过 KV-Cache 投影/融合实现更深层的模型间通信。

    Attributes / 属性:
        device (str): 计算设备，如 "cuda" 或 "cpu"
        max_new_tokens (int): 生成时的最大新 token 数
        background_prompt (str): stage1 使用的 prompt 模板，包含 {question} 占位符
        generation_config (dict): 模型生成时的额外配置参数
        context_model / context_tokenizer: stage1 使用的模型和分词器
        answer_model / answer_tokenizer: stage2 使用的模型和分词器
    """

    def __init__(
        self,
        context_model_path: str,
        answer_model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 1024,
        background_prompt: str = "Briefly describe the most useful background to solve the problem:\n\n{question}",
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize two-stage LLM pipeline.
        初始化两阶段 LLM 推理流水线。

        Args / 参数:
            context_model_path (str):
                提供背景知识的 LLM 模型路径（sharer 角色）。
                Path to context-providing LLM.
            answer_model_path (str):
                生成最终答案的 LLM 模型路径（receiver 角色）。
                Path to answer-generating LLM.
            device (str):
                计算设备，默认 "cuda"。
                Device to use.
            max_new_tokens (int):
                每次生成的最大新 token 数，默认 1024。
                Maximum number of new tokens to generate.
            background_prompt (str):
                stage1 的 prompt 模板，使用 {question} 占位符插入问题文本。
                Prompt template for background generation.
            generation_config (Optional[Dict[str, Any]]):
                可选的生成配置（如 temperature、top_p 等），会应用到两个模型上。
                Optional generation configuration to apply to models.
        """
        # 存储基本配置参数
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.background_prompt = background_prompt
        # 如果未提供 generation_config，使用空字典作为默认值
        self.generation_config = generation_config or {}
        # 加载两个模型（context model 和 answer model）
        self._load_models(context_model_path, answer_model_path)
    
    def _load_models(self, context_path: str, answer_path: str):
        """
        Load both LLM models.
        加载两个 LLM 模型（context model 和 answer model）。

        加载流程：
            1. 加载 context model 及其 tokenizer（用于 stage1 生成背景知识）
            2. 针对特殊模型（如 Gemma）进行额外配置
            3. 应用生成配置到 context model
            4. 加载 answer model 及其 tokenizer（用于 stage2 生成答案）
            5. 应用生成配置到 answer model

        Args / 参数:
            context_path (str): context LLM 的模型路径
            answer_path (str): answer LLM 的模型路径
        """
        # ===== 加载 Context LLM（Stage 1：背景知识生成模型）=====
        # 加载 context model 的 tokenizer
        self.context_tokenizer = AutoTokenizer.from_pretrained(context_path)
        # 针对 Gemma 模型的特殊处理：
        # Gemma 默认使用滑动窗口注意力（sliding window attention），这里手动设置为 4096
        # 同时需要增大 torch dynamo 的缓存大小限制，避免编译缓存溢出
        if context_path == "google/gemma-3-1b-it":
            # 增大 PyTorch Dynamo 编译缓存上限，防止 Gemma 模型编译时缓存不足
            torch._dynamo.config.cache_size_limit = 64
            self.context_model = AutoModelForCausalLM.from_pretrained(
                context_path, torch_dtype=torch.bfloat16, device_map={"": self.device}, sliding_window=4096
            )
        else:
            # 通用模型加载方式，使用 bfloat16 半精度以节省显存
            self.context_model = AutoModelForCausalLM.from_pretrained(
                context_path, torch_dtype=torch.bfloat16, device_map={"": self.device}
            )
        # 应用生成配置（如 temperature、top_p 等参数）到 context model
        apply_generation_config(self.context_model, self.generation_config)

        # ===== 加载 Answer LLM（Stage 2：答案生成模型）=====
        # 加载 answer model 的 tokenizer
        self.answer_tokenizer = AutoTokenizer.from_pretrained(answer_path)
        # 加载 answer model，同样使用 bfloat16 精度
        self.answer_model = AutoModelForCausalLM.from_pretrained(
            answer_path, torch_dtype=torch.bfloat16, device_map={"": self.device}
        )
        # 应用生成配置到 answer model
        apply_generation_config(self.answer_model, self.generation_config)
    
    def get_background_context(
        self,
        question: str,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        Get background context from the first LLM.
        使用 context model（stage1）生成背景知识文本。

        这是两阶段推理的 Stage 1：
            - 将问题文本填入 background_prompt 模板
            - 使用 chat template 构造对话格式的输入
            - context model 以 greedy decoding（do_sample=False）生成背景知识
            - 仅提取生成的部分（去掉输入 token），解码为文本返回

        Args / 参数:
            question (str):
                不含选项的原始问题文本。
                Question text (without options).
            max_new_tokens (Optional[int]):
                最大生成 token 数，None 时使用实例默认值。
                Max tokens to generate (uses instance default if None).

        Returns / 返回值:
            str: 生成的背景知识文本。
                 Background context string.
        """
        # 将问题文本填入 prompt 模板，生成完整的 stage1 输入
        prompt = self.background_prompt.format(question=question)
        # 构造单轮对话消息：仅 user 角色发送 prompt
        messages = [{"role": "user", "content": prompt}]

        # 禁用思考模式（thinking mode），某些模型（如 QwQ）支持此参数
        template_kwargs = {'enable_thinking': False}

        # 使用 tokenizer 的 chat template 将消息列表编码为模型输入
        # tokenize=True: 返回 token ID 张量
        # add_generation_prompt=True: 在末尾添加 assistant 回复的起始标记（引导模型生成）
        # return_tensors="pt": 返回 PyTorch 张量
        inputs = self.context_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            **template_kwargs
        )
        # 将输入张量移动到指定设备（如 GPU）
        inputs = inputs.to(self.device)

        # 使用实例默认的 max_new_tokens（如果未指定）
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # 在推理模式下生成（不计算梯度，节省内存和计算开销）
        with torch.inference_mode():
            outputs = self.context_model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False  # 使用 greedy decoding，确定性生成
            )

        # 仅保留生成的 token（去掉输入部分）
        # outputs 的 shape: [batch_size, input_len + generated_len]
        # inputs.shape[-1] 是输入长度，切片后只取生成部分
        generated_ids = outputs[:, inputs.shape[-1]:]
        # 将生成的 token ID 解码为文本
        # skip_special_tokens=True: 跳过 [EOS]、[PAD] 等特殊标记
        # clean_up_tokenization_spaces=False: 保留原始空格格式
        context = self.context_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return context
    
    def answer_with_context(
        self,
        question: str,
        context: str,
        max_new_tokens: Optional[int] = None,
        original_question: Optional[str] = None
    ) -> str:
        """
        Answer question using the second LLM with context.
        使用 answer model（stage2）结合背景知识生成答案。

        这是两阶段推理的 Stage 2：
            - 将 stage1 生成的背景知识以多轮对话形式注入
            - 如果提供了 original_question，构造三轮对话：
                user(问背景) → assistant(回答背景) → user(问完整问题)
              这种格式能让 answer model 更好地理解上下文的来源和目的
            - 如果未提供 original_question，退化为单轮对话：
                user(背景知识 + 问题) → assistant(回答)
            - 使用 greedy decoding 生成最终答案

        Args / 参数:
            question (str):
                包含选项的完整问题（使用合适的答题模板格式化）。
                Full question with options and proper template.
            context (str):
                stage1 生成的背景知识文本。
                Background context from first LLM.
            max_new_tokens (Optional[int]):
                最大生成 token 数，None 时使用实例默认值。
                Max tokens to generate (uses instance default if None).
            original_question (Optional[str]):
                stage1 中使用的原始问题（不含选项），用于构造多轮对话格式。
                Original question asked to first LLM (for conversation format).

        Returns / 返回值:
            str: 生成的答案文本。
                 Generated answer string.
        """
        # 构造对话消息列表
        if original_question:
            # 使用多轮对话格式（conversation format）：
            # 轮次1: user 请求背景知识（使用原始问题）
            # 轮次2: assistant 提供背景知识
            # 轮次3: user 提出完整问题（含选项），让 assistant 回答
            # 这种格式模拟了两个 LLM 之间的信息传递过程
            messages = [
                {"role": "user", "content": self.background_prompt.format(question=original_question)},
                {"role": "assistant", "content": context},
                {"role": "user", "content": question}
            ]
        else:
            # 退化为简单的单轮对话：将背景知识和问题拼接在一起
            # Fallback to simple format
            messages = [{"role": "user", "content": f"Background context: {context}\n\n{question}"}]

        # 禁用思考模式
        template_kwargs = {'enable_thinking': False}

        # 使用 answer tokenizer 的 chat template 编码输入
        inputs = self.answer_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # 添加 assistant 回复起始标记
            return_tensors="pt",
            **template_kwargs
        )
        inputs = inputs.to(self.device)

        # 使用实例默认的 max_new_tokens（如果未指定）
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # 推理模式下生成答案
        with torch.inference_mode():
            outputs = self.answer_model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False  # greedy decoding
            )

        # 截取生成的 token（去掉输入部分）并解码
        generated_ids = outputs[:, inputs.shape[-1]:]
        answer = self.answer_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return answer
    
    def forward_with_context(
        self,
        question: str,
        context: str,
        original_question: Optional[str] = None,
        response_text: Optional[str] = None,
        **forward_kwargs
    ) -> Any:
        """
        Run a forward pass on the answer model using provided context (logits mode).
        使用背景知识在 answer model 上执行一次前向传播（logits 模式）。

        与 answer_with_context 的区别：
            - answer_with_context: 调用 generate() 进行自回归生成，返回文本
            - forward_with_context: 调用 model(**inputs) 执行单次前向传播，返回 logits
              用于评估时计算下一个 token 的概率分布（如 logits-based 评测）

        response_text 参数的作用（steering next-token logits）：
            在 chat template 生成的文本之后追加 response_text，使得模型在该文本的末尾
            位置预测下一个 token 的 logits。常用于评测时检查模型对正确答案选项的
            概率是否高于其他选项。

        Args / 参数:
            question (str):
                包含选项的完整问题。
                Full question with options and proper template.
            context (str):
                stage1 生成的背景知识。
                Background context from first LLM.
            original_question (Optional[str]):
                stage1 的原始问题，用于多轮对话格式。
                Original question asked to first LLM (for conversation format).
            response_text (Optional[str]):
                可选的追加文本，附加在 chat template 之后，用于引导 next-token logits。
                例如，如果 response_text="The answer is"，模型将预测 "The answer is" 之后的
                下一个 token 的概率分布。
                Optional text to append after the chat template to steer next-token logits.
            **forward_kwargs:
                额外参数，直接传递给模型的 forward 方法。
                Extra kwargs forwarded to the model's forward.

        Returns / 返回值:
            ModelOutput: 模型前向传播的输出（通常包含 logits 等）。
                         Model outputs from the forward pass (e.g., logits).
        """
        # 构造多轮对话消息（与 answer_with_context 相同逻辑）
        if original_question:
            # 多轮对话格式：user→assistant→user
            messages = [
                {"role": "user", "content": self.background_prompt.format(question=original_question)},
                {"role": "assistant", "content": context},
                {"role": "user", "content": question}
            ]
        else:
            # 简单格式：背景知识 + 问题拼接
            # Fallback to simple format
            messages = [{"role": "user", "content": f"Background context: {context}\n\n{question}"}]

        template_kwargs = {'enable_thinking': False}

        # 构建模型输入
        # 如果提供了 response_text，将其追加到 chat template 生成的文本之后
        if response_text is not None:
            # 先以纯文本形式获取 chat template 的输出（tokenize=False）
            # Build raw text then append response_text
            text = self.answer_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs
            )
            # 在 chat template 文本末尾追加 response_text
            # 这样模型将在 response_text 的最后一个 token 处计算 next-token logits
            text = text + response_text
            # 对拼接后的完整文本进行 tokenize
            tokenized = self.answer_tokenizer(text, return_tensors="pt")
        else:
            # 不追加文本，直接构建标准输入（predict next assistant token）
            # 模型将预测 assistant 回复的第一个 token
            tokenized = self.answer_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **template_kwargs
            )

        # 将所有输入张量移动到指定设备
        inputs = {k: v.to(self.device) for k, v in tokenized.items()}

        # 推理模式下执行前向传播（不计算梯度）
        with torch.inference_mode():
            outputs = self.answer_model(**inputs, **forward_kwargs)

        return outputs

    def forward(
        self,
        question_without_options: str,
        question_with_options: str,
        response_text: Optional[str] = None,
        **forward_kwargs
    ) -> Any:
        """
        Two-stage forward pass (logits mode):
        1) Generate background context with the context model
        2) Run a forward pass on the answer model conditioned on that context

        两阶段前向传播（logits 模式）：
            1) 调用 get_background_context 生成背景知识（Stage 1）
            2) 调用 forward_with_context 在 answer model 上执行前向传播（Stage 2）

        这是 logits-based 评测的主入口方法，用于获取模型对各个选项 token 的 logits。

        Args / 参数:
            question_without_options (str):
                不含选项的问题文本，传给 stage1 的 context model。
                Question text without multiple choice options.
            question_with_options (str):
                含选项的完整问题，传给 stage2 的 answer model。
                Full question with options and proper template.
            response_text (Optional[str]):
                可选的追加文本，用于引导 next-token logits。
                Optional text appended after the chat template to steer next-token logits.
            **forward_kwargs:
                额外参数，传递给模型 forward。

        Returns / 返回值:
            ModelOutput: 模型前向传播输出（含 logits）。
        """
        # Stage 1: 使用 context model 生成背景知识
        context = self.get_background_context(question_without_options)
        # Stage 2: 使用 answer model 执行前向传播
        return self.forward_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options,
            response_text=response_text,
            **forward_kwargs
        )

    def logits_with_context(
        self,
        question_without_options: str,
        question_with_options: str,
        response_text: Optional[str] = None,
        **forward_kwargs
    ) -> Any:
        """
        Two-stage logits helper that also returns the generated background context
        for logging as CoT.

        两阶段 logits 辅助方法，与 forward() 类似，但同时返回背景知识文本。
        背景知识文本可用于日志记录，作为 Chain-of-Thought (CoT) 的中间推理步骤。

        Args / 参数:
            question_without_options (str): 不含选项的问题文本
            question_with_options (str): 含选项的完整问题
            response_text (Optional[str]): 可选的追加文本，用于引导 next-token logits
            **forward_kwargs: 传递给模型 forward 的额外参数

        Returns / 返回值:
            tuple: (outputs, context)
                - outputs: 模型前向传播输出（含 logits）
                - context: stage1 生成的背景知识文本（str），用于日志记录
        """
        # Stage 1: 生成背景知识
        context = self.get_background_context(question_without_options)
        # Stage 2: 在 answer model 上执行前向传播
        outputs = self.forward_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options,
            response_text=response_text,
            **forward_kwargs
        )
        # 同时返回模型输出和背景知识（背景知识可记录为 CoT 推理过程）
        return outputs, context

    def generate(
        self,
        question_without_options: str,
        question_with_options: str,
        communication_max_new_tokens: Optional[int] = None,
        response_max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate answer using two-stage processing.
        使用两阶段流程生成答案（文本生成模式）。

        这是 generate-based 评测的主入口方法：
            Stage 1: context model 生成背景知识（可独立控制 max_new_tokens）
            Stage 2: answer model 基于背景知识生成最终答案（可独立控制 max_new_tokens）

        与 forward() 的区别：
            - forward() 返回 logits（用于 logits-based 评测）
            - generate() 返回文本（用于 generate-based 评测）

        Args / 参数:
            question_without_options (str): 不含选项的问题文本
            question_with_options (str): 含选项的完整问题
            communication_max_new_tokens (Optional[int]):
                Stage 1 生成背景知识的最大 token 数。
                Maximum tokens to generate for the background context.
            response_max_new_tokens (Optional[int]):
                Stage 2 生成答案的最大 token 数。
                Maximum tokens to generate for the answer.
            **kwargs: 额外生成参数（为兼容性保留，当前被忽略）

        Returns / 返回值:
            str: 生成的答案文本。
        """
        # Stage 1: 使用 context model 生成背景知识
        # communication_max_new_tokens 控制背景知识的长度上限
        context = self.get_background_context(question_without_options, communication_max_new_tokens)

        # Stage 2: 使用 answer model 基于背景知识生成答案
        answer = self.answer_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options,
            max_new_tokens=response_max_new_tokens
        )

        return answer
    
    def process(
        self,
        question_without_options: str,
        question_with_options: str
    ) -> Dict[str, str]:
        """
        Full two-stage processing (legacy method for backward compatibility).
        完整两阶段处理流程（遗留方法，用于向后兼容）。

        与 generate() 类似，但额外返回背景知识文本。
        主要用于调试和可视化中间结果。

        Args / 参数:
            question_without_options (str): 不含选项的问题文本
            question_with_options (str): 含选项的完整问题

        Returns / 返回值:
            Dict[str, str]: 包含以下键值对的字典：
                - "context": stage1 生成的背景知识文本
                - "answer": stage2 生成的答案文本
        """
        # Stage 1: 生成背景知识
        context = self.get_background_context(question_without_options)

        # Stage 2: 基于背景知识生成答案
        answer = self.answer_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options
        )

        return {
            "context": context,  # 背景知识（stage1 输出）
            "answer": answer     # 最终答案（stage2 输出）
        }


class TwoStageRosetta(TwoStageInference):
    """
    Two-stage LLM+Rosetta inference pipeline for question answering.
    两阶段 LLM+Rosetta 推理问答流水线。

    继承自 TwoStageInference，但将 Stage 2 的 answer model 替换为 Rosetta 模型。
    Rosetta 是 C2C 框架中经过 KV-Cache 投影/融合训练的 receiver 模型，能够：
        - 接收来自 sharer (context model) 的 KV-Cache
        - 通过 Projector 将 sharer 的 KV 投影到 receiver 兼容的维度
        - 通过 Fuser 将投影后的 KV 与 receiver 自身 KV 融合
        - 基于融合后的 KV-Cache 生成回答

    与父类 TwoStageInference 的主要区别：
        1. Stage 2 使用 Rosetta 模型（而非普通 LLM）
        2. Rosetta 模型的输入格式不同，需要构造 kv_cache_index
           （用于标识每个 token 属于 instruction 还是 response）
        3. 支持 logits 模式和 generate 模式两种评测方式
        4. 需要加载额外的 Rosetta 配置（checkpoint 路径、投影器权重等）

    Attributes / 额外属性:
        rosetta_checkpoint_dir (str): Rosetta 模型 checkpoint 目录路径
        rosetta_subfolder (str): checkpoint 子目录名（如 "final"、"checkpoint-1000"）
        rosetta_model: 加载的 Rosetta 模型实例
        rosetta_tokenizer: Rosetta 模型的 tokenizer
        llm_tokenizer: 可选的 LLM tokenizer（用于 alignment 场景）
    """

    def __init__(
        self,
        context_model_path: str,
        rosetta_checkpoint_dir: str,
        rosetta_subfolder: str = "final",
        device: str = "cuda",
        max_new_tokens: int = 1024,
        background_prompt: str = "Briefly describe the most useful background to solve the problem:\n\n{question}",
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize two-stage pipeline with Rosetta as second model.
        初始化以 Rosetta 作为 Stage 2 模型的两阶段推理流水线。

        Args / 参数:
            context_model_path (str):
                context LLM 模型路径（sharer 角色）。
                Path to context-providing LLM.
            rosetta_checkpoint_dir (str):
                Rosetta checkpoint 目录路径，包含 config.json 和模型权重。
                Path to Rosetta checkpoint directory.
            rosetta_subfolder (str):
                checkpoint 子目录名，默认为 "final"。
                Subfolder name in checkpoint directory (e.g., 'final', 'checkpoint-1000').
            device (str): 计算设备，默认 "cuda"
            max_new_tokens (int): 最大生成 token 数，默认 1024
            background_prompt (str): stage1 的 prompt 模板
            generation_config (Optional[Dict[str, Any]]): 可选的生成配置
        """
        # 调用父类 __init__，但传入 None 作为 answer_model_path
        # 因为我们会覆盖 answer model 的加载逻辑（使用 Rosetta 替代）
        # Initialize parent class with dummy answer model path
        # We'll override the answer model loading
        super().__init__(
            context_model_path=context_model_path,
            answer_model_path=None,  # Will be overridden / 占位符，后续被覆盖
            device=device,
            max_new_tokens=max_new_tokens,
            background_prompt=background_prompt,
            generation_config=generation_config
        )

        # 保存 Rosetta checkpoint 相关路径
        self.rosetta_checkpoint_dir = rosetta_checkpoint_dir
        self.rosetta_subfolder = rosetta_subfolder
        # 加载 Rosetta 模型（包含 base model、projector、fuser 等组件）
        self._load_rosetta_model()
    
    def _load_models(self, context_path: str, answer_path: str):
        """
        Override parent class _load_models to prevent loading dummy answer model.
        We only load the context model here, and the Rosetta model is loaded separately.

        覆盖父类的 _load_models 方法，仅加载 context model。
        Answer model 由 _load_rosetta_model() 单独加载 Rosetta 模型替代。

        Args / 参数:
            context_path (str): context LLM 的模型路径
            answer_path (str): 未使用（传入 None），仅为保持接口一致
        """
        # ===== 仅加载 Context LLM（answer model 被 Rosetta 替代）=====
        self.context_tokenizer = AutoTokenizer.from_pretrained(context_path)
        self.context_model = AutoModelForCausalLM.from_pretrained(
            context_path, torch_dtype=torch.bfloat16, device_map={"": self.device}
        )
        # 应用生成配置到 context model
        apply_generation_config(self.context_model, self.generation_config)

        # 跳过 answer model 加载 —— 使用 Rosetta 模型替代
        # Skip loading answer model - we use Rosetta instead
        print(f"Loaded context model from {context_path}")
        print("Skipping answer model loading - using Rosetta model instead")
    
    def _load_rosetta_model(self):
        """
        Load Rosetta model and related components following load_model_from_checkpoint pattern.
        加载 Rosetta 模型及其相关组件。

        加载流程：
            1. 从 checkpoint 目录读取 config.json 配置文件
            2. 验证子目录中是否存在 projector 权重文件（projector_*.pt）
            3. 构造 model_config 和 eval_config，调用 load_rosetta_model 工具函数加载模型
            4. 如果需要 alignment（token 对齐），额外加载 teacher model 的 tokenizer

        Rosetta checkpoint 目录结构示例：
            checkpoint_dir/
            ├── config.json              # 训练配置（含 base_model、teacher_model 路径等）
            └── final/                   # rosetta_subfolder 指定的子目录
                ├── projector_0.pt       # 第 0 层的 KV 投影器权重
                ├── projector_1.pt       # 第 1 层的 KV 投影器权重
                ├── ...
                ├── fuser_0.pt           # 第 0 层的 KV 融合器权重
                └── base_model/          # receiver 基座模型权重
        """
        import json
        from pathlib import Path
        from rosetta.utils.evaluate import load_rosetta_model  # 导入 Rosetta 模型加载工具

        checkpoint_path = Path(self.rosetta_checkpoint_dir)

        # 步骤 1: 加载配置文件
        config_path = checkpoint_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # 步骤 2: 验证是否为合法的 Rosetta checkpoint（检查是否存在 projector 文件）
        # Projector 是 C2C 框架的核心组件，负责将 sharer 的 KV 维度投影到 receiver 兼容维度
        subfolder_dir = checkpoint_path / self.rosetta_subfolder
        # 检查子目录中是否有 projector_*.pt 格式的权重文件
        has_projectors = subfolder_dir.exists() and any(
            f.name.startswith("projector_") and f.name.endswith(".pt")
            for f in subfolder_dir.iterdir()
        )

        if not has_projectors:
            raise ValueError(f"No projectors found in {subfolder_dir}. This doesn't appear to be a Rosetta checkpoint.")

        # 步骤 3: 构造模型配置，调用 load_rosetta_model 加载完整 Rosetta 模型
        print(f"Loading Rosetta model from {self.rosetta_checkpoint_dir}")

        # 构造 model_config：包含 Rosetta 加载所需的所有配置信息
        # - base_model: receiver 基座模型的路径/名称
        # - teacher_model: sharer (teacher) 模型的路径/名称
        # - is_do_alignment: 是否启用 token 对齐（用于不同 tokenizer 之间的对齐）
        # - alignment_strategy: 对齐策略（如 "first" 表示按首 token 对齐）
        model_config = {
            "model_name": "Rosetta",
            "rosetta_config": {
                "checkpoints_dir": str(subfolder_dir),
                "base_model": config["model"]["base_model"],
                "teacher_model": config["model"]["teacher_model"],
                "is_do_alignment": config["model"].get("is_do_alignment", False),
                "alignment_strategy": config["model"].get("alignment_strategy", "first")
            }
        }

        print(f"Model config: {model_config}")

        # eval_config: 评测相关配置，目前仅需指定 checkpoint 路径
        eval_config = {
            "checkpoints_dir": str(subfolder_dir)
        }

        # 使用 load_rosetta_model 工具函数加载模型
        # 返回 Rosetta 模型实例和对应的 tokenizer
        # Rosetta 模型内部集成了：base model (receiver) + projectors + fusers
        self.rosetta_model, self.rosetta_tokenizer = load_rosetta_model(
            model_config,
            eval_config,
            device=self.device
        )

        # 步骤 4: 如果启用了 alignment，加载 LLM (teacher) 的 tokenizer
        # Alignment 用于处理 sharer 和 receiver 使用不同 tokenizer 的场景
        # 此时需要 teacher tokenizer 来对齐 token 边界
        is_do_alignment = config["model"].get("is_do_alignment", False)
        llm_model_path = config["model"].get("teacher_model")
        self.llm_tokenizer = None

        if is_do_alignment and llm_model_path:
            try:
                # 加载 teacher model 的 tokenizer
                self.llm_tokenizer = AutoTokenizer.from_pretrained(str(llm_model_path))
                # 确保 pad_token 已设置（某些模型默认没有 pad_token，用 eos_token 替代）
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                # 设置默认的 chat template
                set_default_chat_template(self.llm_tokenizer, llm_model_path)
            except Exception as e:
                print(f"Failed to load LLM tokenizer '{llm_model_path}': {e}")
                self.llm_tokenizer = None

        print(f"Initialized TwoStageRosetta with Rosetta model on {self.device}")
    
    def _prepare_rosetta_inputs(
        self,
        question: str,
        context: str,
        original_question: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        answer_method: str = "generate",
        response_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare inputs for Rosetta model using the simpler approach from live_chat_example.py.
        为 Rosetta 模型准备输入数据，核心是构造 kv_cache_index。

        本方法是 Rosetta 推理的关键：它将普通文本输入转换为 Rosetta 模型所需的特殊格式。
        最关键的是 kv_cache_index 的构造——它告诉 Rosetta 模型每个 token 在 KV-Cache
        操作中应如何处理：
            - [1, 0]: instruction token（指令/上下文 token），其 KV 应被缓存用于后续注意力计算
            - [-1, 0]: response/label token（响应/标签 token），其 KV 的处理方式不同
              （通常不参与 KV-Cache 存储，或仅用于计算 loss）

        kv_cache_index 的格式：
            - 是一个列表 (list)，每个元素是一个张量 (tensor)
            - 每个张量的 shape 为 [batch_size, seq_len, 2]
            - 列表中的每个张量对应一个"段"（segment），如 instruction 段和 response 段
            - 两元素向量 [a, b] 的含义：
                * [1, 0]: 该 token 属于 instruction 段（缓存 KV，用于注意力）
                * [-1, 0]: 该 token 属于 response/label 段（不缓存 KV，或特殊处理）

        Args / 参数:
            question (str): 要回答的问题
            context (str): stage1 生成的背景知识
            original_question (Optional[str]): stage1 使用的原始问题（用于多轮对话格式）
            max_new_tokens (Optional[int]): 最大生成 token 数
            answer_method (str): 回答方式，"generate"（文本生成）或 "logits"（logits 计算）
            response_text (Optional[str]): logits 模式下的追加文本

        Returns / 返回值:
            Dict[str, Any]: 包含以下键的字典：
                - "inputs": dict，含 input_ids, attention_mask, position_ids, kv_cache_index
                - "printable_text": str，可打印的输入文本（用于调试/日志）
        """
        # ===== 步骤 1: 构造对话消息列表 =====
        # 使用多轮对话格式：user(问背景) → assistant(回答背景) → user(问完整问题)
        if original_question:
            messages = [
                {"role": "user", "content": self.background_prompt.format(question=original_question)},
                {"role": "assistant", "content": context},
                {"role": "user", "content": question}
            ]
        else:
            # Fallback to simple format / 退化为简单格式
            messages = [{"role": "user", "content": f"Background context: {context}\n\n{question}"}]

        # ===== 步骤 2: 应用 chat template 生成纯文本 =====
        # 将消息列表转换为模型可理解的纯文本格式（不 tokenize）
        base_text = None
        if hasattr(self.rosetta_tokenizer, 'apply_chat_template'):
            # 使用 tokenizer 内置的 chat template
            base_text = self.rosetta_tokenizer.apply_chat_template(
                messages,
                tokenize=False,           # 先不 tokenize，后续再处理
                add_generation_prompt=True,  # 添加 assistant 回复起始标记
                enable_thinking=False     # 禁用思考模式
            )
        else:
            # 如果 tokenizer 不支持 chat template，使用手动格式
            base_text = f"### Human: {question}\n### Assistant:"

        # ===== 步骤 3: 可选地追加 response_text（仅 logits 模式）=====
        # response_text 追加在 chat template 文本之后，用于引导模型在特定位置计算 logits
        if answer_method == 'logits' and response_text is not None:
            text = base_text + response_text
        else:
            text = base_text

        # ===== 步骤 4: Tokenize 输入文本 =====
        # 将文本转换为 token ID 张量
        inputs = self.rosetta_tokenizer(text, return_tensors="pt").to(self.device)

        # ===== 步骤 5: 构造 kv_cache_index（核心 KV-Cache 操作）=====
        # kv_cache_index 是 Rosetta 模型的特有输入，用于控制 KV-Cache 的行为
        # 它将输入序列划分为不同的"段"（segment），每段有不同的 KV-Cache 处理策略
        full_length = inputs.input_ids.shape[1]  # 输入序列的总长度
        if answer_method == 'logits':
            # ----- Logits 模式的 kv_cache_index 构造 -----
            # 需要区分 base_text 部分和 response_text 部分
            # Compute response length as the extra tokens appended by response_text
            if response_text is not None:
                # 计算 response_text 对应的 token 数量：
                # 先对 base_text 单独 tokenize，然后用总长度减去 base 长度
                base_tok = self.rosetta_tokenizer(base_text, return_tensors="pt")
                response_length = int(inputs.input_ids.shape[1] - base_tok.input_ids.shape[1])
                response_length = max(response_length, 0)  # 防止负数
            else:
                response_length = 0
            # instruction 段长度 = 总长度 - response 段长度
            instr_len = max(full_length - response_length, 0)
            # 构造 instruction 段的 kv_cache_index：
            # [1, 0] 表示这些 token 是 instruction/上下文 token
            # 它们的 KV 会被缓存，在后续注意力计算中被查询
            # shape: [1, instr_len, 2]（batch=1, seq_len=instr_len, feature_dim=2）
            instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(instr_len, 1).unsqueeze(0).to(self.device)
            if response_length > 0:
                # 构造 response 段的 kv_cache_index：
                # [-1, 0] 表示这些 token 是 response/标签 token
                # 它们的 KV 通常不被缓存（或仅用于 loss 计算）
                # shape: [1, response_length, 2]
                response_index = torch.tensor([-1, 0], dtype=torch.long).repeat(response_length, 1).unsqueeze(0).to(self.device)
                # kv_cache_list 包含两个段：[instruction段, response段]
                kv_cache_list = [instruction_index, response_index]
            else:
                # 没有 response 段时，仅包含 instruction 段
                kv_cache_list = [instruction_index]
        else:
            # ----- Generate 模式的 kv_cache_index 构造 -----
            # 将输入序列的最后一个 token 视为 response/label（长度=1）
            # 前面的所有 token 视为 instruction
            # instruction 段：前 full_length-1 个 token，标记为 [1, 0]
            # shape: [1, full_length-1, 2]
            instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(full_length - 1, 1).unsqueeze(0).to(self.device)
            # label 段：最后 1 个 token，标记为 [-1, 0]
            # 这个 token 是模型要预测的第一个 token（即 assistant 回复的第一个 token）
            # shape: [1, 1, 2]
            label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(self.device)
            # kv_cache_list: [instruction段, label段]
            kv_cache_list = [instruction_index, label_index]

        # ===== 步骤 6: 构造 position_ids =====
        # position_ids 用于 Transformer 的位置编码
        if inputs.attention_mask is None:
            # 没有 attention_mask 时，使用连续的位置 ID: [0, 1, 2, ..., seq_len-1]
            position_ids = torch.arange(inputs.input_ids.shape[-1], dtype=torch.long).unsqueeze(0).to(self.device)
        else:
            # 有 attention_mask 时，通过累积求和计算位置 ID
            # 这样可以正确处理 padding token（padding 位置的 position_id 不会递增）
            # 例如：mask=[1,1,1,0,0,1] -> cumsum=[1,2,3,3,3,4] -> position=[0,1,2,2,2,3]
            position_ids = inputs.attention_mask.long().cumsum(-1) - 1

        # ===== 步骤 7: 组装输出字典 =====
        outputs = {
            "inputs": {
                "input_ids": inputs.input_ids,          # token ID 张量，shape: [1, seq_len]
                "attention_mask": inputs.attention_mask, # 注意力掩码，shape: [1, seq_len]
                "position_ids": position_ids,            # 位置 ID，shape: [1, seq_len]
                "kv_cache_index": kv_cache_list          # KV-Cache 索引列表（核心！）
            },
            "printable_text": text  # 可打印的文本（用于调试和日志）
        }

        return outputs
    
    def answer_with_context(
        self,
        question: str,
        context: str,
        max_new_tokens: Optional[int] = None,
        original_question: Optional[str] = None
    ) -> str:
        """
        Answer question using Rosetta model with context.
        Overrides parent class method to use Rosetta instead of regular LLM.

        使用 Rosetta 模型（stage2）结合背景知识生成答案。
        覆盖父类方法，使用 Rosetta 模型替代普通 LLM。

        与父类 answer_with_context 的区别：
            - 父类：使用普通 LLM 的 generate() 方法
            - 本类：使用 Rosetta 模型的 generate() 方法，需要传入 kv_cache_index
              Rosetta 模型内部会利用 KV-Cache 投影/融合机制来生成回答

        Args / 参数:
            question (str): 要回答的问题
            context (str): stage1 生成的背景知识
            max_new_tokens (Optional[int]): 最大生成 token 数
            original_question (Optional[str]): stage1 的原始问题（用于多轮对话格式）

        Returns / 返回值:
            str: 生成的答案文本
        """
        # 调用 _prepare_rosetta_inputs 准备 Rosetta 模型的输入
        # 包括 input_ids、attention_mask、position_ids、kv_cache_index
        prepared = self._prepare_rosetta_inputs(
            question=question,
            context=context,
            original_question=original_question,
            max_new_tokens=max_new_tokens
        )

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # 生成参数（greedy decoding）
        # Generation parameters (following live_chat_example.py pattern)
        sampling_params = {
            'do_sample': False,       # 不使用采样，确定性生成
            'max_new_tokens': max_new_tokens  # 最大生成 token 数
        }

        # 记录输入长度，用于后续截取生成的 token
        # Generate using Rosetta model (following live_chat_example.py pattern)
        input_length = prepared['inputs']['input_ids'].shape[1]

        # 推理模式下使用 Rosetta 模型生成
        # 注意：Rosetta 的 generate 接口与普通 HuggingFace 模型不同，
        # 需要额外传入 kv_cache_index 来控制 KV-Cache 的行为
        with torch.inference_mode():
            outputs = self.rosetta_model.generate(
                kv_cache_index=prepared['inputs']['kv_cache_index'],  # KV-Cache 索引（核心！）
                input_ids=prepared['inputs']['input_ids'],            # 输入 token ID
                attention_mask=prepared['inputs']['attention_mask'],   # 注意力掩码
                position_ids=prepared['inputs']['position_ids'],       # 位置编码
                **sampling_params
            )
            # outputs[0] 是生成的 token ID 序列（包含输入+生成）
            generated_ids = outputs[0]

        # 解码生成的文本：仅取 input_length 之后的部分（即新生成的 token）
        # Decode response
        answer = self.rosetta_tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True).strip()

        return answer

    def forward_with_context(
        self,
        question: str,
        context: str,
        original_question: Optional[str] = None,
        response_text: Optional[str] = None,
        **forward_kwargs
    ) -> Any:
        """
        Run a forward pass on the Rosetta model using provided context (logits mode).

        在 Rosetta 模型上执行前向传播（logits 模式）。
        与 answer_with_context 类似，但返回 logits 而非生成文本。

        Args / 参数:
            question (str): 含选项的完整问题
            context (str): stage1 生成的背景知识
            original_question (Optional[str]): stage1 的原始问题
            response_text (Optional[str]): 追加文本，用于引导 next-token logits
            **forward_kwargs: 传递给模型 forward 的额外参数

        Returns / 返回值:
            ModelOutput: 模型前向传播输出（含 logits）
        """
        # 准备 Rosetta 模型输入（logits 模式）
        prepared = self._prepare_rosetta_inputs(
            question=question,
            context=context,
            original_question=original_question,
            answer_method='logits',       # 指定 logits 模式
            response_text=response_text
        )

        inputs = prepared['inputs']
        # 推理模式下执行前向传播
        # Rosetta 的 forward 接口需要 kv_cache_index 来控制 KV-Cache 操作
        with torch.inference_mode():
            outputs = self.rosetta_model.forward(
                kv_cache_index=inputs['kv_cache_index'],  # KV-Cache 索引
                input_ids=inputs['input_ids'],            # 输入 token ID
                attention_mask=inputs['attention_mask'],   # 注意力掩码
                position_ids=inputs['position_ids'],       # 位置编码
                **forward_kwargs
            )
        return outputs

    def forward(
        self,
        question_without_options: str,
        question_with_options: str,
        response_text: Optional[str] = None,
        **forward_kwargs
    ) -> Any:
        """
        Two-stage forward pass (logits mode) for Rosetta:
        1) Generate background context with the context model
        2) Run a forward pass on the Rosetta model conditioned on that context

        Rosetta 的两阶段前向传播（logits 模式）：
            1) 使用 context model 生成背景知识（Stage 1）
            2) 在 Rosetta 模型上执行前向传播（Stage 2）

        注意：当前此方法尚未实现（raise NotImplementedError），标记为 Work in progress。

        Args / 参数:
            question_without_options (str): 不含选项的问题文本
            question_with_options (str): 含选项的完整问题
            response_text (Optional[str]): 追加文本，用于引导 next-token logits
            **forward_kwargs: 传递给模型 forward 的额外参数

        Returns / 返回值:
            ModelOutput: 模型前向传播输出（含 logits）
        """
        # Work in progress / 开发中，尚未完成
        raise NotImplementedError
        # 以下代码在 NotImplementedError 之后，当前不会执行
        context = self.get_background_context(question_without_options)
        return self.forward_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options,
            response_text=response_text,
            **forward_kwargs
        )

    def logits_with_context(
        self,
        question_without_options: str,
        question_with_options: str,
        response_text: Optional[str] = None,
        **forward_kwargs
    ) -> Any:
        """
        Two-stage logits helper that also returns the generated background context
        for logging as CoT (Rosetta backend).

        两阶段 logits 辅助方法（Rosetta 后端），同时返回背景知识文本。
        与父类的 logits_with_context 逻辑相同，但内部使用 Rosetta 模型的 forward。

        Returns / 返回值:
            tuple: (outputs, context)
                - outputs: Rosetta 模型前向传播输出
                - context: stage1 生成的背景知识文本
        """
        # Stage 1: 生成背景知识（使用父类方法，因为 context model 是普通 LLM）
        context = self.get_background_context(question_without_options)
        # Stage 2: 在 Rosetta 模型上执行前向传播
        outputs = self.forward_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options,
            response_text=response_text,
            **forward_kwargs
        )
        return outputs, context

    def generate(
        self,
        question_without_options: str,
        question_with_options: str,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate answer using two-stage processing with Rosetta.

        使用 Rosetta 的两阶段处理生成答案。
        Stage 1 使用 context model（普通 LLM），Stage 2 使用 Rosetta 模型。

        Args / 参数:
            question_without_options (str): 不含选项的问题文本
            question_with_options (str): 含选项的完整问题
            max_new_tokens (Optional[int]): 最大生成 token 数（两个阶段共用）
            **kwargs: 额外参数（为兼容性保留）

        Returns / 返回值:
            str: 生成的答案文本
        """
        # Stage 1: 使用 context model 生成背景知识（继承自父类）
        # Stage 1: Get background context (uses parent class method)
        context = self.get_background_context(question_without_options, max_new_tokens)

        # Stage 2: 使用 Rosetta 模型基于背景知识生成答案
        # Stage 2: Answer question with context using Rosetta
        answer = self.answer_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options,
            max_new_tokens=max_new_tokens
        )

        return answer

    def process(
        self,
        question_without_options: str,
        question_with_options: str
    ) -> Dict[str, str]:
        """
        Full two-stage processing with Rosetta (legacy method for backward compatibility).

        使用 Rosetta 的完整两阶段处理（遗留方法，用于向后兼容）。
        与 generate() 类似，但额外返回背景知识文本。

        Args / 参数:
            question_without_options (str): 不含选项的问题文本
            question_with_options (str): 含选项的完整问题

        Returns / 返回值:
            Dict[str, str]: 包含 "context" 和 "answer" 的字典
        """
        # Stage 1: 生成背景知识（使用父类方法）
        # Stage 1: Get background context (uses parent class method)
        context = self.get_background_context(question_without_options)

        # Stage 2: 使用 Rosetta 模型生成答案
        # Stage 2: Answer question with context using Rosetta
        answer = self.answer_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options
        )

        return {
            "context": context,  # 背景知识
            "answer": answer     # 最终答案
        }



class MultiModalInference:
    """
    Multi-modal VLM+LLM inference pipeline.
    多模态 VLM+LLM 推理流水线。

    工作原理（两阶段多模态推理流程）：
        Stage 1 (Image Description / 图像描述):
            - 使用 VLM（视觉语言模型，如 Qwen2.5-VL）接收图像和描述 prompt
            - VLM 生成对图像的详细描述文本
        Stage 2 (Answer Generation / 答案生成):
            - 使用 LLM 接收图像描述（作为上下文）和问题
            - LLM 基于图像描述回答问题

    与 TwoStageInference 的区别：
        - TwoStageInference 的 stage1 使用 LLM 生成背景知识（纯文本）
        - MultiModalInference 的 stage1 使用 VLM 描述图像（多模态输入）

    Attributes / 属性:
        device (str): 计算设备
        max_new_tokens (int): 最大生成 token 数
        generation_config (dict): 生成配置
        vlm_model / vlm_processor: stage1 使用的 VLM 模型和处理器
        llm_model / llm_tokenizer: stage2 使用的 LLM 模型和分词器
    """

    def __init__(
        self,
        vlm_model_path: str,
        llm_model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 1024,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-modal pipeline.
        初始化多模态推理流水线。

        Args / 参数:
            vlm_model_path (str):
                VLM 模型路径（如 Qwen2.5-VL）。
                Path to VLM model.
            llm_model_path (str):
                LLM 模型路径。
                Path to LLM model.
            device (str): 计算设备，默认 "cuda"
            max_new_tokens (int): 最大生成 token 数，默认 1024
            generation_config (Optional[Dict[str, Any]]): 可选的生成配置
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.generation_config = generation_config or {}
        # 加载 VLM 和 LLM 模型
        self._load_models(vlm_model_path, llm_model_path)
    
    def _load_models(self, vlm_path: str, llm_path: str):
        """
        Load VLM and LLM models.
        加载 VLM 和 LLM 模型。

        Args / 参数:
            vlm_path (str): VLM 模型路径（如 Qwen2.5-VL）
            llm_path (str): LLM 模型路径
        """
        # ===== 加载 VLM（视觉语言模型，Stage 1）=====
        # 使用 Qwen2.5-VL 的条件生成模型类加载
        # Load VLM
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_path,
            torch_dtype=torch.bfloat16,  # 使用 bfloat16 半精度
            device_map={"": self.device},  # 将模型放在指定设备上
        )
        # 应用生成配置到 VLM 模型
        # Apply generation config to VLM model
        apply_generation_config(self.vlm_model, self.generation_config)
        # 加载 VLM 的多模态处理器（处理图像+文本输入）
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_path)

        # ===== 加载 LLM（语言模型，Stage 2）=====
        # Load LLM
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=torch.bfloat16, device_map={"": self.device}
        )
        # 应用生成配置到 LLM 模型
        # Apply generation config to LLM model
        apply_generation_config(self.llm_model, self.generation_config)

    def get_image_description(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail.",
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        Get image description from VLM.
        使用 VLM 获取图像描述（Stage 1）。

        这是多模态流水线的 Stage 1：
            - 构造包含图像和文本的多模态消息
            - 使用 process_vision_info 提取图像输入
            - VLM 处理器将图像和文本编码为模型输入
            - VLM 生成图像描述文本

        Args / 参数:
            image_path (str): 图像文件路径
            prompt (str): 描述 prompt，默认 "Describe this image in detail."
            max_new_tokens (Optional[int]): 最大生成 token 数

        Returns / 返回值:
            str: VLM 生成的图像描述文本
        """
        # 构造多模态消息：包含图像和文本
        # 使用 Qwen2.5-VL 的消息格式，content 是一个列表，包含图像和文本元素
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},  # 图像输入
                {"type": "text", "text": prompt}          # 文本 prompt
            ]
        }]

        # 使用 VLM processor 的 chat template 生成文本部分
        text = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 从消息列表中提取图像和视频输入（process_vision_info 来自 qwen_vl_utils）
        # 该函数会加载图像文件并预处理为模型可接受的格式
        image_inputs, video_inputs = process_vision_info(messages)
        # 使用 VLM processor 将文本、图像、视频编码为模型输入张量
        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,    # 预处理后的图像张量
            videos=video_inputs,    # 预处理后的视频张量（如果有）
            padding=True,           # 启用 padding 对齐
            return_tensors="pt",    # 返回 PyTorch 张量
        )
        # 将输入移动到指定设备
        inputs = inputs.to(self.device)

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # 推理模式下使用 VLM 生成图像描述
        with torch.inference_mode():
            outputs = self.vlm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy decoding
            )

        # 仅保留生成的 token（去掉输入部分）并解码
        # outputs 的 shape: [batch_size, input_len + generated_len]
        generated_ids = outputs[:, inputs["input_ids"].shape[-1]:]
        description = self.vlm_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return description

    def answer_with_context(
        self,
        question: str,
        context: str,
        max_new_tokens: Optional[int] = None,
        original_question: Optional[str] = None
    ) -> str:
        """
        Answer question using LLM with context.
        使用 LLM 结合上下文（如图像描述）生成答案（Stage 2）。

        与 TwoStageInference 的 answer_with_context 逻辑基本相同，
        区别在于 context 来源是 VLM 的图像描述而非 LLM 的背景知识。

        Args / 参数:
            question (str): 要回答的问题
            context (str): 上下文（如 VLM 生成的图像描述）
            max_new_tokens (Optional[int]): 最大生成 token 数
            original_question (Optional[str]): 原始问题（用于多轮对话格式）

        Returns / 返回值:
            str: 生成的答案文本
        """
        # 构造对话消息列表
        # Use conversation format: user asks about image, assistant describes, user asks follow-up
        if original_question:
            # 多轮对话格式：user(问图像) → assistant(描述图像) → user(追问)
            messages = [
                {"role": "user", "content": original_question},
                {"role": "assistant", "content": context},
                {"role": "user", "content": question}
            ]
        else:
            # Fallback to simple format / 退化为简单格式
            messages = [{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}]

        template_kwargs = {'enable_thinking': False}

        # 使用 LLM tokenizer 的 chat template 编码输入
        # Some tokenizers may not support enable_thinking parameter
        inputs = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            **template_kwargs
        )

        # 将输入移动到 LLM 模型所在设备
        inputs = inputs.to(self.llm_model.device)

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # 推理模式下生成答案
        with torch.inference_mode():
            outputs = self.llm_model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False  # greedy decoding
            )

        # 截取生成的 token 并解码
        generated_ids = outputs[:, inputs.shape[-1]:]
        answer = self.llm_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return answer

    def process(
        self,
        image_path: str,
        question: str,
        description_prompt: str = "Briefly describe this image."
    ) -> Dict[str, str]:
        """
        Full multi-stage processing.
        完整的多阶段多模态处理流程。

        执行流程：
            Stage 1: VLM 描述图像
            Stage 2: LLM 基于图像描述回答问题

        Args / 参数:
            image_path (str): 图像文件路径
            question (str): 要回答的问题
            description_prompt (str): 图像描述的 prompt

        Returns / 返回值:
            Dict[str, str]: 包含以下键值对的字典：
                - "description": VLM 生成的图像描述
                - "answer": LLM 生成的答案
        """
        # Stage 1: 使用 VLM 获取图像描述
        # Stage 1: Get image description
        description = self.get_image_description(image_path, description_prompt)

        # Stage 2: 使用 LLM 基于图像描述回答问题
        # 传入原始 description_prompt 以构造多轮对话格式
        # Stage 2: Answer question with context (pass original prompt for conversation format)
        answer = self.answer_with_context(
            question=question,
            context=description,
            original_question=description_prompt
        )

        return {
            "description": description,  # 图像描述（stage1 输出）
            "answer": answer             # 最终答案（stage2 输出）
        }