"""
数据集适配器模块 (Dataset Adapters)
====================================

本文件是 Rosetta/C2C 训练管线中的**数据层**核心模块，负责将各种异构数据集
（MMLU、LongBench、OpenHermes、Dolly 等）统一转换为标准的 chat message 格式，
再经过 tokenize 和 collate 后送入模型训练。

整体数据流:
    原始数据集 --> InstructionDataset（转为 chat messages）
                --> ChatDataset（tokenize + 生成 kv_cache_index / labels）
                --> DataCollator（batch padding + 分 section 对齐）
                --> 模型输入

核心组件:
    1. 数据集注册系统 (DATASET_REGISTRY + @register_dataset):
       通过装饰器自动注册数据集类，支持按名称工厂化创建。
    2. 批式过滤器 (batch filters):
       create_text_length_filter  -- 按词数/token 数过滤
       create_field_value_filter -- 按字段值精确/包含匹配过滤
       create_modulo_filter      -- 按 ID 取模过滤（用于数据划分）
       create_conversation_length_filter -- 按对话轮数过滤
    3. 文本提取函数 (extract_*):
       从不同格式的数据样本中提取出统一文本或 chat messages。
    4. Instruction 数据集类:
       LongBenchChatDataset, MMLUChatDataset, MMLUCotChatDataset,
       LLMGeneratedChatDataset, OpenBookChatDataset, OpenHermesChatDataset
       —— 各自加载原始数据并转换为 [{role, content}, ...] 的 message 列表。
    5. Tokenize 层数据集:
       ChatDataset           -- 单模型 tokenize + kv_cache_index 生成
       AlignedChatDataset    -- 双模型（SLM/LLM）token 对齐版本
       BaselineChatDataset   -- 基线模型（不含 KV-Cache 索引）
    6. Data Collator:
       RosettaDataCollator   -- 支持多模型、按 kv_cache_index 分段 padding
       BaselineDataCollator  -- 简单 padding，用于基线模型
    7. KV-Cache 索引生成:
       generate_kv_cache_index -- 为每个 token 标记其 KV-Cache 角色
       (instruction 部分标记为 [1,0]，response 部分标记为 [-1,0])

与其他模块的关系:
    - 被 rosetta/train/trainer.py 调用以构建训练/评估数据集
    - ChatDataset 的输出被 RosettaModel.forward() 消费
    - AlignedChatDataset 依赖 TokenAligner（rosetta/models/aligner.py）
    - kv_cache_index 在两阶段推理中决定哪些 KV 需要缓存/传递
"""

# ============================================================
# 导入依赖
# ============================================================
from typing import List, Dict, Any, Optional, Union, Callable
from datasets import load_dataset, load_from_disk       # HuggingFace datasets 库，用于加载/处理数据集
from torch.utils.data import Dataset                     # PyTorch 数据集基类
import torch
from transformers import AutoTokenizer                   # HuggingFace tokenizer，用于文本分词和 chat template
import inspect                                           # 用于反射获取函数签名（capture_init_args 装饰器）
import os
import hashlib                                           # 用于基于 ID 哈希取模的数据划分

# ============================================================
# 数据集注册系统 (Dataset Registry System)
# 通过全局字典维护 "名称 -> 数据集类" 的映射关系，
# 使得可以通过字符串名称（如配置文件中的 dataset_type）动态创建数据集实例。
# ============================================================
DATASET_REGISTRY = {}  # 全局注册表：{数据集名称(str): 数据集类(class)}

def register_dataset(cls=None, name=None):
    """
    数据集类注册装饰器 (Dataset Registration Decorator)。
    将数据集类注册到全局 DATASET_REGISTRY 中，支持带参和不带参两种用法。

    Register a dataset class in the global registry.
    Can be used as a decorator with or without arguments.
    
    用法示例 / Usage:
        @register_dataset              # 不带参，用类名注册
        @register_dataset()            # 空参，用类名注册
        @register_dataset(name="XXX")  # 指定自定义名称注册
    
    Args:
        cls: 要注册的类 / The class to register
        name: 可选的注册名称，若为 None 则使用类名 / Optional name, defaults to class name
        
    Returns:
        注册后的类（原样返回） / The registered class
    """
    def _register(cls):
        # 确定注册名称：优先使用显式指定的 name，否则使用类的 __name__
        dataset_name = name if name is not None else cls.__name__
        DATASET_REGISTRY[dataset_name] = cls
        # 同时注册小写版本，支持大小写不敏感查找 / Also register lowercase for case-insensitive lookup
        DATASET_REGISTRY[dataset_name.lower()] = cls
        return cls
    
    # 情况1: 作为 @register_dataset 使用（不带括号），cls 直接传入
    # Called as @register_dataset
    if cls is not None:
        return _register(cls)
    
    # 情况2: 作为 @register_dataset() 或 @register_dataset(name="...") 使用，返回内部装饰器
    # Called as @register_dataset() or @register_dataset(name="DatasetName")
    return _register


def capture_init_args(cls):
    """
    初始化参数捕获装饰器 (Init Args Capture Decorator)。
    自动将被装饰类的 __init__ 参数保存到实例的 self._init_args 字典中，
    便于后续序列化/重建数据集时恢复构造参数（例如在分布式训练或 checkpoint 恢复场景）。

    Decorator to capture initialization arguments of a dataset class.
    
    工作原理 / How it works:
        1. 保存原始 __init__ 方法的引用
        2. 用 new_init 包装：先通过 inspect.signature 提取参数名，
           再将位置参数和关键字参数映射到 self._init_args
        3. 最后调用原始 __init__

    Args:
        cls: 要装饰的类 / The class to decorate
        
    Returns:
        包装后的类（__init__ 被替换）/ The decorated class with automatic init args capture
    """
    original_init = cls.__init__  # 保存原始构造函数引用
    
    def new_init(self, *args, **kwargs):
        # 初始化参数存储字典 / Store all initialization arguments
        self._init_args = {}
        
        # 通过反射获取原始 __init__ 的参数名列表（跳过 self）
        # Get parameter names from the original __init__ method
        sig = inspect.signature(original_init)
        param_names = list(sig.parameters.keys())[1:]  # Skip 'self'
        
        # 将位置参数按顺序映射到参数名 / Map positional args to parameter names
        for i, arg in enumerate(args):
            if i < len(param_names):
                self._init_args[param_names[i]] = arg
        
        # 合并关键字参数 / Add keyword args
        self._init_args.update(kwargs)
        
        # 调用原始构造函数 / Call the original __init__
        original_init(self, *args, **kwargs)
    
    cls.__init__ = new_init  # 替换 __init__ 方法
    return cls


# ============================================================
# 统一批式过滤函数 (Unified Batch Filtering Functions)
# 这些工厂函数返回可在 dataset.filter(batched=True) 中使用的过滤函数，
# 用于高效地在数据加载阶段剔除不符合条件的样本。
# ============================================================


def create_text_length_filter(
    max_length: int,
    text_extractor: Callable[[Dict[str, Any]], str],
    tokenizer: Optional[Any] = None,
    use_tokens: bool = False
):
    """
    创建文本长度过滤器工厂函数。
    支持按"词数"或"token 数"两种模式过滤超长样本，防止 OOM。

    Unified text length filter that can handle both word count and token count filtering.
    
    Args:
        max_length: 允许的最大长度（词数或 token 数）/ Maximum allowed length
        text_extractor: 从单个样本中提取文本的函数 / Function that extracts text from a sample
        tokenizer: 用于 token 计数的分词器（use_tokens=True 时必需）
        use_tokens: True=按 token 数过滤, False=按空格分词的词数过滤
        
    Returns:
        可用于 dataset.filter(batched=True) 的过滤函数
        Filter function that can be used with dataset.filter(batched=True)
    """
    if use_tokens and tokenizer is None:
        raise ValueError("Tokenizer must be provided when use_tokens=True")
    
    def _text_length_filter_batch(batch):
        # 将批次的列式字典转为行式样本列表 / Convert columnar batch to list of row samples
        batch_size = len(next(iter(batch.values())))
        samples = [{key: values[i] for key, values in batch.items()} for i in range(batch_size)]
        try:
            # 用提取函数从每个样本中获取文本 / Extract text from each sample
            texts = [text_extractor(sample) for sample in samples]
            if use_tokens:
                # === Token 计数模式 ===
                # 如果文本是 chat message 列表（list），先用 apply_chat_template 渲染为字符串
                if hasattr(tokenizer, 'apply_chat_template') and any(isinstance(t, list) for t in texts):
                    rendered = []
                    for t in texts:
                        if isinstance(t, list):
                            # chat messages -> 渲染为 chat template 字符串
                            rendered.append(tokenizer.apply_chat_template(t, tokenize=False, add_generation_prompt=False))
                        else:
                            rendered.append(str(t))
                    tokenized = tokenizer(rendered, add_special_tokens=False)
                else:
                    # 普通字符串文本直接 tokenize
                    tokenized = tokenizer([str(t) for t in texts], add_special_tokens=False)
                # 取每个样本的 input_ids 长度作为 token 数
                lengths = [len(ids) for ids in tokenized["input_ids"]]
            else:
                # === 词数计数模式（按空格 split） ===
                lengths = [len(str(t).split()) for t in texts]
            # 返回布尔列表：True 表示保留，False 表示过滤掉
            return [length <= max_length for length in lengths]
        except Exception as e:
            print(f"Error in text length filter: {e}")
            return [False] * batch_size  # 出错时过滤掉所有样本（安全策略）
    
    return _text_length_filter_batch


def create_field_value_filter(target_value: Any, field_name: str, comparison: str = 'equal'):
    """
    创建字段值过滤器工厂函数。
    根据指定字段的值进行比较匹配，常用于语言过滤（如只保留英文样本）、类别筛选等。

    Unified field value filter for exact matching, language filtering, etc.
    
    Args:
        target_value: 比较目标值 / Value to compare against
        field_name: 要检查的字段名 / Field name to check
        comparison: 比较类型，支持:
            'equal'     -- 等于目标值
            'not_equal' -- 不等于目标值
            'in'        -- 字段值在目标集合中
            'not_in'    -- 字段值不在目标集合中
        
    Returns:
        可用于 dataset.filter(batched=True) 的过滤函数
        Filter function that can be used with dataset.filter(batched=True)
    """
    def _field_value_filter_batch(batch):
        field_values = batch.get(field_name, [])  # 取出该字段在所有样本中的值列表
        
        if comparison == 'equal':
            return [value == target_value for value in field_values]
        elif comparison == 'not_equal':
            return [value != target_value for value in field_values]
        elif comparison == 'in':
            # target_value 应为集合/列表，判断字段值是否在其中
            return [value in target_value for value in field_values]
        elif comparison == 'not_in':
            return [value not in target_value for value in field_values]
        else:
            raise ValueError(f"Unsupported comparison: {comparison}")
    
    return _field_value_filter_batch


def create_modulo_filter(mod_base: int, exclude_values: Union[int, List[int]], field_name: str = '_id'):
    """
    创建取模过滤器工厂函数。
    基于样本 ID 对 mod_base 取模的结果来过滤样本，常用于训练/验证/测试的数据划分。
    例如 mod_base=4, exclude_values=1 表示过滤掉 ID % 4 == 1 的样本（保留 75% 数据）。

    Unified modulo filter for ID-based filtering.
    
    Args:
        mod_base: 取模基数 / Modulo base
        exclude_values: 要排除的模值（单个 int 或列表）/ Value(s) to exclude
        field_name: 包含 ID 的字段名（默认 '_id'）/ Field name containing the ID
        
    Returns:
        可用于 dataset.filter(batched=True) 的过滤函数
        Filter function that can be used with dataset.filter(batched=True)
    """
    # 统一为列表形式，方便后续 in 判断
    if isinstance(exclude_values, int):
        exclude_values = [exclude_values]
    
    def _modulo_filter_batch(batch):
        ids = batch.get(field_name, [])
        results = []
        
        for _id in ids:
            try:
                # 尝试将 ID 转为整数进行取模 / Try numeric conversion first
                id_num = int(_id)
                mod_result = id_num % mod_base
            except (ValueError, TypeError):
                # 非数字 ID 使用 Python hash 函数取模 / Use hash for non-numeric IDs
                id_hash = hash(str(_id))
                mod_result = id_hash % mod_base
            
            # 如果模值不在排除列表中，则保留该样本
            results.append(mod_result not in exclude_values)
        
        return results
    
    return _modulo_filter_batch


def create_conversation_length_filter(min_messages: int, text_field: str = 'conversations'):
    """
    创建对话长度过滤器工厂函数。
    用于 OpenHermes 等对话格式数据集，过滤掉消息数不足的样本。
    仅统计 human/user 和 gpt/assistant 角色的消息，排除 system 消息。

    Unified conversation length filter for OpenHermes-style datasets.
    
    Args:
        min_messages: 要求的最少消息数（不含 system 消息）/ Minimum number of messages required
        text_field: 包含对话列表的字段名（默认 'conversations'）
        
    Returns:
        可用于 dataset.filter(batched=True) 的过滤函数
        Filter function that can be used with dataset.filter(batched=True)
    """
    def _conversation_length_filter_batch(batch):
        conversations_list = batch.get(text_field, [])
        results = []
        
        for conversations in conversations_list:
            try:
                # 统计有效消息数（排除 system 角色）
                # Extract messages (excluding system)
                message_count = 0
                for msg in conversations:
                    # 兼容两种角色字段名：'from'（OpenHermes 格式）和 'role'（标准 chat 格式）
                    role = msg.get('from') or msg.get('role')
                    if role in ('human', 'user', 'gpt', 'assistant'):
                        message_count += 1
                
                # 保留消息数严格大于 min_messages 的样本
                results.append(message_count > min_messages)
            except Exception:
                results.append(False)  # 解析失败的样本过滤掉
        
        return results
    
    return _conversation_length_filter_batch


# ============================================================
# 文本提取函数 (Text Extraction Functions)
# 将不同数据集的异构格式统一转换为标准文本或 chat messages。
# 这些函数通常作为 text_extractor 参数传给过滤器，
# 或在 Dataset.__getitem__ 中被调用。
# ============================================================

def extract_mmlu_text(sample: Dict[str, Any], question_field: str = 'question', choices_field: str = 'choices') -> str:
    """
    从 MMLU 格式样本中提取文本。
    将问题和所有选项拼接为一个字符串，用于长度过滤。

    Extract text from MMLU-style samples.
    
    Args:
        sample: 数据样本字典
        question_field: 问题字段名（默认 'question'）
        choices_field: 选项字段名（默认 'choices'）
    Returns:
        拼接后的文本字符串
    """
    question = sample.get(question_field, '')
    choices = sample.get(choices_field, [])
    
    # 兼容选项的两种格式：字典 {'text': [...]} 或直接列表 [...]
    # Handle both list and dict formats for choices
    if isinstance(choices, dict):
        choices_text = choices.get('text', [])
    else:
        choices_text = choices
    
    return (str(question) + " " + " ".join(map(str, choices_text))).strip()


def extract_chat_text(sample: Dict[str, Any], input_field: str = 'input', 
                     context_field: str = 'context', answers_field: str = 'answers') -> List[Dict[str, str]]:
    """
    从 LongBench 格式样本中提取 chat messages。
    将 context + instruction 组合为 user 消息，取第一个 answer 作为 assistant 消息。

    Extract chat messages from LongBench-style samples.

    Args:
        sample: 数据样本字典
        input_field: 指令/问题字段名
        context_field: 上下文字段名
        answers_field: 答案列表字段名
    Returns:
        标准 chat message 列表 [{"role": "user", ...}, {"role": "assistant", ...}]
    """
    input_text = str(sample.get(input_field, ''))
    context = str(sample.get(context_field, ''))
    answers = sample.get(answers_field, [])
    
    # 取第一个答案作为 assistant 回复
    assistant_message = answers[0] if answers and len(answers) > 0 else "No answer provided"
    
    # 构建完整的 chat 格式：如果有上下文则拼接，否则只用指令
    # Build complete chat format
    if context:
        human_message = f"Context: {context}\n\nInstruction: {input_text}"
    else:
        human_message = f"Instruction: {input_text}"
    
    return [
        {"role": "user", "content": human_message.strip()},
        {"role": "assistant", "content": assistant_message.strip()}
    ]


def extract_conversation_text(sample: Dict[str, Any], text_field: str = 'conversations') -> str:
    """
    从 OpenHermes 格式对话样本中提取第一条消息的文本（用于长度过滤）。

    Extract text from OpenHermes-style conversation samples.

    Args:
        sample: 数据样本字典
        text_field: 对话列表字段名
    Returns:
        第一条消息的 value 文本
    """
    conversations = sample.get(text_field, [])
    
    if conversations and len(conversations) > 0:
        return conversations[0].get('value', '')
    return ''


def extract_first_user_message(sample: Dict[str, Any], text_field: str = 'conversations') -> str:
    """
    从对话样本中提取第一条 human/user 消息的文本。
    用于按"用户输入长度"进行过滤。

    Extract the first human/user message from conversation-style samples.

    Args:
        sample: 数据样本字典
        text_field: 对话列表字段名
    Returns:
        第一条用户消息文本；若未找到则回退到第一条消息
    """
    conversations = sample.get(text_field, [])
    for msg in conversations:
        role = msg.get('from') or msg.get('role')
        if role in ('human', 'user'):
            return str(msg.get('value', ''))
    # Fallback to first message if role tags are missing
    # 回退策略：如果没有匹配的角色标签，取第一条消息
    if conversations:
        return str(conversations[0].get('value', ''))
    return ''


def extract_first_assistant_message(sample: Dict[str, Any], text_field: str = 'conversations') -> str:
    """
    从对话样本中提取第一条 gpt/assistant 消息的文本。

    Extract the first gpt/assistant message from conversation-style samples.

    Args:
        sample: 数据样本字典
        text_field: 对话列表字段名
    Returns:
        第一条助手消息文本；若未找到则回退到第二条消息
    """
    conversations = sample.get(text_field, [])
    for msg in conversations:
        role = msg.get('from') or msg.get('role')
        if role in ('gpt', 'assistant'):
            return str(msg.get('value', ''))
    # Fallback to second message if present
    # 回退策略：取第二条消息（如果存在）
    if len(conversations) > 1:
        return str(conversations[1].get('value', ''))
    return ''


def extract_openhermes_messages(sample: Dict[str, Any], text_field: str = 'conversations') -> List[Dict[str, str]]:
    """
    从 OpenHermes 格式样本中构建标准 chat messages。
    跳过 system 消息，保留所有 human/user 和 gpt/assistant 消息并按序排列。
    角色映射：human/user -> "user", gpt/assistant -> "assistant"

    Build chat messages excluding system; include all human/user and gpt/assistant in order.

    Args:
        sample: 数据样本字典
        text_field: 对话列表字段名
    Returns:
        标准 chat message 列表
    """
    conversation = sample.get(text_field, [])
    messages: List[Dict[str, str]] = []
    for msg in conversation:
        role = msg.get('from') or msg.get('role')
        if role == 'system':
            continue  # 跳过系统消息
        if role in ('human', 'user'):
            messages.append({"role": "user", "content": str(msg.get('value', '')).strip()})
        elif role in ('gpt', 'assistant'):
            messages.append({"role": "assistant", "content": str(msg.get('value', ''))})
    return messages


def extract_instruction_text(sample: Dict[str, Any], instruction_field: str = 'instruction', 
                           inputs_field: str = 'inputs') -> str:
    """
    从 Inkuba 格式的指令样本中提取文本。
    将 instruction 和 inputs 拼接，用于长度过滤。

    Extract text from Inkuba-style instruction samples.

    Args:
        sample: 数据样本字典
        instruction_field: 指令字段名
        inputs_field: 输入字段名
    Returns:
        拼接后的文本
    """
    instruction = sample.get(instruction_field)
    inputs = sample.get(inputs_field, '')
    
    if instruction is not None:
        return str(instruction) + "\n\n" + str(inputs)
    else:
        return str(inputs)


def extract_chat_pair_text(sample: Dict[str, Any], user_field: str = 'inputs', 
                          assistant_field: str = 'targets') -> List[Dict[str, str]]:
    """
    从 Aya 格式样本中提取 chat messages。
    inputs 字段作为 user 消息，targets 字段作为 assistant 消息。

    Extract chat messages from Aya-style samples.

    Args:
        sample: 数据样本字典
        user_field: 用户输入字段名
        assistant_field: 助手回复字段名
    Returns:
        标准 chat message 列表
    """
    user_text = str(sample.get(user_field, ''))
    assistant_text = str(sample.get(assistant_field, ''))
    
    return [
        {"role": "user", "content": user_text.strip()},
        {"role": "assistant", "content": assistant_text.strip()}
    ]



def extract_dolly_chat_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    从 Dolly 格式样本中提取 chat messages。
    Dolly 数据集包含 instruction（指令）、context（可选上下文）、response（回复）和 category（类别）。

    Extract chat messages from Dolly-style samples.

    字段说明 / Fields:
      - instruction: str  -- 用户指令
      - context: str      -- 可选的上下文信息（可能为空）
      - response: str     -- 模型回复
      - category: optional -- 类别标签（可能为空/缺失）

    Returns:
        标准 chat message 列表
    """
    instruction = str(sample.get('instruction', '')).strip()
    context = str(sample.get('context', '') or '').strip()
    response = str(sample.get('response', '')).strip()

    # 如果有上下文，将其拼接到用户消息前面
    if context:
        user_message = f"{context}\n\n{instruction}"
    else:
        user_message = f"{instruction}"

    return [
        {"role": "user", "content": user_message.strip()},
        {"role": "assistant", "content": response}
    ]


def extract_mmmlu_chat_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    从 MMMLU (多语言 MMLU) 格式样本中提取 chat messages。
    使用斯瓦希里语 (Swahili) 提示模板，适用于 OpenAI/MMMLU 多语言基准测试。

    Extract chat messages from MMMLU-style samples (OpenAI/MMMLU).
    Uses Swahili prompt template for multilingual benchmarking.

    Args:
        sample: 数据样本字典，包含 Question、A/B/C/D 选项和 Answer 字段
    Returns:
        标准 chat message 列表（斯瓦希里语提示）
    """
    choice_labels = ['A', 'B', 'C', 'D']

    # 斯瓦希里语提示模板：要求模型仅回答选项字母
    template = (
            "Jibu kwa usahihi swali lifuatalo:\n\n"   # 请正确回答以下问题
            "{{question}}\n\n"
            "Chaguo:\n"                                  # 选项
            "{{choices}}\n\n"
            "Maelekezo:\n"                               # 说明
            "- Soma swali na chaguo zote kwa makini.\n"
            "- Chagua jibu sahihi zaidi kati ya yaliyotolewa.\n"
            "- Jibu TU kwa herufi (A, B, C, D) inayolingana na jibu sahihi.\n"
            "- Usijumuishe maelezo, maandishi ya ziada, au alama yoyote ya uakifishaji.\n\n"
            "Jibu lako:"                                  # 你的回答
        )

    # 拼接选项文本
    choices_text = ""
    for label in choice_labels:
        content = sample.get(label, '')
        choices_text += f"{label}. {content}\n"

    # 填充模板中的占位符
    user_prompt = template.replace("{{choices}}", choices_text).replace("{{question}}", str(sample.get('Question', '')))

    # 构建 assistant 回复：正确答案标签 + 内容
    correct_label = sample.get('Answer', '')
    correct_content = sample.get(correct_label, '')
    assistant_response = f"**Jibu lako: {correct_label}. {correct_content}.**"

    return [
        {"role": "user", "content": user_prompt.strip()},
        {"role": "assistant", "content": assistant_response}
    ]




def apply_batch_filters(dataset, filters: list, filter_descriptions: list = None, 
                       batch_size: int = 4096, combine_filters: bool = True,
                       num_proc: Optional[int] = None):
    """
    对数据集应用多个批式过滤器，支持合并或顺序两种执行策略。
    合并模式将所有过滤条件在一次遍历中完成（AND 逻辑），效率更高。

    Apply multiple filters using native batched filtering for maximum performance.
    
    Args:
        dataset: 要过滤的 HuggingFace Dataset 对象
        filters: 批式过滤函数列表（每个函数接受 batch 字典，返回布尔列表）
        filter_descriptions: 可选的过滤器描述列表，用于日志输出
        batch_size: 过滤操作的批大小（默认 4096）
        combine_filters: True=合并所有过滤器为单次遍历; False=逐个顺序应用
        num_proc: 并行进程数（>1 时启用多进程过滤）
        
    Returns:
        (过滤后的数据集, 过滤前的原始长度) 元组
        Filtered dataset and original length
    """
    if not filters:
        return dataset, len(dataset)
    
    original_len = len(dataset)  # 记录过滤前的样本总数
    
    if combine_filters and len(filters) > 1:
        # === 合并模式：将所有过滤器合并为单次批操作，最大化效率 ===
        # Combine all filters into a single batched operation for maximum efficiency
        def _combined_batch_filter(batch):
            # 收集所有过滤器的结果 / Get results from all filters
            filter_results = []
            for filter_func in filters:
                filter_results.append(filter_func(batch))
            
            # 用 AND 逻辑合并所有过滤结果 / Combine results with AND logic
            combined_results = []
            batch_size = len(filter_results[0]) if filter_results else 0
            
            for i in range(batch_size):
                # 只有当所有过滤器都返回 True 时才保留该样本
                combined_results.append(all(result[i] for result in filter_results))
            
            return combined_results
        
        # 在单次遍历中应用合并后的过滤器 / Apply combined filter in a single pass
        filtered_dataset = dataset.filter(
            _combined_batch_filter,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc if num_proc and (num_proc or 0) > 1 else None,
            desc="Combined batch filtering"
        )
        
        # 输出过滤统计信息 / Print filtering results
        final_len = len(filtered_dataset)
        if original_len != final_len:
            print(f"Applied combined batch filtering: {original_len} -> {final_len} samples")
            if filter_descriptions:
                for desc in filter_descriptions:
                    print(f"  - {desc}")
    
    else:
        # === 顺序模式：逐个应用过滤器 ===
        # Apply each filter sequentially with batched processing
        current_dataset = dataset
        
        for i, (filter_func, desc) in enumerate(zip(filters, filter_descriptions or [''] * len(filters))):
            pre_filter_len = len(current_dataset)
            
            current_dataset = current_dataset.filter(
                filter_func,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc if num_proc and (num_proc or 0) > 1 else None,
                desc=f"Filtering: {desc}" if desc else f"Filter {i+1}"
            )
            
            post_filter_len = len(current_dataset)
            if desc and pre_filter_len != post_filter_len:
                print(f"  - {desc}: {pre_filter_len} -> {post_filter_len} samples")
        
        filtered_dataset = current_dataset
        final_len = len(filtered_dataset)
        if original_len != final_len:
            print(f"Applied sequential batch filtering: {original_len} -> {final_len} samples")
    
    return filtered_dataset, original_len


def generate_kv_cache_index(instruction_length: int, full_length: int) -> torch.tensor:
    """
    生成 KV-Cache 索引张量，标记每个 token 在 C2C 两阶段推理中的 KV-Cache 角色。

    **KV-Cache 索引含义（关键！）**:
    每个 token 对应一个 shape=(2,) 的索引向量 [flag, offset]：
      - instruction 部分（prompt）: [1, 0]
        表示该 token 的 KV 需要在 Stage 1 被 sharer 模型计算并缓存，
        后续会通过 Projector+Fuser 传递给 receiver。
      - response 部分（answer）: [-1, 0]
        表示该 token 的 KV 在 Stage 2 由 receiver 自身生成，
        不需要从 sharer 传递。

    在 RosettaDataCollator 中，这些索引会被用来将序列切分为不同 section，
    每个 section 内的 token 具有相同的 KV-Cache 角色，便于模型前向传播时
    分别处理（如只对 instruction section 执行 cache-to-cache 传递）。

    Generate KV cache index for the input sequence.
    
    Args:
        instruction_length: 指令（prompt）部分的 token 数
        full_length: 完整对话（prompt + response）的总 token 数
        
    Returns:
        shape=(seq_len, 2) 的 LongTensor，每行是该 token 的 KV-Cache 索引
        Tensor with KV cache index, shape: (full_length, 2)
    """
    assert instruction_length <= full_length

    # instruction 部分：每个 token 标记为 [1, 0]，表示"需要缓存并传递的 KV"
    # 注意：repeat(instruction_length - 1, ...) 意味着第一个 instruction token
    # 没有被标记（可能因为 BOS token 不参与 KV-Cache 传递）
    instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(instruction_length - 1, 1)

    # response 部分：每个 token 标记为 [-1, 0]，表示"receiver 自行生成的 KV"
    # +1 是因为 instruction 部分少了一个 token（instruction_length - 1），
    # 需要补足使总长度为 full_length
    label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(full_length - instruction_length + 1, 1)

    # 拼接：前 instruction_length-1 个为 instruction，后面为 response
    # 最终 shape: (full_length, 2)
    kv_cache_index = torch.cat([instruction_index, label_index], dim=0)  # shape: (seq_len, 2)

    return kv_cache_index


"""
============================================================
指令数据集 (Instruction Dataset)
将任意格式的原始数据转换为标准的 message 列表格式:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
每个数据集类都使用 @register_dataset 装饰器注册到全局注册表，
并使用 @capture_init_args 装饰器捕获构造参数。
============================================================
"""

# ============================================================
# LongBenchChatDataset: LongBench 长文本理解基准数据集
# 支持 21 个子任务（阅读理解、摘要、代码补全等），
# 每个子任务有独立的 prompt 模板。
# 支持 LongBench-E（筛选后的子集，13 个子任务）。
# ============================================================

@register_dataset
@capture_init_args
class LongBenchChatDataset(Dataset):
    """
    LongBench 长文本理解基准数据集适配器。
    将 LongBench 的 21 个异构子任务（阅读理解、摘要、少样本学习、代码补全、检索等）
    统一转换为标准 chat message 格式。

    核心逻辑:
    1. 加载指定子任务的数据（支持 LongBench-E 子集）
    2. 为每个子任务应用专属 prompt 模板
    3. 支持按词数/token 数/SHA256 哈希取模进行过滤
    4. 超长文本采用"首尾截断"策略（保留前半和后半）

    输出格式: [{"role": "user", "content": 格式化后的 prompt},
               {"role": "assistant", "content": 答案}]
    """
    
    def __init__(self, split: str = "test", num_samples: Optional[int] = None,
                 dataset_name: Optional[str] = None, language: Optional[str] = None,
                 max_word_count: Optional[int] = None, max_length: Optional[int] = 14000,
                 use_longbench_e: bool = True, filter_mod4: bool = True):
        """
        初始化LongBench数据集
        
        Args:
            split: 数据集分割 ("test" - LongBench主要使用test分割)
            num_samples: 使用的样本数量 (None表示全部)
            dataset_name: 特定数据集名称 (None表示所有数据集)
            language: 语言过滤 ("en" 或 "zh")
            max_word_count: 最大词数限制（用于英文文本）
            max_length: 最大字符长度限制
            use_longbench_e: 是否使用LongBench-E版本
            filter_mod4: 是否过滤_id mod4余1的样本
        """
        print(f"Loading LongBench{' -E' if use_longbench_e else ''} dataset (split: {split}, dataset: {dataset_name})...")
        
        # LongBench包含的数据集列表
        longbench_datasets = [
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", 
            "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news", 
            "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count", 
            "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"
        ]
        
        longbench_e_datasets = [
            "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", 
            "multi_news", "trec", "triviaqa", "samsum", "passage_count", 
            "passage_retrieval_en", "lcc", "repobench-p"
        ]
        
        target_datasets = longbench_e_datasets if use_longbench_e else longbench_datasets
        
        # 定义LongBench提示模板
        self.dataset_prompt_formats = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}
        
        # 定义不使用聊天模板的任务（这些任务直接使用原始 prompt 格式，不套用 chat template）
        #self.no_chat_template_tasks = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
        self.no_chat_template_tasks=['']  # 当前为空列表，即所有任务都使用 chat template
        self.use_longbench_e = use_longbench_e
        self.max_length = max_length  # token 级最大长度限制，用于超长截断

        if dataset_name:
            # 指定了单个子任务名称
            if dataset_name not in target_datasets:
                raise ValueError(f"Dataset {dataset_name} not found in LongBench{' -E' if use_longbench_e else ''}")
            target_datasets = [dataset_name]
            self.current_evaluating_subject = dataset_name  # 记录当前评估的子任务名
        else:
            self.current_evaluating_subject = None  # None 表示使用全部子任务
        
        # === 加载所有选定的子数据集并合并 ===
        all_data = []
        for dataset in target_datasets:
            try:
                # LongBench-E 的子任务名带 _e 后缀
                dataset_suffix = f"{dataset}_e" if use_longbench_e else dataset
                data = load_dataset('THUDM/LongBench', dataset_suffix, split=split)
                print(f"  Loaded {len(data)} samples from {dataset}")
                
                # 为每个样本添加 dataset_source 字段，标识来源子任务
                data = data.map(lambda x: {"dataset_source": dataset})
                all_data.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {dataset}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No datasets were successfully loaded")
        

        # 将所有子数据集拼接为一个统一的 Dataset
        from datasets import concatenate_datasets
        self.dataset = concatenate_datasets(all_data)
        




        # === 基于 SHA256 哈希的 mod4 过滤 ===
        # 过滤掉 SHA256(_id) % 4 == 1 的样本（保留约 75% 的数据）
        # 这是一种确定性的数据划分方法，比随机采样更稳定可复现
        if filter_mod4:
            original_len = len(self.dataset)
            
            def _mod4_not_1(example):
                _id = example.get('_id', '')
                # 使用 SHA256 哈希确保划分结果确定性且均匀分布
                id_hash = int(hashlib.sha256(str(_id).encode('utf-8')).hexdigest(), 16)
                
                return id_hash % 4 != 1
            
            self.dataset = self.dataset.filter(_mod4_not_1)
            print(f"Filtered by _id mod4 != 1: {original_len} -> {len(self.dataset)} samples")
        
        # === 限制样本数量（用于调试或子集实验） ===
        if num_samples and num_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_samples))
            
        print(f"Loaded total {len(self.dataset)} samples from LongBench{' -E' if use_longbench_e else ''}")    
    def __len__(self):
        return len(self.dataset)
    
    def _format_longbench_example(self, example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
        """
        将 LongBench 原始样本格式化为模型可接受的 prompt 字符串。
        
        关键步骤:
        1. 确定子任务类型 -> 选择对应的 prompt 模板
        2. 用样本字段填充模板占位符（如 {context}, {input}）
        3. 超长文本采用"首尾截断"策略：保留前 half_len 和后 half_len 个 token
           这种策略保留了文章的开头（通常含主题信息）和结尾（通常含关键细节）
        """
        # 1. 确定任务类型 / Determine the task type
        dataset_source = example.get('dataset_source', '')
        if self.current_evaluating_subject:
            current_subject = self.current_evaluating_subject
        else:
            current_subject = dataset_source
            
        # 去除 LongBench-E 的 "_e" 后缀以获取原始子任务名
        # 仅当字符串以"_e"结尾时才替换
        import re
        subject = re.sub(r"_e$", "", current_subject) if self.use_longbench_e else current_subject
        
        # 2. 获取该子任务的 prompt 模板
        if subject not in self.dataset_prompt_formats:
            subject = "narrativeqa"  # 默认模板（兜底策略）
        prompt_format = self.dataset_prompt_formats[subject]
        
        # 3. 用 **example 展开所有字段来填充模板占位符
        # 例如 {context} -> example['context'], {input} -> example['input']
        raw_prompt = prompt_format.format(**example)
        
        # 4. 超长截断逻辑：首尾各保留一半
        # 先 tokenize 检查长度，如果超过 max_length 则截取首尾
        tokenized_raw = tokenizer(raw_prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_raw) > self.max_length:
            half_len = int(self.max_length / 2)
            # 保留前 half_len 个 token 和 后 half_len 个 token
            raw_prompt = tokenizer.decode(tokenized_raw[:half_len], skip_special_tokens=True) + \
                        tokenizer.decode(tokenized_raw[-half_len:], skip_special_tokens=True)
        
        # 5. 当前直接返回 raw_prompt（chat template 在外层 apply）
        # 应用Chat Template

        final_prompt = raw_prompt
        print(len(tokenized_raw))
        return final_prompt
    
    def __getitem__(self, idx):
        """
        获取第 idx 个样本，返回标准 chat message 列表。
        
        Returns:
            [{"role": "user", "content": 格式化 prompt},
             {"role": "assistant", "content": 答案}]
        """
        sample = self.dataset[idx]
        
        # 使用 _format_longbench_example 格式化 prompt（含模板填充和超长截断）
        formatted_prompt = self._format_longbench_example(sample, self.tokenizer)
        
        # 提取答案：取 answers 列表的第一个元素
        answers = sample.get('answers', [])
        assistant_message = answers[0] if answers and len(answers) > 0 else "No answer provided"
        
        return [
            {
                "role": "user",
                "content": formatted_prompt.strip()
            },
            {
                "role": "assistant", 
                "content": assistant_message.strip()
            }
        ]

# ============================================================
# MMLUChatDataset: MMLU 多任务语言理解基准数据集
# 包含 57 个学科的选择题（STEM、人文、社科等），
# 转换为问答 chat 格式用于模型训练/评估。
# ============================================================
@register_dataset
@capture_init_args
class MMLUChatDataset(Dataset):
    """
    MMLU (Massive Multitask Language Understanding) 数据集适配器。
    将 cais/mmlu "all" 版本的选择题目转换为 chat 问答格式。
    支持按 token 数过滤过长的样本。

    输出格式: [{"role": "user", "content": "Question: ... Choices: A. ... B. ..."},
               {"role": "assistant", "content": "The correct answer is X."}]
    """

    def __init__(self, split: str = "train", num_samples: Optional[int] = None, max_word_count: Optional[int] = None):
        """
        初始化 MMLU 数据集。

        Args:
            split: 数据集分割（train/dev/test）
            num_samples: 使用的样本数量（None 表示全部）
            max_word_count: 最大 token 数限制（按完整 chat 计算：user + assistant）
        """
        print(f"Loading MMLU dataset (split: {split})...")
        # 加载 cais/mmlu 的 "all" 配置（包含全部 57 个学科）
        dataset = load_dataset("cais/mmlu", "all")
        dataset = dataset[split]

        # Ensure we have a proper Dataset object
        if hasattr(dataset, 'select'):
            self.dataset = dataset
        else:
            raise ValueError(f"Unexpected dataset type: {type(dataset)}")

        # 限制样本数量
        if num_samples and num_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_samples))
            
        # === 按完整 chat（user + assistant）的 token 数过滤 ===
        # 使用轻量级 tokenizer（Qwen3-0.6B）以平衡速度和准确性
        if max_word_count is not None:
            self._mmlu_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            extractor = lambda sample: self._build_chat_messages(sample)
            filters = [create_text_length_filter(max_word_count, extractor, self._mmlu_tokenizer, use_tokens=True)]
            filter_descriptions = [f"Token count filter (full chat): max {max_word_count}"]
            self.dataset, _ = apply_batch_filters(self.dataset, filters, filter_descriptions)

        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self._build_chat_messages(sample)

    def _build_chat_messages(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        将 MMLU 样本构建为 chat message 列表。
        
        Args:
            sample: 包含 question, choices, answer 字段的字典
        Returns:
            标准 chat message 列表
        """
        choice_labels = ['A', 'B', 'C', 'D']
        question = sample.get('question', '')
        choices_list = sample.get('choices', [])
        # 构建用户 prompt：问题 + 选项列表
        user_prompt = f"Question: {question}\n\nChoices:\n"
        for i, choice in enumerate(choices_list):
            label = choice_labels[i] if i < len(choice_labels) else chr(65 + i)
            user_prompt += f"{label}. {choice}\n"
        # 获取正确答案标签
        ans_idx = sample.get('answer', 0)
        if isinstance(ans_idx, str) and ans_idx.isdigit():
            ans_idx = int(ans_idx)
        ans_label = choice_labels[ans_idx] if 0 <= int(ans_idx) < len(choice_labels) else chr(65 + int(ans_idx))
        assistant_text = f"The correct answer is {ans_label}."
        return [
            {"role": "user", "content": user_prompt.strip()},
            {"role": "assistant", "content": assistant_text.strip()},
        ]

# ============================================================
# MMLUCotChatDataset: MMLU-Pro 带思维链 (Chain-of-Thought) 训练数据集
# 使用 Brench/MMLU-Pro-CoT-Train-43K 数据集，
# 包含带有详细推理步骤的答案，用于训练模型的推理能力。
# ============================================================
@register_dataset
@capture_init_args
class MMLUCotChatDataset(Dataset):
    """
    MMLU-Pro-CoT 数据集适配器。
    使用预生成的思维链 (Chain-of-Thought) 答案进行训练，
    帮助模型学习逐步推理的能力。

    输出格式: [{"role": "user", "content": 问题},
               {"role": "assistant", "content": 思维链推理过程}]
    """

    def __init__(self, split: str = "train", num_samples: Optional[int] = None):
        """
        初始化 MMLU-CoT 数据集。

        Args:
            split: 数据集分割
            num_samples: 使用的样本数量（None 表示全部）
        """
        print(f"Loading MMLUCot dataset (split: {split})...")
        # 加载 MMLU-Pro-CoT 训练集（约 43K 样本）
        dataset = load_dataset("Brench/MMLU-Pro-CoT-Train-43K")
        dataset = dataset[split]

        if hasattr(dataset, 'select'):
            self.dataset = dataset
        else:
            raise ValueError(f"Unexpected dataset type: {type(dataset)}")

        if num_samples and num_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_samples))

        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        获取第 idx 个 CoT 样本。
        user_prompt = 原始问题，assistant_response = 思维链推理过程。
        """
        sample = self.dataset[idx]

        user_prompt = sample['question'] + "\n"  # 问题作为用户输入

        assistant_response = sample['chain_of_thoughts']  # 思维链作为助手回复

        return [
            {
                "role": "user",
                "content": user_prompt.strip()
            },
            {
                "role": "assistant",
                "content": assistant_response  # 包含完整的推理步骤
            }
        ]

# ============================================================
# LLMGeneratedChatDataset: LLM 生成的合成训练数据集
# 从本地磁盘加载由教师模型（如 GPT-4）生成的合成数据，
# 包含 input_text（用户输入）和 model_response（模型回复）。
# 支持按 token 数进行双向过滤（问题和总长度）。
# ============================================================
@register_dataset
@capture_init_args
class LLMGeneratedChatDataset(Dataset):
    """
    LLM 生成的合成数据集适配器。
    从本地路径加载已经由教师模型生成的问答对数据，
    支持 token 级别的长度过滤（问题部分 + 总长度双重限制）。

    输出格式: [{"role": "user", "content": 输入文本},
               {"role": "assistant", "content": 模型生成的回复}]
    """

    def __init__(self, split: str = "train", num_samples: Optional[int] = None, data_path: str = "./teacher_datasets/output/dataset_finished", max_word_count: Optional[int] = None):
        """
        初始化 LLM 生成的数据集。

        Args:
            split: 数据集分割（此数据集通常只有一个 split）
            num_samples: 使用的样本数量（None 表示全部）
            data_path: 本地数据集路径（HuggingFace save_to_disk 格式）
            max_word_count: 最大 token 数限制
        """
        print(f"Loading LLMGeneratedCot dataset (split: {split})...")
        # 从本地磁盘加载数据集（HuggingFace datasets 的 save_to_disk 格式）
        dataset = load_from_disk(data_path)

        if hasattr(dataset, 'select'):
            self.dataset = dataset
        else:
            raise ValueError(f"Unexpected dataset type: {type(dataset)}")

        # 使用轻量级 tokenizer 进行长度过滤
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        
        if max_word_count is not None:
            original_len = len(self.dataset)
            half = max_word_count // 2  # 问题部分最多占一半的 token 预算
            def _under_token_limit(batch):
                # 分别 tokenize 问题和回复
                q = tokenizer(batch["input_text"], add_special_tokens=False, padding=False, truncation=False)
                a = tokenizer(batch["model_response"], add_special_tokens=False, padding=False, truncation=False)
                return [
                    # 双重限制：问题 token 数 <= half 且 总 token 数 <= max_word_count
                    (len(q_ids) <= half) and (len(q_ids) + len(a_ids) <= max_word_count)
                    for q_ids, a_ids in zip(q["input_ids"], a["input_ids"])
                ]

            self.dataset = self.dataset.filter(
                _under_token_limit,
                batched=True,
                batch_size=2048,                    # 视显存/内存调大
                num_proc=min(8, os.cpu_count() or 1),
                load_from_cache_file=True,
                desc=f"Filter max_word_count={max_word_count}",
            )
            print(f"Filtered by max_word_count={max_word_count}: {original_len} -> {len(self.dataset)} samples")

        # 限制样本数量
        if num_samples and num_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_samples))

        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        获取第 idx 个合成样本，返回标准 chat message 列表。
        input_text 直接作为 user 消息（当前未使用内部的 _extract_question 函数），
        model_response 作为 assistant 消息。
        """
        sample = self.dataset[idx]

        input_text = sample.get('input_text', '') or ''

        # === 内部定义的问题提取函数（当前未使用，question = input_text） ===
        # Extract question from the original prompt
        # Original format: instruction paragraph + question paragraph + reminder paragraph
        def _extract_question(text: str) -> str:
            """Extract the question part from a math problem prompt.
            
            Expected format:
            - First paragraph: instruction (e.g., "Solve the following math problem...")
            - Middle paragraph(s): the actual question
            - Last paragraph: reminder (e.g., "Remember to put your answer...")
            """
            if not text.strip():
                return text.strip()
            
            # Split by double newlines (paragraphs)
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            if len(paragraphs) <= 1:
                # Single paragraph or no clear structure, return as is
                return text.strip()
            
            # If we have multiple paragraphs, assume:
            # - First paragraph is instruction
            # - Last paragraph is reminder
            # - Middle paragraphs are the question
            if len(paragraphs) == 2:
                first_lower = paragraphs[0].lower()
                if any(keyword in first_lower for keyword in ['solve', 'answer', 'problem', 'step by step', 'form answer']):
                    return paragraphs[1]
                else:
                    return paragraphs[0]
            else:
                question_paragraphs = paragraphs[1:-1]
                return '\n\n'.join(question_paragraphs)
        
        # 当前直接使用完整 input_text 作为问题（未调用 _extract_question）
        # question = _extract_question(input_text)
        question = input_text
        
        # 应用简单模板（当前为空模板，即不添加额外格式）
        new_template = "{question}"
        filled_prompt = new_template.format(question=question)

        user_prompt = filled_prompt.strip()

        assistant_response = sample['model_response']

        return [
            {
                "role": "user",
                "content": user_prompt.strip()
            },
            {
                "role": "assistant",
                "content": assistant_response
            }
        ]

# ============================================================
# OpenBookChatDataset: OpenBookQA 科学常识问答数据集
# 来自 allenai/openbookqa，包含基于科学常识的选择题。
# ============================================================
@register_dataset
@capture_init_args
class OpenBookChatDataset(Dataset):
    """
    OpenBookQA 数据集适配器。
    将科学常识选择题转换为 chat 问答格式。

    输出格式: [{"role": "user", "content": "Question: ... Choices: A. ... B. ..."},
               {"role": "assistant", "content": "The correct answer is X."}]
    """

    def __init__(self, split: str = "train", num_samples: Optional[int] = None):
        """
        初始化 OpenBookQA 数据集。

        Args:
            split: 数据集分割（train/validation/test）
            num_samples: 使用的样本数量
        """
        print(f"Loading OpenBook dataset (split: {split})...")
        # 加载 allenai/openbookqa 数据集
        dataset = load_dataset("allenai/openbookqa", "main")
        dataset = dataset[split]

        if hasattr(dataset, 'select'):
            self.dataset = dataset
        else:
            raise ValueError(f"Unexpected dataset type: {type(dataset)}")

        if num_samples and num_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_samples))

        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        获取第 idx 个 OpenBookQA 样本。
        将 question_stem + choices 构建为 user 消息，answerKey 构建为 assistant 消息。
        """
        sample = self.dataset[idx]
        choice_labels = ['A', 'B', 'C', 'D']

        # 构建用户 prompt
        user_prompt = (
            f"Question: {sample['question_stem']}\n\n"
            f"Choices:\n"
        )
        for idx, choice in enumerate(sample['choices']['text']):
            label = choice_labels[idx]
            user_prompt += f"{label}. {choice}\n"

        correct_label = sample["answerKey"]  # 正确答案的字母标签
        assistant_response = f"The correct answer is {correct_label}."

        return [
            {
                "role": "user",
                "content": user_prompt.strip()
            },
            {
                "role": "assistant",
                "content": assistant_response
            }
        ]

# ============================================================
# OpenHermesChatDataset: OpenHermes-2.5 通用对话数据集
# 来自 teknium/OpenHermes-2.5，包含约 100 万条对话数据，
# 来源多样（GPT-4、Claude 等合成数据）。
# 支持按对话轮数和 token 数过滤。
# ============================================================
@register_dataset
@capture_init_args
class OpenHermesChatDataset(Dataset):
    """
    OpenHermes-2.5 通用对话数据集适配器。
    包含多轮对话数据，支持按最小对话轮数和最大 token 数进行过滤。
    角色映射：human/user -> "user", gpt/assistant -> "assistant", system -> 跳过

    输出格式: [{"role": "user"/"assistant", "content": "..."}, ...]（多轮对话列表）
    """

    def __init__(self, split: str = "train", num_samples: Optional[int] = None, max_word_count: Optional[int] = None, min_conversation_turns: int = 0):
        """
        初始化 OpenHermes 数据集。

        Args:
            split: 数据集分割
            num_samples: 使用的样本数量
            max_word_count: 最大 token 数限制（所有消息合并后计算）
            min_conversation_turns: 最小对话轮数（默认 0 表示不过滤；
                                    设为 3 可过滤掉短对话，只保留多轮对话）
        """
        print(f"Loading OpenHermes dataset (split: {split})...")
        # 加载 teknium/OpenHermes-2.5 数据集
        dataset = load_dataset("teknium/OpenHermes-2.5")
        dataset = dataset[split]

        if hasattr(dataset, 'select'):
            self.dataset = dataset
        else:
            raise ValueError(f"Unexpected dataset type: {type(dataset)}")
        
        if num_samples and num_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_samples))

        # === 应用过滤器 ===
        filters = []
        filter_descriptions = []

        # 对话长度过滤：排除消息数 <= min_conversation_turns - 1 的对话
        # 例如 min_conversation_turns=3 时，排除消息数 <= 2 的对话（只保留多轮对话）
        if min_conversation_turns > 0:
            filters.append(create_conversation_length_filter(min_conversation_turns - 1, 'conversations'))
            filter_descriptions.append(f"Conversation length filter: min {min_conversation_turns} messages (multi-turn only)")

        # Token 数过滤：所有消息合并后的总 token 数不超过 max_word_count
        if max_word_count is not None:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            # 使用 extract_openhermes_messages 提取所有有效消息用于 token 计数
            extractor = lambda sample: extract_openhermes_messages(sample, 'conversations')
            filters.append(create_text_length_filter(max_word_count, extractor, tokenizer, use_tokens=True))
            filter_descriptions.append(f"Token count filter: max {max_word_count}")

        # 应用所有过滤器（使用多进程加速）
        if filters:
            self.dataset, _ = apply_batch_filters(self.dataset, filters, filter_descriptions, num_proc=8)

        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """获取第 idx 个对话样本，返回标准 chat message 列表（多轮）。"""
        sample = self.dataset[idx]
        # 使用 extract_openhermes_messages 将 OpenHermes 格式转为标准 chat 格式
        return extract_openhermes_messages(sample, 'conversations')
    
"""
============================================================
聊天数据集 (Chat Dataset)
将标准 message 格式转换为模型所需的 input_ids + labels。
这一层在 Instruction Dataset 之上，添加了 tokenize、
label 掩码和 kv_cache_index 生成。
============================================================
"""

# ============================================================
# ChatDataset: 单模型 tokenize 层
# 将 chat messages 转换为 token IDs，同时生成:
#   - input_ids: 完整对话的 token 序列
#   - labels: 指令部分为 -100（不计算损失），回复部分为实际 token ID
#   - kv_cache_index: 标记每个 token 的 KV-Cache 角色
# ============================================================
class ChatDataset(Dataset):
    """
    聊天格式训练数据集（单模型版本）。
    包装一个 Instruction Dataset（如 MMLUChatDataset），
    使用 tokenizer.apply_chat_template 将 messages 渲染为模型可理解的格式，
    然后 tokenize 并生成 labels 和 kv_cache_index。

    与 HuggingFace Trainer 兼容，可直接用于训练循环。

    输出字典:
        - input_ids: List[int] -- 完整对话的 token ID 列表
        - labels: List[int] -- 标签序列（指令部分为 -100）
        - kv_cache_index: Tensor -- shape (seq_len, 2) 的 KV-Cache 索引
    """
    
    def __init__(self, chat_dataset, tokenizer: AutoTokenizer, max_length: int = 32768):
        self.chat_dataset = chat_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.chat_dataset)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        获取第 idx 个 tokenize 后的样本。
        
        流程:
        1. 从底层 chat_dataset 获取 messages
        2. 分别渲染 instruction（不含 assistant 回复）和 full_text（含回复）
        3. tokenize 两者
        4. 生成 labels: instruction 部分设为 -100（不参与损失计算）
        5. 生成 kv_cache_index: 标记每个 token 的 KV-Cache 角色

        Returns:
            包含 input_ids, labels, kv_cache_index 的字典
        """
        messages = self.chat_dataset[idx]
        
        # === Step 1: 渲染 instruction（去掉最后一条 assistant 消息） ===
        # 使用 apply_chat_template + add_generation_prompt=True 在末尾添加生成提示
        instruction = self.tokenizer.apply_chat_template(
            messages[:-1],      # 只取 user 消息（不含 assistant 回复）
            tokenize=False,
            add_generation_prompt=True,   # 在末尾添加 "Assistant:" 等生成提示
            enable_thinking=False,        # 禁用思维模式（Qwen 特有）
        )

        # === Step 2: 渲染完整对话（user + assistant） ===
        full_text = self.tokenizer.apply_chat_template(
            messages,           # 完整对话
            tokenize=False,
            add_generation_prompt=False,  # 不在末尾添加生成提示
            enable_thinking=False,
        )

        # === Step 3: Tokenize ===
        instruction_tokens = self.tokenizer(instruction, add_special_tokens=False)["input_ids"]
        full_tokens = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        
        # === Step 4: 截断过长序列 ===
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[:self.max_length]
        
        # === Step 5: 生成 labels（关键！） ===
        # 指令部分的 token 设为 -100（PyTorch CrossEntropyLoss 的忽略值），
        # 只有 assistant 回复部分的 token 参与损失计算
        labels = [-100] * len(instruction_tokens) + full_tokens[len(instruction_tokens):]
        # labels = [-100] * (len(full_tokens) - 4) + full_tokens[-4:]
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
        
        # === Step 6: 生成 KV-Cache 索引 ===
        # instruction 部分的 token 标记为 [1, 0]（需要缓存的 KV），
        # response 部分标记为 [-1, 0]（不需要缓存的 KV）
        kv_cache_index = generate_kv_cache_index(len(instruction_tokens), len(full_tokens))
        # kv_cache_index = generate_kv_cache_index(len(full_tokens)-4, len(full_tokens))
        # kv_cache_index = generate_kv_cache_index(len(full_tokens) + 1, len(full_tokens))

        return {
            "input_ids": full_tokens,       # 完整对话 token IDs
            "labels": labels,               # 标签序列（指令部分 -100）
            "kv_cache_index": kv_cache_index # KV-Cache 角色索引
        }


# ============================================================
# AlignedChatDataset: 双模型 token 对齐层（SLM + LLM）
# 使用 TokenAligner 将同一条对话分别 tokenize 为 SLM 和 LLM 的 token 序列，
# 并建立 token 级别的对齐关系（section map）。
# 这是 C2C 框架中实现跨模型 KV-Cache 传递的关键数据层组件。
# ============================================================
class AlignedChatDataset(Dataset):
    """
    双模型对齐聊天数据集。
    使用 TokenAligner 预计算 SLM（小语言模型/receiver）和 LLM（大语言模型/sharer）
    的对齐 token 序列，并生成对应的 labels、kv_cache_index 和 padding mask。

    关键设计:
    - slm_ids / llm_ids: 两个模型各自的 token 序列（已 padding 到相同长度）
    - sections: token 对齐段落信息，标记每个段落属于哪个模型/消息
    - message_mask: 标记哪些 token 属于实际消息内容（非模板/特殊 token）
    - kv_cache_index 额外用 message_mask 掩码非消息部分

    输出字典:
        - input_ids: [slm_ids, llm_ids] -- 两个模型的 token 序列
        - labels: List[int] -- 标签序列
        - kv_cache_index: Tensor -- KV-Cache 索引
        - messages: 原始 messages（用于调试）
        - model_padding_mask: [slm_mask, llm_mask] -- 每个模型的 padding mask
    """
    
    def __init__(self, instruct_dataset: Dataset, aligner: Any, max_length: int = 32768):
        """
        Args:
            instruct_dataset: 底层的 Instruction Dataset
            aligner: TokenAligner 实例，负责双模型 token 对齐
            max_length: 最大序列长度
        """
        self.dataset = instruct_dataset
        self.aligner = aligner
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        获取第 idx 个对齐后的双模型样本。
        
        核心流程:
        1. 调用 aligner.align_chat_messages 获取对齐详情
        2. 从 sections 中确定指令边界（最后一条消息的起始位置）
        3. 生成 labels（指令前为 -100）
        4. 生成 kv_cache_index（额外用 message_mask 掩码非消息部分）
        """
        messages = self.dataset[idx]

        # === 调用 TokenAligner 进行双模型 token 对齐 ===
        # 返回对齐后的 token IDs、padding mask、message mask 和 section 信息
        details = self.aligner.align_chat_messages(messages, add_generation_prompt=False, return_details=True)
        slm_ids: List[int] = details['slm_ids_padded']      # SLM（receiver）的 token 序列
        llm_ids: List[int] = details['llm_ids_padded']       # LLM（sharer）的 token 序列
        sections = details['sections']                        # token 对齐段落信息

        slm_pad_mask = torch.tensor(details['slm_padding_mask'])  # SLM padding 位置掩码
        llm_pad_mask = torch.tensor(details['llm_padding_mask'])  # LLM padding 位置掩码
        message_mask = torch.tensor(details['message_mask'])       # 实际消息内容掩码

        # === 确定指令边界 ===
        # 从后向前遍历 sections，找到最后一条 message 类型段的起始位置
        # 该位置之前为 instruction（不参与损失计算），之后为 assistant 回复
        instr_end = 0
        for sec_idx in range(len(sections) - 1, -1, -1):
            sec = sections[sec_idx]
            if sec['type'] == 'message':
                instr_end = sec['slm_range'][0]  # 该段在 SLM token 序列中的起始位置
                break

        # === 生成 labels ===
        # 与 ChatDataset 相同策略：指令部分 -100，回复部分为实际 token ID
        labels = [-100] * instr_end + slm_ids[instr_end:]
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]

        # === 截断过长序列 ===
        if len(slm_ids) > self.max_length:
            slm_ids = slm_ids[:self.max_length]
            slm_pad_mask = slm_pad_mask[:self.max_length]  # 同步截断 padding mask
        if len(llm_ids) > self.max_length:
            llm_ids = llm_ids[:self.max_length]
            llm_pad_mask = llm_pad_mask[:self.max_length]

        # === 生成 KV-Cache 索引 ===
        kv_cache_index = generate_kv_cache_index(instr_end, len(slm_ids))
        # 额外掩码：将非消息部分（如模板 token）的 KV-Cache 索引设为 [-1, 0]
        # 这样这些 token 的 KV 不会被缓存或传递
        kv_cache_index[~message_mask] = torch.tensor([[-1,0]])

        return {
            "input_ids": [slm_ids, llm_ids],        # 双模型 token 序列
            "labels": labels,                         # 标签序列
            "kv_cache_index": kv_cache_index,         # KV-Cache 索引（含 message_mask 掩码）
            "messages": messages,                     # 原始 messages（调试用）
            # 每个模型的 padding mask（用于 attention 计算时屏蔽 padding 位置）
            "model_padding_mask": [slm_pad_mask, llm_pad_mask],
        }


# ============================================================
# BaselineChatDataset: 基线模型的简单 tokenize 层
# 与 ChatDataset 类似，但不生成 kv_cache_index。
# 用于不需要 C2C 特性的基线模型训练/评估。
# ============================================================
class BaselineChatDataset(Dataset):
    """
    基线模型聊天数据集。
    简单 tokenize 层，不包含 KV-Cache 索引生成。
    用于不需要 Rosetta/C2C 特性的基线模型训练。

    输出字典:
        - input_ids: List[int] -- 完整对话的 token ID 列表
        - labels: List[int] -- 标签序列（指令部分为 -100）
    """
    
    def __init__(self, chat_dataset, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.chat_dataset = chat_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.chat_dataset)
    
    def __getitem__(self, idx):
        """获取第 idx 个基线样本（不含 kv_cache_index）。"""
        messages = self.chat_dataset[idx]
        
        # 渲染 instruction（仅 user 消息）
        instruction = self.tokenizer.apply_chat_template(
            messages[:1],       # 只取第一条消息（user）
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # 渲染完整对话
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        # Tokenize
        instruction_tokens = self.tokenizer(instruction, add_special_tokens=False)["input_ids"]
        full_tokens = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        
        # 截断
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[:self.max_length]
        
        # 生成 labels（指令部分 -100）
        labels = [-100] * len(instruction_tokens) + full_tokens[len(instruction_tokens):]
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]

        # 注意：基线版本不返回 kv_cache_index
        return {
            "input_ids": full_tokens,
            "labels": labels,
        }

"""
============================================================
数据整理器 (Data Collator)
将多个样本组合为一个 padding 后的 batch。
RosettaDataCollator 支持多模型（SLM/LLM）输入，
并按 kv_cache_index 的变化点将序列切分为多个 section，
每个 section 独立 padding 后再拼接，确保 KV-Cache 边界对齐。
============================================================
"""

# ============================================================
# RosettaDataCollator: Rosetta 模型专用数据整理器
# 核心流程（4 步）:
#   1. _normalize_input_format -- 统一输入格式（单/双模型）
#   2. _split_into_sections    -- 按 kv_cache_index 变化点切分序列
#   3. _pad_sections           -- 对每个 section 独立 padding
#   4. _apply_length_constraints -- 应用 max_length 截断
# ============================================================
class RosettaDataCollator:
    """
    Rosetta 模型训练数据整理器。
    支持单模型和多模型（SLM/LLM）输入的统一 batch padding。
    
    核心特性:
    - 按 kv_cache_index 的变化点将序列切分为多个 section
      （如 instruction section 和 response section）
    - 每个 section 内独立 padding，确保 KV-Cache 角色在 batch 内对齐
    - 支持每个模型使用不同的 pad_token_id
    - 支持 max_length 截断和 pad_to_multiple_of 对齐

    输入: List[Dict] -- 来自 ChatDataset/AlignedChatDataset 的样本列表
    输出: Dict -- padding 后的 batch 字典，可直接送入模型 forward
    """

    def __init__(self, slm_tokenizer: AutoTokenizer, llm_tokenizer: AutoTokenizer = None, 
                 pad_to_multiple_of: Optional[int] = None, max_length: Optional[int] = None, 
                 aligner: Optional[Any] = None, do_alignment: bool = False):
        """
        初始化数据整理器。

        Initialize the collator.
        
        Args:
            slm_tokenizer: 小语言模型 (SLM/receiver) 的 tokenizer
            llm_tokenizer: 大语言模型 (LLM/sharer) 的 tokenizer（可选，单模型时不需要）
            pad_to_multiple_of: 将序列长度填充到此值的倍数（用于 GPU 内存对齐）
            max_length: 最大序列长度
            aligner: Token 对齐模块（如果需要运行时对齐）
            do_alignment: 是否在 collate 时执行对齐
        """
        self.slm_tokenizer = slm_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_length = max_length
        self.aligner = aligner
        self.do_alignment = do_alignment
        
        if self.do_alignment:
            assert self.aligner is not None, "Aligner must be provided if do_alignment is True"
        
        # 存储不同模型的 padding token ID
        # SLM 和 LLM 可能使用不同的 pad token（不同模型的 vocab 不同）
        self.slm_pad_token_id = self.slm_tokenizer.pad_token_id
        self.llm_pad_token_id = self.llm_tokenizer.pad_token_id if self.llm_tokenizer else self.slm_pad_token_id

    def _normalize_input_format(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        """
        统一输入格式，处理单模型和双模型两种情况。
        将 input_ids 统一为 tensor 列表，生成 attention_mask、labels、position_ids。

        Normalize input format to handle both single and dual model inputs.
        
        Args:
            feature: 单个样本的特征字典
            
        Returns:
            标准化后的特征字典，包含:
            - input_ids: List[Tensor] (单模型时长度为 1，双模型时长度为 2)
            - attention_mask: List[Tensor]
            - labels: Tensor
            - kv_cache_index: Tensor
            - position_ids: Tensor
        """
        # === 统一 input_ids 格式 ===
        input_ids = feature['input_ids']
        if isinstance(input_ids, list) and len(input_ids) > 0:
            if isinstance(input_ids[0], list):
                # 双模型情况: [[slm_ids], [llm_ids]] -> 转为 tensor 列表
                input_ids_tensors = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
            else:
                # 单模型情况: [id1, id2, ...] -> 包装为单元素列表
                input_ids_tensors = [torch.tensor(input_ids, dtype=torch.long)]
        else:
            # 兜底: 假设为单模型
            input_ids_tensors = [torch.tensor(input_ids, dtype=torch.long)]
        
        # === 生成 attention_mask ===
        attention_masks = []
        if "model_padding_mask" in feature:
            # 使用模型特定的 padding mask（来自 AlignedChatDataset）
            # padding_mask 中 True 表示 padding 位置，取反后转为 float 作为 attention mask
            for model_padding_mask in feature["model_padding_mask"]:
                attention_masks.append((~model_padding_mask).float())
        else:
            # 生成默认全 1 attention mask（无 padding）
            for input_tensor in input_ids_tensors:
                attention_masks.append(torch.ones(len(input_tensor), dtype=torch.float))
        
        return {
            'input_ids': input_ids_tensors,
            'attention_mask': attention_masks,
            'labels': torch.tensor(feature['labels'], dtype=torch.long),
            'kv_cache_index': feature['kv_cache_index'],
            'position_ids': torch.arange(len(feature['labels']), dtype=torch.long)  # 位置编码 [0, 1, 2, ...]
        }

    def _split_into_sections(self, normalized_feature: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        按 kv_cache_index 的变化点将序列切分为多个 section。
        
        这是 RosettaDataCollator 的核心设计：将一条序列按 KV-Cache 角色分段，
        使得同一 section 内的 token 具有相同的 KV-Cache 行为（如全部需要缓存或全部不需要）。
        例如一条典型对话会被切分为:
          section 0: instruction tokens (kv_cache_index = [1, 0])
          section 1: response tokens (kv_cache_index = [-1, 0])
        在 batch padding 时，每个 section 独立 padding，确保不同样本的同角色段对齐。

        Split sequence into sections based on kv_cache_index changes.
        
        Args:
            normalized_feature: 标准化后的特征字典
            
        Returns:
            section 列表，每个 section 包含该片段的 input_ids/attention_mask/labels 等
        """
        kv_idx = normalized_feature['kv_cache_index']
        
        # 寻找 kv_cache_index 发生变化的位置（即 section 边界）
        change_points = [0]  # 序列起始总是第一个 section 的边界
        for i in range(1, kv_idx.size(0)):
            if not torch.equal(kv_idx[i], kv_idx[i - 1]):
                change_points.append(i)  # 发现变化点
        change_points.append(kv_idx.size(0))  # 序列结束也是最后一个 section 的边界
        
        # 根据变化点切分所有字段
        sections = []
        for i in range(len(change_points) - 1):
            start, end = change_points[i], change_points[i + 1]
            section = {
                'input_ids': [ids[start:end] for ids in normalized_feature['input_ids']],      # 每个模型的片段
                'attention_mask': [mask[start:end] for mask in normalized_feature['attention_mask']],
                'labels': normalized_feature['labels'][start:end],
                'kv_cache_index': normalized_feature['kv_cache_index'][start:end],
                'position_ids': normalized_feature['position_ids'][start:end]
            }
            sections.append(section)
        
        return sections

    def _pad_sections(self, all_sections: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        对 batch 内所有样本的 section 进行 padding，确保结构统一。
        
        核心逻辑:
        1. 找到 batch 中最大的 section 数量
        2. 对每个 section 索引（0, 1, 2, ...），收集所有样本的该 section 数据
        3. 用 pad_sequence 对每个 section 内的 tensor 进行 padding
        4. 将所有 section 拼接为最终的 batch tensor

        Pad sections to ensure uniform structure across batch.
        
        Args:
            all_sections: 每个样本的 section 列表
            
        Returns:
            Padding 后的 batch 字典
        """
        max_sections = max(len(sections) for sections in all_sections)  # batch 中最大 section 数
        num_models = len(all_sections[0][0]['input_ids']) if all_sections else 1  # 模型数量
        
        # 初始化输出结构 - 保持每个模型的数据分开
        padded_output = {
            'input_ids_per_model': [[] for _ in range(num_models)],     # 每个模型一个列表
            'attention_mask_per_model': [[] for _ in range(num_models)],
            'labels': [],
            'kv_cache_index': [],
            'position_ids': []
        }
        
        # 按 section 索引逐一处理
        for sec_idx in range(max_sections):
            # 收集所有样本的第 sec_idx 个 section 数据
            section_data = self._collect_section_data(all_sections, sec_idx, num_models)
            # 对该 section 内的 tensor 进行 padding
            padded_section = self._pad_single_section(section_data, num_models)
            
            # 将 padding 后的结果添加到输出
            for model_idx in range(num_models):
                padded_output['input_ids_per_model'][model_idx].append(
                    padded_section['input_ids_per_model'][model_idx])
                padded_output['attention_mask_per_model'][model_idx].append(
                    padded_section['attention_mask_per_model'][model_idx])
            
            padded_output['labels'].append(padded_section['labels'])
            padded_output['kv_cache_index'].append(padded_section['kv_cache_index'])
            padded_output['position_ids'].append(padded_section['position_ids'])
        
        # 将所有 section 拼接为最终输出
        return self._finalize_output(padded_output, num_models, len(all_sections))

    def _collect_section_data(self, all_sections: List[List[Dict[str, Any]]], 
                            sec_idx: int, num_models: int) -> Dict[str, List]:
        """
        收集 batch 中所有样本的指定 section 数据。
        如果某个样本的 section 数量不足，用空 tensor 填充。

        Collect data for a specific section across all samples.
        
        Args:
            all_sections: 所有样本的 section 列表
            sec_idx: 要收集的 section 索引
            num_models: 模型数量
        Returns:
            该 section 在所有样本中的数据集合
        """
        # 每个模型单独收集，避免混淆
        section_data = {
            'input_ids_per_model': [[] for _ in range(num_models)],     # [[slm_seqs], [llm_seqs]]
            'attention_mask_per_model': [[] for _ in range(num_models)],
            'labels': [],
            'kv_cache_index': [],
            'position_ids': []
        }
        
        for sample_sections in all_sections:
            # 某些样本可能 section 数较少；缺失时创建空 tensor
            if sec_idx < len(sample_sections):
                sec = sample_sections[sec_idx]
                for model_idx in range(num_models):
                    section_data['input_ids_per_model'][model_idx].append(sec['input_ids'][model_idx])
                    section_data['attention_mask_per_model'][model_idx].append(sec['attention_mask'][model_idx])
                section_data['labels'].append(sec['labels'])
                section_data['kv_cache_index'].append(sec['kv_cache_index'])
                section_data['position_ids'].append(sec['position_ids'])
            else:
                # 缺失的 section 用空 tensor 填充，下游 pad_sequence 会自动处理
                for model_idx in range(num_models):
                    section_data['input_ids_per_model'][model_idx].append(torch.tensor([], dtype=torch.long))
                    section_data['attention_mask_per_model'][model_idx].append(torch.tensor([], dtype=torch.float))
                section_data['labels'].append(torch.tensor([], dtype=torch.long))
                section_data['kv_cache_index'].append(torch.empty((0, 2), dtype=torch.long))
                section_data['position_ids'].append(torch.tensor([], dtype=torch.long))
                
        return section_data

    def _pad_single_section(self, section_data: Dict[str, List], num_models: int) -> Dict[str, Any]:
        """
        对单个 section 内的 tensor 进行 padding。
        每个模型使用各自的 pad_token_id（SLM 和 LLM 的 vocab 可能不同）。
        labels 使用 -100 padding（不参与损失计算），kv_cache 使用 -1 padding。

        Pad tensors within a single section.
        
        Args:
            section_data: 该 section 在所有样本中的数据
            num_models: 模型数量
        Returns:
            padding 后的 section 数据
        """
        padded_input_ids_per_model = []
        padded_attention_mask_per_model = []
        
        for model_idx in range(num_models):
            # 每个模型使用自己的 pad_token_id 进行 padding
            pad_token_id = self.slm_pad_token_id if model_idx == 0 else self.llm_pad_token_id
            
            # 对该模型的 input_ids 进行 padding（batch_first=True 输出 shape: [batch, seq_len]）
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                section_data['input_ids_per_model'][model_idx], 
                batch_first=True, 
                padding_value=pad_token_id
            )
            padded_input_ids_per_model.append(padded_input_ids)
            
            # 对 attention_mask 进行 padding（padding 位置为 0）
            padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
                section_data['attention_mask_per_model'][model_idx],
                batch_first=True,
                padding_value=0
            )
            padded_attention_mask_per_model.append(padded_attention_mask)
        
        # labels 使用 -100 padding（CrossEntropyLoss 的忽略值）
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            section_data['labels'], batch_first=True, padding_value=-100)
        # kv_cache_index 使用 -1 padding（-1 表示无效/不参与 KV 缓存）
        padded_kv_cache = torch.nn.utils.rnn.pad_sequence(
            section_data['kv_cache_index'], batch_first=True, padding_value=-1)
        # position_ids 使用 0 padding
        padded_position_ids = torch.nn.utils.rnn.pad_sequence(
            section_data['position_ids'], batch_first=True, padding_value=0)
        
        return {
            'input_ids_per_model': padded_input_ids_per_model,
            'attention_mask_per_model': padded_attention_mask_per_model,
            'labels': padded_labels,
            'kv_cache_index': padded_kv_cache,
            'position_ids': padded_position_ids,
            'num_models': num_models
        }

    def _finalize_output(self, padded_output: Dict[str, List], 
                        num_models: int, batch_size: int) -> Dict[str, Any]:
        """
        最终化输出：将所有 section 的 padding 结果拼接为连续的 batch tensor。
        单模型时 input_ids 为单个 tensor，双模型时为 tensor 列表。

        Finalize the output by concatenating sections - keep models separate throughout.
        
        Args:
            padded_output: 按 section 组织的 padding 结果
            num_models: 模型数量
            batch_size: batch 大小
        Returns:
            最终化的 batch 字典
        """
        final_output = {}
        
        # === 处理 input_ids 和 attention_mask ===
        if num_models == 1:
            # 单模型：沿 seq_len 维度拼接所有 section
            final_output['input_ids'] = torch.cat(padded_output['input_ids_per_model'][0], dim=1)
            final_output['attention_mask'] = torch.cat(padded_output['attention_mask_per_model'][0], dim=1)
        else:
            # 双模型：每个模型分别拼接，保持为列表
            final_output['input_ids'] = [
                torch.cat(padded_output['input_ids_per_model'][model_idx], dim=1) 
                for model_idx in range(num_models)
            ]
            final_output['attention_mask'] = [
                torch.cat(padded_output['attention_mask_per_model'][model_idx], dim=1)
                for model_idx in range(num_models)
            ]
        
        # 拼接 labels 和 position_ids
        final_output['labels'] = torch.cat(padded_output['labels'], dim=1)
        final_output['position_ids'] = torch.cat(padded_output['position_ids'], dim=1)
        # kv_cache_index 保持为 section 列表（不按 dim=1 拼接，因为每个 section 有独立含义）
        final_output['kv_cache_index'] = padded_output['kv_cache_index']
        
        return final_output

    def _apply_length_constraints(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用 max_length 截断约束。
        如果序列长度超过 max_length，截断所有 tensor 和 kv_cache_index sections。

        Apply max_length truncation if specified.
        """
        if self.max_length is None:
            return output
        
        # 确定当前序列长度
        if isinstance(output['input_ids'], list):
            seq_length = output['input_ids'][0].size(1)  # 双模型取第一个模型的长度
        else:
            seq_length = output['input_ids'].size(1)
        
        if seq_length <= self.max_length:
            return output  # 不需要截断
        
        # 截断 input_ids 和 attention_mask
        if isinstance(output['input_ids'], list):
            output['input_ids'] = [ids[:, :self.max_length] for ids in output['input_ids']]
            output['attention_mask'] = [mask[:, :self.max_length] for mask in output['attention_mask']]
        else:
            output['input_ids'] = output['input_ids'][:, :self.max_length]
            output['attention_mask'] = output['attention_mask'][:, :self.max_length]
        
        output['labels'] = output['labels'][:, :self.max_length]
        output['position_ids'] = output['position_ids'][:, :self.max_length]
        
        # 截断 kv_cache_index sections（按位置逐个 section 截断）
        output['kv_cache_index'] = self._truncate_kv_cache_sections(
            output['kv_cache_index'], self.max_length)
        
        return output

    def _truncate_kv_cache_sections(self, kv_cache_sections: List[torch.Tensor], 
                                  max_length: int) -> List[torch.Tensor]:
        """
        截断 kv_cache sections 使其总长度不超过 max_length。
        按顺序遍历 sections，累计长度达到 max_length 时停止。

        Truncate kv_cache sections to fit within max_length.
        """
        truncated_sections = []
        current_pos = 0  # 当前累计位置
        
        for section in kv_cache_sections:
            section_length = section.size(1)  # 该 section 的序列长度
            remaining_length = max_length - current_pos  # 剩余可用长度
            
            if remaining_length <= 0:
                break  # 已用完所有长度，丢弃后续 section
            elif remaining_length >= section_length:
                truncated_sections.append(section)  # 完整保留
                current_pos += section_length
            else:
                # 部分截断：只取前 remaining_length 个 token
                truncated_section = section[:, :remaining_length]
                truncated_sections.append(truncated_section)
                break
        
        return truncated_sections

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        主 collation 函数：将样本列表组合为 padding 后的 batch。

        完整的 4 步流水线:
        1. 标准化输入格式（统一单/双模型）
        2. 按 kv_cache_index 切分 section
        3. 对 section 进行 padding
        4. 应用长度约束

        Main collation function with improved logic.
        
        Args:
            features: 来自 Dataset.__getitem__ 的样本字典列表
            
        Returns:
            Padding 后的 batch 字典，可直接送入模型
        """
        if not features:
            return {}
        
        # Step 1: 标准化所有样本的输入格式
        normalized_features = [self._normalize_input_format(feat) for feat in features]
        
        # Step 2: 将每个样本按 kv_cache_index 切分为 sections
        all_sections = [self._split_into_sections(feat) for feat in normalized_features]
        
        # Step 3: 对 sections 进行 padding（每个 section 独立 padding 后拼接）
        output = self._pad_sections(all_sections)
        
        # Step 4: 应用 max_length 截断（如果需要）
        output = self._apply_length_constraints(output)
        
        return output


# ============================================================
# BaselineDataCollator: 基线模型数据整理器
# 简单的 padding collator，不处理 KV-Cache section 切分。
# 用于不需要 C2C 特性的基线模型训练。
# ============================================================
class BaselineDataCollator:
    """
    基线模型数据整理器。
    对 input_ids 和 labels 进行简单 padding，生成 attention_mask。
    不包含 KV-Cache 索引处理或 section 切分逻辑。

    输出字典:
        - input_ids: Tensor [batch, seq_len]
        - labels: Tensor [batch, seq_len]
        - attention_mask: Tensor [batch, seq_len]
    """
    
    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: Optional[int] = None):
        """
        Args:
            tokenizer: 用于获取 pad_token_id 的 tokenizer
            pad_to_multiple_of: 将序列长度填充到此值的倍数
        """
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """将样本列表 padding 为 batch tensor。"""
        # 提取 input_ids 和 labels
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        # 找到 batch 中的最大长度
        max_length = max(len(ids) for ids in input_ids)
        
        # 应用 pad_to_multiple_of 对齐
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # 逐样本 padding
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        
        for ids, lbls in zip(input_ids, labels):
            # 用 pad_token_id 填充 input_ids
            padded_ids = ids + [self.tokenizer.pad_token_id] * (max_length - len(ids))
            batch_input_ids.append(padded_ids)
            
            # 用 -100 填充 labels（padding 位置不参与损失计算）
            padded_labels = lbls + [-100] * (max_length - len(lbls))
            batch_labels.append(padded_labels)
            
            # 生成 attention_mask：实际 token 为 1，padding 为 0
            attention_mask = [1] * len(ids) + [0] * (max_length - len(ids))
            batch_attention_mask.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }



"""
============================================================
辅助函数 (Helper Functions)
工厂函数，通过字符串名称创建数据集实例。
============================================================
"""


def create_dataset(dataset_type: str, **kwargs) -> Dataset:
    """
    数据集工厂函数。
    根据字符串类型名称从全局注册表中查找并创建对应的数据集实例。
    支持精确匹配和大小写不敏感匹配。

    Factory function to create a dataset based on type.
    
    Args:
        dataset_type: 数据集类型字符串（如 "MMLUChatDataset"、"OpenHermesChatDataset" 等）
        **kwargs: 传递给数据集构造函数的额外参数
        
    Returns:
        对应数据集类的实例
        
    Raises:
        ValueError: 如果 dataset_type 未在注册表中找到
        
    用法示例:
        dataset = create_dataset("MMLUChatDataset", split="test", num_samples=100)
    """
    # 首先精确匹配
    if dataset_type in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_type](**kwargs)
    
    # 然后大小写不敏感匹配
    dataset_type_lower = dataset_type.lower()
    if dataset_type_lower in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_type_lower](**kwargs)
    
    # 未找到时，列出所有可用的数据集类型
    valid_options = list(
        set([name for name, cls in DATASET_REGISTRY.items() if name == cls.__name__])
    )  # 只保留类名（排除小写别名）
    raise ValueError(
        f"Unknown dataset type: {dataset_type}. Valid options are: {valid_options}"
    )