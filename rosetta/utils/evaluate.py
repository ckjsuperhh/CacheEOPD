"""
Common evaluation utilities for benchmark tasks.

This module provides shared functions for model evaluation across different benchmarks
like MMLU-Redux and MMMLU.

=== 中文说明 ===
评测工具模块：为各种基准测试（如 MMLU、MMMLU）提供通用的评测辅助函数。

本模块是 C2C (Cache-to-Cache) 框架中评测流程的核心工具集，主要负责：
1. 提示词构建 (build_prompt): 根据数据集和语言区域生成评测用的 prompt
2. 答案解析与提取 (parse_answer, extract_answer_from_content): 从模型输出中自动解析出选项答案
3. 模型加载 (load_hf_model, load_rosetta_model, load_oracle_rosetta_model): 加载不同类型的模型
   - HuggingFace 标准模型
   - Rosetta 模型（带投影器的 C2C 模型，支持多教师模型融合）
   - Oracle Rosetta 模型（使用 ground-truth KV-Cache 的参考模型）
4. 答案生成 (generate_answer_with_logits, generate_answer_with_generate): 两种生成策略
   - logits 方法：直接比较选项 token 的概率分布（更快、更确定性）
   - generate 方法：自回归生成文本后解析答案（更灵活、支持 CoT 推理）
5. 配置管理 (apply_generation_config, set_default_chat_template): 生成参数和模板管理

与其他模块的关系：
- rosetta.model.projector: 加载投影器（用于 KV-Cache 跨模型映射）
- rosetta.model.wrapper.RosettaModel: Rosetta 模型的包装器
- rosetta.model.oracle.OracleRosettaModel: Oracle 模型的包装器（用于上界评测）
- 被 rosetta.eval 下的各评测脚本调用（如 mmlu_eval.py、mmmlu_eval.py 等）
"""

import re
import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入 C2C 框架的核心组件
from rosetta.model.projector import load_projector        # 加载投影器（实现 KV-Cache 跨模型映射）
from rosetta.model.wrapper import RosettaModel            # Rosetta 模型包装器（整合 SLM + 多个 LLM + 投影器）
from rosetta.model.oracle import OracleRosettaModel       # Oracle 模型包装器（使用 ground-truth cache 的参考实现）

def build_prompt(dataset: str, locale: str, question: str, choices: str, use_cot: bool, use_template: bool = True) -> str:
    """
    Build a localized prompt for a given dataset and locale.
    
    构建评测用的提示词（prompt），将题目和选项填入预定义的模板中。
    支持 Chain-of-Thought (CoT) 和普通直接回答两种模式。

    Currently supports:
    - dataset: "mmmlu"
      - locale: "SW_KE" (Swahili). Other locales fall back to English.

    Args:
        dataset: 数据集标识符，如 "mmmlu"（多语言 MMLU）、"mmlu" 等
        locale: 语言区域代码，如 "SW_KE"（斯瓦希里语）
        question: 题目文本
        choices: 格式化后的选项字符串（如 "A. xxx\nB. xxx\n..."）
        use_cot: 是否使用 Chain-of-Thought（逐步推理）指令。
                 True: 提示模型先推理再给出答案
                 False: 要求模型只输出答案，不输出解释
        use_template: 是否使用完整模板。False 时仅拼接题目和选项，不包含指令说明

    Returns:
        构建好的 prompt 字符串，可直接用于模型输入
    """
    
        # 统一的默认英文模板（MMLU 和 MMMLU 共用）
        # Unified default English templates (shared by MMLU and MMMLU)
    if not use_cot:
        # 非 CoT 模式：要求模型严格只输出答案，不附带任何解释
        # 这对于 logits 方法评测尤为重要，因为模型输出越短，答案提取越准确
        template = """Accurately answer the following question:

{{question}}

Choices:
{{choices}}

Instructions:
- Carefully read the question and all options.
- Select the single most correct answer.
- Respond ONLY in the following format: "The correct answer is A/B/C/D".
- Do not include any explanations, additional text, or punctuation besides the answer.

The correct answer is"""

    else:
        # CoT 模式：允许模型先进行逐步推理，再给出最终答案
        # 这种方式通常能提升模型在困难题目上的表现
        template = """Accurately answer the following question:
                   
{{question}}

Choices:
{{choices}}

Instructions:
- Carefully read the question and all options.
- Let's think step by step and explain your reasoning briefly.
- Then give the final answer starting with The correct answer is"""

    # 将模板中的占位符替换为实际的题目和选项
    prompt = template.replace("{{question}}", question)
    prompt = prompt.replace("{{choices}}", choices)

    # 如果不使用模板，则只拼接最基本的题目和选项（用于某些特殊评测场景）
    if not use_template:
        prompt = question + "\n\nChoices:\n" + choices

    return prompt


def parse_answer(answer_str: str) -> List[str]:
    """
    Parse answer string to extract valid answer options.
    Converts digits 0/1/2/3 to letters A/B/C/D.
    
    从答案字符串中解析出有效的选项字母。
    某些数据集中答案以数字 0/1/2/3 存储，此函数将其统一转换为 A/B/C/D。
    
    Args:
        answer_str: 包含答案数字的字符串，如 "0"、"2" 等
        
    Returns:
        解析后的答案字母列表（已排序去重），如 ["A"]、["C"]
        如果输入不是字符串则返回空列表
    """
    if not isinstance(answer_str, str):
        return []
    # 提取字符串中所有有效的数字字符（仅接受 0/1/2/3，对应 A/B/C/D 四个选项）
    valid_digits = [c for c in answer_str if c in {'0','1','2','3'}]
    # 将数字转换为字母（0->A, 1->B, 2->C, 3->D），去重后排序返回
    return sorted(list({
        chr(65 + int(d))  # 0->A, 1->B, 2->C, 3->D（利用 ASCII 码偏移）
        for d in valid_digits
    }))


def extract_answer_from_content(text: str) -> Optional[str]:
    """
    Extract answer from model output with robust multi-pattern matching.
    Supports multiple languages and response formats.
    
    从模型的生成文本中提取答案选项（A/B/C/D）。
    采用多级策略逐步提取，确保在各种输出格式下都能正确提取答案：
      第1级：使用预定义的正则模式匹配常见答案格式（支持英语和斯瓦希里语）
      第2级：查找文本末尾或明显位置的独立选项字母
      第3级：宽松匹配所有 A-D 字母（排除数学表达式干扰）
      第4级：终极兜底——从文本末尾反向搜索第一个 A-D 字母
    
    Args:
        text: 模型生成的输出文本
        
    Returns:
        提取到的答案字母（'A'/'B'/'C'/'D'），无法提取时返回 None
    """
    text = text.strip()
    if not text:
        return None

    # 定义多语言、多格式的答案匹配正则模式
    # Define multiple answer patterns for different languages and formats
    answer_patterns = [
        # 英语模式 —— 匹配各种常见的答案表述方式
        # English patterns
        r'Answer:\s*(.*)',
        r'answer:\s*(.*)',
        r'ANSWER:\s*(.*)',
        r'Your answer:\s*(.*)',
        r'your answer:\s*(.*)',
        r'YOUR ANSWER:\s*(.*)',
        r'The answer is\s*(.*)',
        r'the answer is\s*(.*)',
        r'THE ANSWER IS\s*(.*)',
        r'Correct answer is\s*(.*)',
        r'correct answer is\s*(.*)',
        r'Correct answer is:\s*(.*)',
        r'correct answer is:\s*(.*)',
        r'Correct answer:\s*(.*)',
        r'correct answer:\s*(.*)',
        r'CORRECT ANSWER:\s*(.*)',
        
        # 斯瓦希里语模式 —— MMMLU 多语言评测中使用
        # Swahili patterns
        r'Jibu lako:\s*(.*)',       # "Your answer:" 的斯瓦希里语
        r'jibu lako:\s*(.*)',
        r'JIBU LAKO:\s*(.*)',
        r'Jibu:\s*(.*)',            # "Answer:" 的斯瓦希里语
        r'jibu:\s*(.*)',
        r'JIBU:\s*(.*)',
        r'Jibu sahihi:\s*(.*)',     # "Correct answer:" 的斯瓦希里语
        r'jibu sahihi:\s*(.*)',
        r'JIBU SAHIHI:\s*(.*)',
        
        # 其他通用模式 —— 捕获更多可能的输出格式
        # Other common patterns
        r'Response:\s*(.*)',
        r'response:\s*(.*)',
        r'RESPONSE:\s*(.*)',
        r'Choice:\s*(.*)',
        r'choice:\s*(.*)',
        r'CHOICE:\s*(.*)',
        r'Option:\s*(.*)',
        r'option:\s*(.*)',
        r'OPTION:\s*(.*)',
    ]
    
    # === 第1级策略：尝试匹配预定义的答案模式 ===
    # 1. Try to match any of the answer patterns
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer_part = match.group(1).strip()
            # 在匹配到的答案部分中查找第一个 A-D 字母
            # Search for first A-D letter in the matched part
            for char in answer_part:
                if char in {'A', 'B', 'C', 'D'}:
                    return char
    
    # === 第2级策略：查找独立出现的选项字母（通常在文本末尾） ===
    # 2. Look for standalone A-D letters that are likely answers
    # Prioritize letters at the end of text or with clear answer-like context
    standalone_patterns = [
        r'\b([A-D])(?:\s*[.,!?:)]?\s*$)',  # A-D 在文本末尾，可带标点（如 "A." 或 "B)"）
        r'\b([A-D])(?:\s*[.,!?:)]\s)',     # A-D 后跟标点和空格（如 "C. " ）
        r'(?:^|\s)([A-D])(?:\s*$)',        # A-D 在行首或行尾的单词边界处
    ]
    
    for pattern in standalone_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # 检查文本是否看起来像数学表达式而非答案
            # 数学题目中 A-D 可能是变量名（如公式 "A = B + C"），需要排除
            # Check if this looks like mathematical expressions rather than answers
            math_indicators = ['+', '-', '*', '/', '=', '^', 'x^', 'y^', 'z^', 'mod', 'sqrt', 'sin', 'cos', 'tan']
            has_math = any(indicator in text for indicator in math_indicators)
            # 检查文本中是否有答案相关的关键词（如 "answer"、"jibu" 等）
            has_answer_indicators = any(phrase in text.lower() for phrase in ['jibu', 'answer', 'choice', 'option', 'response', 'correct', 'sahihi'])
            
            # 如果文本包含数学符号但不包含答案关键词，说明 A-D 可能是数学变量，跳过
            # If it has math indicators but no answer indicators, it's likely mathematical notation
            if has_math and not has_answer_indicators:
                continue  # Skip this match, try next pattern / 跳过此匹配，尝试下一个模式
            
            # 返回最后一个匹配的字母（通常最后出现的答案更可能是最终答案）
            return matches[-1].upper()
    
    # === 第3级策略：宽松匹配所有 A-D 单词边界字母 ===
    # 3. Fallback: find all A-D letters but be more selective
    all_letters = re.findall(r'\b([A-D])\b', text, re.IGNORECASE)
    if all_letters:
        # 同样排除数学表达式的干扰
        # Check if this looks like mathematical expressions rather than answers
        math_indicators = ['+', '-', '*', '/', '=', '^', 'x^', 'y^', 'z^', 'mod', 'sqrt', 'sin', 'cos', 'tan']
        has_math = any(indicator in text for indicator in math_indicators)
        has_answer_indicators = any(phrase in text.lower() for phrase in ['jibu', 'answer', 'choice', 'option', 'response', 'correct', 'sahihi'])
        
        # 数学表达式场景：无法可靠提取答案，返回 None
        # If it has math indicators but no answer indicators, it's likely mathematical notation
        if has_math and not has_answer_indicators:
            return None
        
        # 否则返回最后一个找到的字母（通常最终答案出现在文本末尾）
        # Otherwise, return the last letter found
        return all_letters[-1].upper()
    
    # === 第4级策略（终极兜底）：从文本末尾反向搜索任意 A-D 字母 ===
    # 3. Search backwards for any A-D letter as fallback
    for char in reversed(text):
        if char in {'A', 'B', 'C', 'D'}:
            return char

    # 所有策略均未找到答案，返回 None
    return None


def apply_generation_config(model: Any, generation_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Apply generation configuration to a model and handle sampling parameters.
    
    将生成配置应用到模型上，并处理采样参数的兼容性问题。
    当 do_sample=False（贪心解码）时，自动清除采样相关参数以避免 transformers 库的警告。
    
    This function applies the provided generation config to the model and removes
    sampling parameters (temperature, top_p, top_k, min_p) when do_sample=False
    to avoid warnings from the transformers library. If no config is provided,
    it defaults to greedy decoding with cleaned sampling parameters.
    
    Args:
        model: 模型对象，需具有 generation_config 属性（HuggingFace 标准接口）
        generation_config: 生成配置字典，可选键值包括：
            - do_sample (bool): 是否启用采样（False=贪心解码）
            - temperature (float): 采样温度
            - top_p (float): nucleus sampling 阈值
            - top_k (int): top-k sampling 的 k 值
            - max_new_tokens (int): 最大生成 token 数
            如果为 None，默认使用贪心解码（do_sample=False）
    """
    # 如果模型没有 generation_config 属性，直接返回（兼容非标准模型）
    if not hasattr(model, 'generation_config'):
        return
    
    # If no config provided, default to greedy decoding
    # 未提供配置时，默认使用贪心解码
    if not generation_config:
        generation_config = {'do_sample': False}
    
    # 将配置中的所有参数逐一设置到模型的 generation_config 上
    # Apply all configuration parameters
    for key, value in generation_config.items():
        setattr(model.generation_config, key, value)
    
    # 当 do_sample=False 时，禁用采样相关参数以避免 transformers 库的警告
    # transformers 在 do_sample=False 但设置了 temperature 等参数时会打印警告
    # Disable sampling parameters if do_sample=False to avoid warnings
    # We set them to None instead of deleting, since some model code may
    # access these attributes unconditionally.
    if not generation_config.get('do_sample', True):
        sampling_params = ['temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty']
        for param in sampling_params:
            try:
                # 设为 None 而非删除，因为某些模型代码可能会无条件访问这些属性
                setattr(model.generation_config, param, None)
            except Exception:
                # If the backend does not allow setting, ignore silently
                # 某些后端可能不允许修改这些属性，静默忽略
                pass


def set_default_chat_template(tokenizer, model_name: str):
    """
    Set default chat template for models without one.
    
    为没有自带聊天模板（chat_template）的模型设置默认模板。
    聊天模板用于将对话消息列表格式化为模型可理解的输入字符串。
    
    特殊处理：
    - UlizaLlama3 模型使用 LLaMA3 风格模板（<|start_header_id|> 格式）
    - 其他模型使用通用的 "### Human: / ### Assistant:" 格式
    
    Args:
        tokenizer: HuggingFace tokenizer 对象
        model_name: 模型名称或路径，用于判断是否需要特殊模板
    """
    if tokenizer.chat_template is None:
        # 特殊处理 UlizaLlama3 模型（非洲语言微调的 LLaMA3 模型）
        # 使用 LLaMA3 原生的 special token 格式
        if "UlizaLlama3".lower() in model_name.lower():
            tokenizer.chat_template = (
                "{%- for message in messages %}"
                "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}"
                "{{- message['content'] }}"
                "{{- '<|eot_id|>' }}"
                "{%- endfor %}"
                "{%- if add_generation_prompt %}"
                "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
                "{%- endif %}"
            )
        else:
            # 对于其他没有聊天模板的模型，设置通用的默认模板
            print(f"Model {model_name} has no chat template, setting default template...")
            default_template = """{% for message in messages %}{% if message['role'] == 'user' %}### Human: {{ message['content'] }}{% elif message['role'] == 'assistant' %}### Assistant: {{ message['content'] }}{% endif %}{% if not loop.last %}
    {% endif %}{% endfor %}{% if add_generation_prompt %}
    ### Assistant:{% endif %}"""
            tokenizer.chat_template = default_template
            print("Default chat template has been set.")
    else:
        # 模型已有聊天模板，无需修改
        print(f"Model {model_name} already has a chat template.")


def load_hf_model(model_name: str, device: torch.device, generation_config: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
    """
    Load Hugging Face model and tokenizer.
    
    加载标准的 HuggingFace 因果语言模型（CausalLM）及其 tokenizer。
    用于评测中的 baseline 模型加载（即不使用 C2C 投影器的原始模型）。
    
    特殊处理：
    - google/gemma-3-1b-it 模型需要设置 sliding_window=4096 和 dynamo cache_size_limit
    - 所有模型使用 bfloat16 精度以节省显存
    - tokenizer 设为左填充（padding_side='left'），适配 batch 生成
    
    Args:
        model_name: 模型名称（HuggingFace Hub ID）或本地路径
        device: 加载模型的目标设备（如 torch.device('cuda:0')）
        generation_config: 可选的生成配置字典
        
    Returns:
        (model, tokenizer) 元组
    """
    # 加载 tokenizer，trust_remote_code=True 支持自定义 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_name),
        trust_remote_code=True,
        padding_side='left'  # 左填充，适合自回归模型的 batch 推理
    )

    # 如果 tokenizer 没有 pad_token，使用 eos_token 替代
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 检查并设置聊天模板（某些模型可能没有自带模板）
    # Check and set chat template
    set_default_chat_template(tokenizer, model_name)

    # 特殊处理 Gemma-3 模型：需要设置 dynamo 缓存大小限制和滑动窗口
    if model_name == "google/gemma-3-1b-it":
        torch._dynamo.config.cache_size_limit = 64  # 增大 dynamo 缓存以避免重编译
        model = AutoModelForCausalLM.from_pretrained(
            str(model_name),
            torch_dtype=torch.bfloat16,           # 使用 bfloat16 精度
            device_map={"": device},               # 将模型放在指定设备上
            sliding_window=4096                    # Gemma-3 的滑动窗口注意力
        ).eval()
    else:
        # 标准模型加载
        model = AutoModelForCausalLM.from_pretrained(
            str(model_name),
            torch_dtype=torch.bfloat16,
            device_map={"": device}
    ).eval()
    
    # 应用生成配置（如 do_sample、temperature 等）
    # Apply generation config
    apply_generation_config(model, generation_config)
    
    return model, tokenizer


def load_rosetta_model(model_config: Dict[str, Any], eval_config: Dict[str, Any], 
                      device: torch.device, generation_config: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
    """
    Load Rosetta model with projectors.
    
    加载 C2C (Cache-to-Cache) Rosetta 模型。这是本模块最核心的函数，负责：
    1. 加载 SLM（Small Language Model，学生模型/目标模型）
    2. 加载一个或多个 LLM（Large Language Model，教师模型）
    3. 从检查点目录加载投影器（Projector），用于 KV-Cache 跨模型映射
    4. 加载投影器配置（projector_config.json），定义哪些层之间进行 KV-Cache 注入
    5. 将所有组件组装成 RosettaModel 对象
    
    C2C 核心思想：通过投影器将教师模型（LLM）的 KV-Cache 映射到学生模型（SLM）的空间，
    使 SLM 可以直接利用 LLM 的缓存来提升推理能力，无需重新训练 SLM。
    
    Args:
        model_config: 模型配置字典，包含 rosetta_config 子字典：
            - base_model: SLM 模型路径
            - teacher_model: 教师模型配置（字符串=单模型，字典=多模型）
            - checkpoints_dir: 检查点目录路径（列表，每个元素对应一个教师模型）
            - multi_source_fusion_mode: 多教师融合模式（默认 "sequential"）
            - include_response: 是否包含响应部分的 KV-Cache
        eval_config: 评测配置字典（向后兼容，提供 fallback 的 checkpoints_dir）
        device: 目标设备
        generation_config: 可选的生成配置
        
    Returns:
        (rosetta_model, slm_tokenizer) 元组
    """
    # === 步骤1：解析配置，构建 LLM 配置列表 ===
    # Prefer checkpoints_dir under model.rosetta_config; fall back to eval config for backward compatibility
    rosetta_config = model_config["rosetta_config"]
    slm_model_path = rosetta_config["base_model"]             # SLM（学生模型）的路径
    teacher_model_config = rosetta_config["teacher_model"]    # 教师模型配置

    # 构建 LLM 配置列表：[(model_path, checkpoint_dir), ...]
    # Dict of models with list of checkpoints: {"model_name": "model_path", ...} + ckpt: ["ckpt1", "ckpt2"]
    
    llm_configs = []  # List of (model_path, checkpoint_dir) tuples / (模型路径, 检查点目录) 元组列表
    
    if isinstance(teacher_model_config, str):
        # 单教师模型模式（向后兼容）
        # Single model - backward compatibility
        checkpoint_dir = rosetta_config.get("checkpoints_dir", eval_config.get("checkpoints_dir"))
        llm_configs.append((teacher_model_config, checkpoint_dir))
    
    elif isinstance(teacher_model_config, dict):
        # 多教师模型模式：teacher_model 是一个字典 {模型名: 模型路径}
        # 同时 checkpoints_dir 是一个列表，每个元素对应一个教师模型的检查点目录
        # Format 4: Dict format with separate ckpt list
        # teacher_model: {"model1_name": "model1_path", "model2_name": "model2_path"}
        # ckpt: ["ckpt1_path", "ckpt2_path"]
        checkpoints_dir = rosetta_config.get("checkpoints_dir", [])
        model_items = list(teacher_model_config.items())
        
        # 校验：检查点数量必须与教师模型数量一致
        if len(checkpoints_dir) != len(model_items):
            raise ValueError(f"Number of checkpoints ({len(checkpoints_dir)}) must match number of models ({len(model_items)})")
        
        for (model_name, model_path), ckpt_dir in zip(model_items, checkpoints_dir):
            llm_configs.append((model_path, ckpt_dir))

    # === 步骤2：加载 SLM 的 tokenizer 和模型 ===
    # Load tokenizer
    slm_tokenizer = AutoTokenizer.from_pretrained(str(slm_model_path))
    set_default_chat_template(slm_tokenizer, slm_model_path)
    
    # Load SLM model（加载学生模型）
    slm_model = AutoModelForCausalLM.from_pretrained(
        str(slm_model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    
    # Apply generation config to SLM（应用生成配置到学生模型）
    apply_generation_config(slm_model, generation_config)
    
    # === 步骤3：加载所有教师模型（LLM） ===
    # Load LLM models
    llm_models = []
    for llm_model_path, _ in llm_configs:
        # 特殊处理 Gemma-3 模型
        if llm_model_path == "google/gemma-3-1b-it":
            llm_model = AutoModelForCausalLM.from_pretrained(
                str(llm_model_path),
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                sliding_window=4096
            ).eval()
        else:
            llm_model = AutoModelForCausalLM.from_pretrained(
                str(llm_model_path),
                torch_dtype=torch.bfloat16,
                device_map={"": device}
            ).eval()
        
        # Apply generation config to LLM（应用生成配置到教师模型）
        apply_generation_config(llm_model, generation_config)
        llm_models.append(llm_model)
    
    # === 步骤4：从各教师模型的检查点目录加载投影器 ===
    # 投影器是 C2C 的核心组件，负责将教师模型（LLM）某层的 KV-Cache
    # 映射为学生模型（SLM）可使用的格式
    # 每个检查点目录使用标准命名：projector_{idx}.pt（权重）+ projector_{idx}.json（结构配置）
     # Load projectors for each LLM from their respective checkpoint directories
    # Each checkpoint directory contains standard format: projector_{idx}.pt
    projector_list = []
    num_llms = len(llm_models)
    
    # 记录每个 LLM 的投影器偏移量（用于后续合并多个教师模型的投影器配置）
    # 例如：LLM_0 有 3 个投影器（索引 0-2），则 LLM_1 的投影器从索引 3 开始
    # Track projector offset for each LLM (for config index adjustment)
    projector_offsets = [0]
    
    for llm_idx, (_, checkpoint_dir) in enumerate(llm_configs):
        # 统计该检查点目录下有多少个投影器文件
        # Load projectors from this LLM's checkpoint directory
        # Standard naming: projector_{proj_idx}.pt / .json
        num_projectors = len([f for f in os.listdir(checkpoint_dir) 
                             if re.match(r"projector_\d+\.pt", f)])
        
        for proj_idx in range(num_projectors):
            # 先从 JSON 文件加载投影器的结构配置（层类型、维度等）
            json_cfg = os.path.join(checkpoint_dir, f"projector_{proj_idx}.json")
            proj = load_projector(json_cfg)
            proj = proj.to(device)
            # 再从 .pt 文件加载投影器的训练权重
            pt_path = os.path.join(checkpoint_dir, f"projector_{proj_idx}.pt")
            if os.path.exists(pt_path):
                state_dict = torch.load(pt_path, map_location=device)
                proj.load_state_dict(state_dict, strict=False)
            projector_list.append(proj)
        
        # 记录当前 LLM 处理完后投影器列表的总长度（即下一个 LLM 的起始偏移）
        # Record offset for next LLM
        projector_offsets.append(len(projector_list))

    # === 步骤5：组装 RosettaModel ===
    # Initialize Rosetta model
    # model_list: [slm_model, llm_model_1, llm_model_2, ...]
    # 模型列表中，索引 0 是 SLM，后续索引依次是各教师模型
    model_list = [slm_model] + llm_models
    
    # 从配置中获取多源融合模式和响应包含选项
    # Get multi-source fusion mode from config (default to "sequential" for backward compatibility)
    multi_source_fusion_mode = rosetta_config.get("multi_source_fusion_mode", "sequential")
    include_response = rosetta_config.get("include_response", False)
    
    # 创建 RosettaModel 实例
    # base_model_idx=0 表示 model_list[0] 即 SLM 为基础模型
    rosetta_model = RosettaModel(
        model_list=model_list,
        base_model_idx=0,
        projector_list=projector_list,
        include_response=include_response,
        multi_source_fusion_mode=multi_source_fusion_mode,
    ).to(device).eval()

    # === 步骤6：加载并合并投影器映射配置 ===
    # 投影器配置定义了哪些层之间建立 KV-Cache 映射关系
    # 格式：{target_model_idx: {source_model_idx: {target_layer_idx: [(source_layer_idx, projector_idx), ...]}}}
    # Load projector mapping configs from each LLM's checkpoint directory
    # Each directory has standard config file: projector_config.json
    
    # 辅助函数：调整配置中的投影器索引（因为多个教师模型的投影器被合并到一个列表中）
    # Helper function to adjust config indices for flattened lists
    def adjust_config_indices(config_dict, proj_offset, actual_source_idx=None):
        """Adjust projector indices in config dict by adding offsets.
        
        调整投影器配置中的索引值：
        - 投影器索引加上偏移量（因为多个教师的投影器被展平到一个列表）
        - 源模型索引可被重映射为实际的 model_list 索引
        
        Args:
            config_dict: 原始配置字典
            proj_offset: 投影器索引的偏移量
            actual_source_idx: 若提供，将所有 source_model_idx 重映射为此值
        """
        adjusted = {}
        for target_model_idx, sources in config_dict.items():
            adjusted[int(target_model_idx)] = {}
            for source_model_idx, layers in sources.items():
                # 如果提供了 actual_source_idx，使用它；否则保留原始索引
                # Use actual_source_idx if provided, otherwise keep original
                actual_src_idx = actual_source_idx if actual_source_idx is not None else int(source_model_idx)
                adjusted[int(target_model_idx)][actual_src_idx] = {}
                for target_layer_idx, mappings in layers.items():
                    adjusted_mappings = []
                    for source_layer_idx, idx in mappings:
                        # 关键：将投影器索引加上偏移量，使其对应到合并后的 projector_list
                        # Adjust the projector index
                        adjusted_idx = idx + proj_offset
                        adjusted_mappings.append((source_layer_idx, adjusted_idx))
                    adjusted[int(target_model_idx)][actual_src_idx][int(target_layer_idx)] = adjusted_mappings
        return adjusted
    
    # 逐个加载每个教师模型的投影器配置并合并到 RosettaModel 中
    # Load and merge configs from each LLM's checkpoint directory
    for llm_idx, (_, checkpoint_dir) in enumerate(llm_configs):
        proj_cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
        
        # 计算该教师模型在 model_list 中的实际索引
        # llm_idx=0 对应 model_list[1]，llm_idx=1 对应 model_list[2]，依此类推
        # Actual source model index in model_list (llm_idx=0 -> model_list[1], llm_idx=1 -> model_list[2], etc.)
        actual_source_model_idx = llm_idx + 1
        
        # Load projector config
        if os.path.exists(proj_cfg_path):
            with open(proj_cfg_path, 'r') as f:
                config = json.load(f)
                # 调整投影器索引并设置正确的源模型索引
                # Adjust projector indices based on offset and set actual source_idx
                adjusted_config = adjust_config_indices(config, projector_offsets[llm_idx], actual_source_model_idx)
                # 将调整后的配置合并到 rosetta_model 的 projector_dict 中
                # Merge into rosetta_model.projector_dict
                for target_idx, sources in adjusted_config.items():
                    if target_idx not in rosetta_model.projector_dict:
                        rosetta_model.projector_dict[target_idx] = {}
                    for source_idx, layers in sources.items():
                        if source_idx not in rosetta_model.projector_dict[target_idx]:
                            rosetta_model.projector_dict[target_idx][source_idx] = {}
                        # 使用 update 合并同一目标层的映射配置
                        rosetta_model.projector_dict[target_idx][source_idx].update(layers)

    return rosetta_model, slm_tokenizer


def load_oracle_rosetta_model(model_config: Dict[str, Any], eval_config: Dict[str, Any], 
                      device: torch.device) -> Tuple[Any, Any]:
    """
    Load Rosetta model with projectors.
    
    加载 Oracle Rosetta 模型（用于评测上界参考）。
    
    Oracle 模型与普通 Rosetta 模型的区别：
    - Oracle 模型使用 ground-truth（真实的）KV-Cache 而非通过投影器生成的 cache
    - 因此 Oracle 模型代表了 C2C 方法的性能上界（理论上最好的效果）
    - 用于对比分析：Rosetta 模型 vs Oracle 模型，可以衡量投影器的质量
    
    注意：此函数仅支持单教师模型（不支持多教师融合）。
    
    Args:
        model_config: 模型配置字典，包含 rosetta_config 子字典
        eval_config: 评测配置字典（向后兼容）
        device: 目标设备
        
    Returns:
        (oracle_rosetta_model, slm_tokenizer) 元组
    """
    # Prefer checkpoints_dir under model.rosetta_config; fall back to eval config for backward compatibility
    # 优先从 rosetta_config 获取检查点目录，否则回退到 eval_config（向后兼容）
    rosetta_config = model_config["rosetta_config"]
    checkpoint_dir = rosetta_config.get("checkpoints_dir", eval_config.get("checkpoints_dir"))
    if checkpoint_dir is None:
        raise KeyError("checkpoints_dir must be provided under model.rosetta_config (preferred) or eval config (legacy)")
    slm_model_path = rosetta_config["base_model"]       # SLM（学生模型）路径
    llm_model_path = rosetta_config["teacher_model"]    # 教师模型路径（Oracle 模式仅支持单教师）

    # 加载 SLM 的 tokenizer
    # Load tokenizer
    slm_tokenizer = AutoTokenizer.from_pretrained(str(slm_model_path))
    set_default_chat_template(slm_tokenizer, slm_model_path)
    
    # 加载 SLM 和 LLM 模型
    # Load models
    slm_model = AutoModelForCausalLM.from_pretrained(
        str(slm_model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        str(llm_model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    
    # 加载投影器（与 load_rosetta_model 中步骤4的逻辑相同）
    # Load projectors
    num_projectors = len([f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)])
    projector_list = []
    for t in range(num_projectors):
        json_cfg = os.path.join(checkpoint_dir, f"projector_{t}.json")
        proj = load_projector(json_cfg)
        proj = proj.to(device)
        pt_path = os.path.join(checkpoint_dir, f"projector_{t}.pt")
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=device)
            proj.load_state_dict(state_dict, strict=False)
        projector_list.append(proj)
    
    # 初始化 Oracle Rosetta 模型（model_list 结构：[slm_model, llm_model]）
    # Initialize Rosetta model
    rosetta_model = OracleRosettaModel(
        model_list=[slm_model, llm_model],
        base_model_idx=0,
        projector_list=projector_list,
    ).to(device).eval()

    # 加载投影器映射配置（Oracle 模型有专用的加载方法）
    # Load projector mapping configs
    proj_cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
    rosetta_model.load_projector_config(proj_cfg_path)

    return rosetta_model, slm_tokenizer


def get_option_token_ids(tokenizer, num_options: int = 4) -> List[int]:
    """
    Get token IDs for options A, B, C, D (or more up to J).
    
    获取选项字母（A/B/C/D/...）对应的 token ID。
    在 logits 方法评测中，需要直接比较模型对各选项 token 的输出概率，
    因此需要先知道每个选项字母在 tokenizer 词汇表中的 token ID。
    
    注意：使用 " " + letter（前置空格）来编码，因为大多数 tokenizer 对
    " A" 和 "A" 会产生不同的 token ID。前置空格是更常见的选项格式。
    
    Args:
        tokenizer: HuggingFace tokenizer 对象
        num_options: 选项数量（默认 4 对应 A-D，最大 10 对应 A-J）
        
    Returns:
        各选项对应的 token ID 列表，如 [319, 350, 315, 360]
    """
    # 最多支持 10 个选项（A-J）
    # Limit to maximum of 10 options (A-J)
    num_options = min(num_options, 10)
    option_ids = []
    for i in range(num_options):
        letter = chr(65 + i)  # A=65, B=66, etc. / ASCII 码转字母
        # 用 " " + letter 编码，取第一个 token 的 ID
        # 某些 tokenizer 可能将 "A" 拆分为多个 token，这里只取第一个
        ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        option_ids.append(ids[0] if ids else tokenizer.eos_token_id)
    return option_ids

"""
Deprecated / 已弃用的函数
以下两个函数（generate_answer_with_logits 和 generate_answer_with_generate）
标记为 Deprecated，但仍在部分评测脚本中使用。
"""

@torch.no_grad()  # 禁用梯度计算，推理阶段不需要反向传播
def generate_answer_with_logits(model, tokenizer, prompt: str, option_ids: List[int], 
                               device: torch.device, model_type: str = "hf") -> Tuple[str, np.ndarray]:
    """
    Generate answer using logits method.
    
    使用 logits 方法生成答案。这是一种基于概率分布的评测方式：
    1. 将 prompt 输入模型，获取最后一个 token 位置处的 logits（未归一化的分数）
    2. 提取 A/B/C/D 四个选项 token 对应的 logit 值
    3. 对这四个 logit 值做 softmax 归一化，得到概率分布
    4. 选择概率最高的选项作为预测答案
    
    这种方法的优势：
    - 速度快：只需一次前向传播，不需要自回归生成
    - 确定性：贪心选择最高概率，无随机性
    - 适合非 CoT 场景（题目简单、不需要逐步推理时）
    
    Args:
        model: 模型对象（可以是 HuggingFace 模型或 RosettaModel）
        tokenizer: tokenizer 对象
        prompt: 输入提示词（已构建好的评测 prompt）
        option_ids: 各选项（A/B/C/D）对应的 token ID 列表
        device: 运行设备
        model_type: 模型类型，影响 chat template 和 forward 调用方式
            - "hf": 标准 HuggingFace 模型
            - "rosetta": RosettaModel（带 KV-Cache 投影的 C2C 模型）
            - "qwen": Qwen 系列模型（需要禁用 thinking 模式）
        
    Returns:
        (predicted_answer, probabilities) 元组
        - predicted_answer: 预测答案字母（'A'/'B'/'C'/'D'）
        - probabilities: 各选项的概率分布 numpy 数组，如 [0.1, 0.7, 0.15, 0.05]
    """
    # 构建单轮对话消息
    messages = [{
        "role": "user",
        "content": prompt
    }]
    
    # 尝试应用 chat template 将消息格式化为模型输入
    # Try to apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False if model_type == "qwen" else None  # Qwen 模型需禁用思考模式
        )
    except Exception as e:
        print(f"Failed to apply chat template for {model_type} model: {e}")
        # 如果 chat template 应用失败，使用 fallback 格式
        text = f"### Human: {prompt}\n### Assistant:"
    
    # 在文本末尾追加 "The correct answer is"，引导模型输出答案
    text += "The correct answer is"
    input_ids = tokenizer(text, return_tensors="pt").to(device)['input_ids']
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    # 计算位置编码：position_ids 为 attention_mask 的累积和减 1
    position_ids = attention_mask.long().cumsum(-1) - 1
    
    if model_type == "rosetta":
        # RosettaModel 需要特殊的 KV-Cache 索引参数
        # instruction_index: 标记哪些 token 属于指令部分（1=指令，0=非指令）
        # response_index: 标记响应部分的起始位置（-1 表示无响应）
        instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(input_ids.shape[1]-1, 1).unsqueeze(0).to(device)
        response_index = torch.tensor([[-1, 0]], dtype=torch.long).unsqueeze(0)
        outputs = model.forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            kv_cache_index=[instruction_index, response_index]
        )
    else:
        # 标准 HuggingFace 模型前向传播
        outputs = model(input_ids)
    
    # 提取最后一个 token 位置的 logits（模型对下一个 token 的预测分布）
    logits = outputs.logits[0, -1]
    # 取出 A/B/C/D 四个选项对应的 logit 值
    option_logits = torch.tensor([
        logits[option_ids[0]].item(),
        logits[option_ids[1]].item(),
        logits[option_ids[2]].item(),
        logits[option_ids[3]].item()
    ])
    
    # 对选项 logits 做 softmax 归一化，得到各选项的概率分布
    probs = torch.nn.functional.softmax(option_logits, dim=0).numpy()
    # 选择概率最高的选项（argmax），转换为字母（0->A, 1->B, 2->C, 3->D）
    pred = chr(65 + np.argmax(probs))
    return pred, probs


@torch.no_grad()  # 禁用梯度计算
def generate_answer_with_generate(model, tokenizer, prompt: str, device: torch.device,
                                 model_type: str = "hf") -> Tuple[str, np.ndarray, int, int, str]:
    """
    Generate answer using text generation method.
    
    使用文本生成方法生成答案。这是一种基于自回归生成的评测方式：
    1. 将 prompt 输入模型
    2. 模型自回归生成文本（可能包含推理过程）
    3. 从生成的文本中用正则表达式提取答案选项
    
    与 logits 方法的区别：
    - logits 方法：直接比较选项概率，快速但不支持 CoT
    - generate 方法：生成完整文本，支持 CoT 推理但更慢
    - generate 方法更适合需要模型"思考"的困难题目
    
    Args:
        model: 模型对象（HuggingFace 模型或 RosettaModel）
        tokenizer: tokenizer 对象
        prompt: 输入提示词
        device: 运行设备
        model_type: 模型类型（"hf" 或 "rosetta"）
        
    Returns:
        (predicted_answer, probabilities, input_length, generation_length, content) 元组
        - predicted_answer: 从生成文本中提取的答案（可能为 None，若提取失败）
        - probabilities: 均匀分布 [0.25, 0.25, 0.25, 0.25]（generate 方法无法提供概率）
        - input_length: 输入序列的 token 数量
        - generation_length: 生成序列的 token 数量
        - content: 模型生成的原始文本内容
    """
    # 构建单轮对话消息
    messages = [{
        "role": "user",
        "content": prompt
    }]
    
    # 应用 chat template 格式化输入
    # Apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # 禁用思考模式（适用于某些支持 thinking 的模型）
        )
    except Exception as e:
        print(f"Failed to apply chat template: {e}")
        text = f"### Human: {prompt}\n### Assistant:"

    # 准备模型输入（tokenize 并移到目标设备）
    # Prepare model input
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # 生成参数配置
    # Generation parameters
    sampling_params = {
        'do_sample': True,          # 启用采样（相对于贪心解码，增加输出多样性）
        'temperature': 0.7,         # 采样温度：0.7 是常用平衡值（低=更确定，高=更随机）
        'top_p': 0.8,               # nucleus sampling：只从累积概率达 0.8 的 token 中采样
        'top_k': 20,                # top-k sampling：只从概率最高的 20 个 token 中采样
        'min_p': 0.0,               # 最小概率阈值（0.0 表示不限制）
        'repetition_penalty': 1.2,  # 重复惩罚：>1.0 抑制模型重复生成相同内容
        'max_new_tokens': 1024      # 最大生成 token 数
    }
    
    # 自回归生成文本
    # Generate text
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        **sampling_params
    )
    
    # 解析生成输出：提取模型新生成的 token（去掉输入部分）
    # Parse output
    if isinstance(model, RosettaModel):
        # RosettaModel 的 generate 方法直接返回生成的 token ID
        generated_ids = outputs[0]
    else:
        # 标准 HuggingFace 模型的输出包含输入+生成，需要切片去掉输入部分
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    # 将 token ID 解码为文本，去除特殊 token 和首尾换行
    content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip("\n")
    
    # 从生成的文本中提取答案选项（使用多级正则匹配策略）
    # Extract answer
    pred = extract_answer_from_content(content)
    
    # generate 方法无法提供各选项的概率分布，返回均匀分布作为占位
    # Return uniform distribution for generate method
    probs = np.array([0.25, 0.25, 0.25, 0.25])

    # 记录输入和生成的序列长度（用于统计分析）
    input_length = inputs.input_ids.shape[1]
    gen_length = generated_ids.shape[0]

    return pred, probs, input_length, gen_length, content

