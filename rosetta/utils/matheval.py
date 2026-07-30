"""
数学评测工具模块 (Math Evaluation Utility Module)

本模块提供了一套完整的数学题目自动评分系统，用于评测 LLM（大语言模型）在各类数学基准测试上的表现。
它是 C2C (Cache-to-Tache) 框架中 rosetta 包的评测工具子模块。

核心类:
    - MathEvaluator: 评测器基类，定义了规则评判和 LLM 评判的通用接口
    - AIMEEvaluator: AIME（美国数学邀请赛）评测器，处理竞赛级数学题
    - GSM8KEvaluator: GSM8K（小学数学应用题）评测器，支持单位/百分比等特殊等价判断
    - MATH500Evaluator: MATH500 数据集评测器，处理多种数学领域题目
    - AMCEvaluator: AMC（美国数学竞赛）评测器
    - GPQAEvaluator: GPQA（研究生级别问答）评测器，使用字符串匹配而非数学表达式

核心函数:
    - set_client(): 配置 OpenAI API 客户端，用于 LLM 评判

评测流程（两层评判机制）:
    1. 规则评判 (rule_judge): 使用 math_verify 库自动提取答案并与标准答案进行数学等价性验证
    2. LLM 评判 (llm_judge): 当规则评判失败时，调用 LLM（如 GPT-4）作为裁判判断答案等价性

注册表机制:
    - evaluator_map 字典将数据集名称映射到对应的评测器实例，实现按数据集自动选择评测策略

依赖关系:
    - math_verify: 用于数学表达式解析和验证
    - latex2sympy2_extended: LaTeX 表达式归一化
    - openai: 调用 LLM API 进行裁判评判
    - transformers: 分词器（用于 token 计数，当前未直接使用）
"""

# ===== 标准库导入 =====
import re                    # 正则表达式：用于从模型输出中提取 </think> 标签后的答案内容
import json                  # JSON 处理：用于 API 请求/响应的序列化（当前未直接使用，保留备用）
import argparse              # 命令行参数解析：用于评测脚本的参数传递（如模型名称、温度等）
import jsonlines             # JSON Lines 格式读写：用于批量评测数据的输入/输出
from collections import defaultdict  # 默认字典：用于评测结果的分组统计（如按难度/类别统计正确率）

# ===== 第三方库导入 =====
from openai import OpenAI    # OpenAI Python SDK：用于创建 LLM 裁判客户端，调用 GPT-4 等模型进行答案评判
from typing import Dict, Any # 类型注解：为函数签名提供类型提示，增强代码可读性

# math_verify 库：核心数学验证工具，提供三个关键组件：
#   - parse(): 将文本解析为数学表达式，支持多种提取配置
#   - verify(): 验证两个数学表达式是否语义等价（如 "1/2" == "0.5"）
#   - LatexExtractionConfig: LaTeX 格式提取配置，用于从 \boxed{} 等 LaTeX 结构中提取答案
#   - ExprExtractionConfig: 纯数学表达式提取配置，用于提取数值/代数表达式
#   - StringExtractionConfig: 字符串提取配置，用于非数学类答案（如 GPQA 的选项文本）
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig

# latex2sympy2_extended: LaTeX 到 SymPy 的扩展转换库
# NormalizationConfig 用于配置 LaTeX 表达式的归一化行为（如是否处理 malformed 运算符、是否支持单位等）
from latex2sympy2_extended import NormalizationConfig

# transformers: Hugging Face 模型库
# AutoTokenizer 用于加载分词器，在注释掉的代码中用于统计 token 数量（推理部分 vs 答案部分）
from transformers import AutoTokenizer

import requests              # HTTP 请求库：用于直接调用 REST API（备选方案，当前主要通过 OpenAI SDK）
import os                    # 操作系统接口：用于从环境变量读取 API 密钥 (OPENAI_API_KEY)

# =============================================================================
# 评测器类层次结构 (Class Hierarchy)
# =============================================================================
#
# MathEvaluator (基类)
# ├── AIMEEvaluator      -- AIME 竞赛数学（答案: 000-999 整数，使用 ExprExtractionConfig 解析标准答案）
# ├── GSM8KEvaluator     -- 小学数学应用题（答案: 数值，LLM 评判支持单位/百分比等价）
# ├── MATH500Evaluator   -- MATH 数据集（答案: 分数/表达式，使用 LatexExtractionConfig 解析标准答案）
# ├── AMCEvaluator       -- AMC 竞赛数学（评测逻辑与 MATH500 相同）
# └── GPQAEvaluator      -- 研究生级问答（非数学，使用 StringExtractionConfig 字符串匹配）
#
# =============================================================================
# 两层评判机制 (Two-Tier Evaluation)
# =============================================================================
#
# 对每个评测样本，评测流程如下：
#
#   模型输出 (solution_str)
#         │
#         ▼
#   [1] rule_judge()  ← 基于 math_verify 库的自动数学验证
#         │                 提取答案 → parse() → verify(gold, answer)
#         │
#    成功? ──── Yes ──→ 返回 (True/False, extracted_answer)
#         │
#         No
#         │
#         ▼
#   [2] llm_judge()   ← 调用 LLM (如 GPT-4) 作为裁判
#         │                 构造提示词 → 调用 API → 检查回复是否含 "YES"
#         │
#         ▼
#   返回 True/False
#
# 设计思想：规则评判速度快、确定性强，但无法处理所有表达形式；
# LLM 评判更灵活（能理解语义等价），但成本高、有不确定性。
# 两者结合实现了高覆盖率和高准确率的评测。
# =============================================================================

class MathEvaluator:
    """
    数学评测器基类 (Base Math Evaluator)

    定义了评测器的通用接口，包括：
    - 规则评判 (rule_judge): 基于数学表达式解析的自动评分
    - LLM 评判 (llm_judge): 调用大语言模型作为裁判进行评分
    - 答案提取 (extract_after_think): 从模型输出中提取 </think> 标签后的最终答案
    - 提示词生成 (get_llm_judge_prompt): 生成供 LLM 裁判使用的评判提示词

    所有具体的评测器（如 AIMEEvaluator、GSM8KEvaluator 等）都应继承此类，
    并根据各自数据集的特点覆写 rule_judge 和 get_llm_judge_prompt 方法。
    """
    
    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        """
        规则评判：使用数学表达式解析库自动比较模型答案与标准答案。

        子类必须实现此方法。不同数据集有不同的答案提取和比较策略。

        Args:
            solution_str: 模型生成的完整解答文本
            ground_truth: 标准答案（ground truth）
            finish_generation: 模型是否已完成生成（True 表示已完成，False 表示可能被截断）

        Returns:
            bool: 答案是否正确

        Raises:
            NotImplementedError: 子类未实现此方法时抛出
        """
        raise NotImplementedError

    def extract_after_think(self, text: str, truncate_length: int = 1000, finish_generation: bool = True) -> str:
        """
        从模型输出中提取 </think> 标签之后的内容作为最终答案部分。

        在 C2C 框架中，LLM 通常使用 CoT（Chain-of-Thought）格式：
        先输出 <think>...</think> 推理过程，然后在 </think> 之后输出最终答案。
        本方法提取最终答案部分，减少 LLM 裁判需要处理的文本量。

        如果模型尚未完成生成（finish_generation=False），则回退到取文本末尾的
        truncate_length 个字符，因为此时 </think> 标签可能尚未出现。

        Args:
            text: 模型生成的完整文本
            truncate_length: 当生成未完成时，截取文本末尾的字符数（默认 1000）
            finish_generation: 模型是否已完成生成

        Returns:
            str: 提取后的答案文本
        """
        # 使用正则表达式匹配 </think> 标签后的所有内容
        # re.DOTALL 使 . 可以匹配换行符，确保跨行内容也能被捕获
        pattern = r"</think>(.*)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if (match and finish_generation) else text[-truncate_length:]
    
    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extracted_answer: str = "", finish_generation: bool = True) -> str:
        """
        生成供 LLM 裁判使用的评判提示词。

        子类必须实现此方法。不同数据集的提示词可能有不同的评判标准和示例。

        Args:
            solution_str: 模型生成的完整解答文本
            ground_truth: 标准答案
            extracted_answer: 通过规则方法提取的答案（可选，部分评判场景使用）
            finish_generation: 模型是否已完成生成

        Returns:
            str: 供 LLM 裁判使用的提示词文本

        Raises:
            NotImplementedError: 子类未实现此方法时抛出
        """
        raise NotImplementedError

    def get_llm_judge_prompt_not_finished(self, solution_str: str, ground_truth: str, extracted_answer: str = "", finish_generation: bool = True) -> str:
        """
        生成模型尚未完成生成时的 LLM 裁判提示词。

        当模型生成被截断（例如达到 token 上限）时，使用此方法生成特殊的评判提示词。
        评判逻辑不同于完成生成时的评判：这里关注的是模型是否已经在推理过程中
        正确推导出了答案，即使后续内容包含不必要的验证、过度思考或重复推理。

        这是一个通用的实现，子类通常不需要覆写此方法。

        Args:
            solution_str: 模型生成的（可能被截断的）解答文本
            ground_truth: 标准答案
            extracted_answer: 通过规则方法提取的答案（当前未使用）
            finish_generation: 模型是否已完成生成（此方法中始终为 False）

        Returns:
            str: 供 LLM 裁判使用的提示词文本，要求 LLM 返回 "YES" 或 "NO"
        """
        # 构造"未完成生成"场景的评判提示词
        # 核心逻辑：不要求模型的最终答案正确，而是检查模型是否已经在推理过程中
        # 正确推导出了答案。这适用于以下场景：
        # - 模型达到 token 上限被截断，但答案已在推理中得出
        # - 模型在得出正确答案后继续进行不必要的验证/重复推理
        # 评判标准：如果正确答案在推理过程中已被推导出 → YES，否则 → NO
        return f"""Please determine whether the final answer in the model-generated response was already correctly derived early in the reasoning process, and that the subsequent content consists mainly of unnecessary verification, overthinking, or repetitive reasoning. If correct is derived early, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
Reference answer: {ground_truth}
Model-generated response: {solution_str}
""".strip()

    def llm_judge(self, solution_str: str, ground_truth: str, extracted_answer: str = "", finish_generation: bool = True) -> bool:
        """
        LLM 评判：调用大语言模型（如 GPT-4）作为裁判来判断模型答案的正确性。

        评判流程：
        1. 根据生成是否完成，选择对应的提示词生成方法
        2. 构造 user message 发送给 LLM 裁判
        3. 解析 LLM 裁判的回复，如果包含 "YES" 则判定为正确

        此方法是规则评判 (rule_judge) 的补充/后备方案。当规则评判无法
        提取有效答案或无法判断等价性时，可回退到此方法。

        Args:
            solution_str: 模型生成的完整解答文本
            ground_truth: 标准答案
            extracted_answer: 通过规则方法提取的答案（可选）
            finish_generation: 模型是否已完成生成

        Returns:
            bool: 如果 LLM 裁判回复中包含 "YES" 返回 True，否则返回 False

        Note:
            依赖模块级全局变量 OPENAI_CLIENT 和 MODEL_NAME，
            需要先调用 set_client() 进行初始化。
        """
        # 引用全局变量：OpenAI 客户端和模型名称
        # 这些变量通过 set_client() 函数设置
        global OPENAI_CLIENT, MODEL_NAME

        def get_inputs(scene_description):
            """
            构造发送给 LLM 裁判的消息列表。

            遵循 OpenAI Chat Completion API 的消息格式规范：
            每条消息包含 role（角色）和 content（内容）字段。
            这里仅构造单条 user 消息，不包含 system prompt，
            因为评判提示词本身已经包含了完整的指令和上下文。

            Args:
                scene_description: 评判提示词文本（由 get_llm_judge_prompt 等方法生成）

            Returns:
                list: 包含单条 user 消息的列表，符合 OpenAI Chat API 格式
            """
            body = [
                {"role": "user", "content": scene_description},
            ]
            return body

        def run_api(inputs):
            """
            调用 OpenAI Chat API 获取 LLM 裁判的回复。

            使用同步调用方式（非异步），因为评测通常是逐样本串行处理的。
            温度等采样参数使用 API 默认值（temperature=1.0），
            确保评判结果尽可能确定性。

            Args:
                inputs: 消息列表（由 get_inputs 构造）

            Returns:
                str: LLM 裁判回复的文本内容（去除首尾空白）
            """
            # 调用 OpenAI Chat Completion API
            # OPENAI_CLIENT 和 MODEL_NAME 是通过 set_client() 设置的全局变量
            completion = OPENAI_CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=inputs
            )
            return completion.choices[0].message.content.strip()

        # 根据生成是否完成，选择不同的提示词模板
        # - finish_generation=True: 使用标准的等价性判断提示词
        # - finish_generation=False: 使用"是否已提前推导出正确答案"的判断提示词
        if finish_generation:
            scene_description = self.get_llm_judge_prompt(solution_str, ground_truth, extracted_answer, finish_generation)
        else:
            scene_description = self.get_llm_judge_prompt_not_finished(solution_str, ground_truth, extracted_answer, finish_generation)
        inputs = get_inputs(scene_description)
        response = run_api(inputs)

        # 简单的关键词匹配：只要回复中包含 "YES" 就判定为正确
        # LLM 裁判被要求只返回 "YES" 或 "NO"，但实际输出可能包含额外内容
        return "YES" in response


class AIMEEvaluator(MathEvaluator):
    """
    AIME（美国数学邀请赛）评测器 (AIME Evaluator)

    AIME 是高中水平的数学竞赛，题目答案为 000-999 的整数。
    本评测器使用数学表达式解析库进行规则评判：
    - 标准答案使用 ExprExtractionConfig 解析为数学表达式
    - 模型答案优先匹配 \boxed{} 中的 LaTeX 表达式，其次匹配普通数学表达式
    - 使用 math_verify 库的 verify() 函数进行数学等价性验证

    同时也提供 LLM 裁判提示词，用于规则评判失败时的后备方案。
    """

    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        """
        AIME 规则评判：提取模型答案并与标准答案进行数学等价性验证。

        算法流程：
        1. 使用 ExprExtractionConfig 将标准答案解析为数学表达式 (gold)
        2. 从模型输出中提取答案 (answer)，提取策略按优先级：
           a. 优先匹配 \boxed{} 中的 LaTeX 内容（含归一化处理）
           b. 其次匹配普通数学表达式
        3. 使用 verify() 进行数学等价性验证

        Args:
            solution_str: 模型生成的完整解答文本
            ground_truth: 标准答案（通常为 000-999 的整数）
            finish_generation: 模型是否已完成生成

        Returns:
            tuple: (bool, str) - (是否正确, 提取到的答案文本)
        """
        # if not ground_truth.startswith("$"):
        #     ground_truth = f"${ground_truth}$"

        # 解析标准答案为数学表达式
        # 使用 ExprExtractionConfig 提取纯数值/表达式
        gold = parse(
            ground_truth,
            extraction_config=[ExprExtractionConfig()],
        )

        # 从模型输出中提取答案
        # 使用两种提取配置，按优先级尝试匹配：
        # 1. LatexExtractionConfig: 优先从 \boxed{} 中提取 LaTeX 表达式
        #    - nits=False: 不进行细节检查（如忽略多余空格）
        #    - malformed_operators=False: 不严格检查运算符格式
        #    - basic_latex=True: 支持基础 LaTeX 语法
        #    - boxed="all": 匹配所有 \boxed{} 内容
        #    - units=True: 支持带单位的答案
        #    - boxed_match_priority=0: \boxed{} 匹配具有最高优先级
        #    - try_extract_without_anchor=False: 必须找到锚点（如 \boxed）才提取
        # 2. ExprExtractionConfig: 后备方案，直接提取数学表达式
        # extraction_mode="first_match": 使用第一个成功匹配的配置
        answer = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )

        # 如果未提取到任何答案，返回失败
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            # 使用 math_verify 的 verify() 进行数学等价性验证
            # verify() 会检查两个数学表达式在语义上是否等价
            # 例如：verify(parse("42"), parse("42")) -> True
            return verify(gold, answer), str(answer)

    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        """
        生成 AIME 评测的 LLM 裁判提示词。

        提示词要求 LLM 裁判判断模型答案是否与标准答案数学等价，
        答案可能被 \boxed{} 包裹或出现在 "Answer:" 之后。

        Args:
            solution_str: 模型生成的完整解答文本
            ground_truth: 标准答案
            extract_answer: 已提取的答案（当前未使用）
            finish_generation: 模型是否已完成生成

        Returns:
            str: LLM 裁判提示词
        """
        # 先提取 </think> 之后的内容，减少 LLM 裁判需要处理的文本量
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a math question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}""".strip()


class GSM8KEvaluator(MathEvaluator):
    """
    GSM8K（小学数学应用题）评测器 (GSM8K Evaluator)

    GSM8K (Grade School Math 8K) 是一个包含约 8500 道小学数学应用题的数据集。
    答案通常为单个数值，但模型生成的答案可能包含单位、百分比符号或时间格式。

    规则评判 (rule_judge) 与 AIME 评测器逻辑相同：
    - 使用 LaTeX + 表达式双重提取配置
    - 使用 math_verify 进行数学等价性验证

    LLM 评判的特殊设计：
    - 支持单位等价判断：1% 等价于 1，1 kg 等价于 1
    - 支持时间格式等价判断：1:00 pm 等价于 1
    - 支持多数值等价判断：当标准答案为单数值，模型答案为多数值时，
      判断数值组合（如求和）是否等价于标准答案
      例如：模型答案 "5, 5" 等价于标准答案 "10"
    """

    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        """
        GSM8K 规则评判：提取模型答案并与标准答案进行数学等价性验证。

        与 AIME 评测器使用相同的提取和验证策略。

        Args:
            solution_str: 模型生成的完整解答文本
            ground_truth: 标准答案（通常为单个数值）
            finish_generation: 模型是否已完成生成

        Returns:
            tuple: (bool, str) - (是否正确, 提取到的答案文本)
        """
        # if not ground_truth.startswith("$"):
        #     ground_truth = f"${ground_truth}$"

        # 解析标准答案为数学表达式
        gold = parse(
            ground_truth,
            extraction_config=[ExprExtractionConfig()],
        )

        # 从模型输出中提取答案（与 AIME 相同的提取配置）
        # 优先匹配 \boxed{} 内容，其次匹配普通数学表达式
        answer = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,  # GSM8K 答案经常带有单位（如 kg, %）
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            return verify(gold, answer), str(answer)

    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        """
        生成 GSM8K 评测的 LLM 裁判提示词。

        GSM8K 的 LLM 评判提示词包含两类特殊等价规则的示例：
        1. 单位/百分比/时间格式等价：
           - "1%" 等价于 "1"
           - "1 kg" 等价于 "1"
           - "1:00 pm" 等价于 "1"
        2. 多数值等价：
           - 当标准答案为单数值时，模型给出的多数值组合可以等价
           - 例如 "5, 5" 等价于 "10"（求和），"4, 6" 等价于 "10"

        这些规则反映了 GSM8K 数据集的特点：答案可能以不同形式表达相同含义。

        Args:
            solution_str: 模型生成的完整解答文本
            ground_truth: 标准答案
            extract_answer: 已提取的答案（当前未使用）
            finish_generation: 模型是否已完成生成

        Returns:
            str: LLM 裁判提示词
        """
        # 先提取 </think> 之后的内容
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response with rule-based extracted answer is equivalent to the reference answer from a math question. The final answer may either be enclosed in the \\boxed{{}} or appear after the "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.

1. The reference answer does not include percentage signs, units or time formats (e.g., am, pm), but the Model-generated answer may include them.
For example, 1 is equivalent to 1 %, 1 kg, 1 am, 1 pm, 1:00 am, 1:00 pm, etc.
Model-generated answer: 1%
Reference answer: 1
Your output: YES

Model-generated answer: 1 kg
Reference answer: 1
Your output: YES

Model-generated answer: 1:00 pm
Reference answer: 1
Your output: YES

2. The reference answer only includes one single number, but the Model-generated answer may include multiple numbers.
For example, 10 is equivalent to \\boxed{{(4, 6)}}, etc.
Model-generated answer: 5, 5
Reference answer: 10
Your output: YES

Model-generated answer: 4, 6
Reference answer: 10
Your output: YES

Model-generated answer: 86, 42
Reference answer: 128
Your output: YES

Now let's try a real example.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}
""".strip()


class MATH500Evaluator(MathEvaluator):
    """
    MATH500 评测器 (MATH500 Evaluator)

    MATH500 是从 MATH 数据集中抽取的 500 道题目，涵盖代数、计数与概率、
    几何、中级代数、数论、预微积分和初级代数等多个数学领域。

    与 AIME 评测器的关键区别：
    - 标准答案用 $ 符号包裹后，使用 LatexExtractionConfig 解析（而非 ExprExtractionConfig）
      这是因为 MATH500 的答案形式更多样（分数、表达式等），需要 LaTeX 解析支持
    - 如果标准答案没有 $ 包裹，会自动添加

    规则评判和 LLM 评判的提示词结构与 AIME 类似。
    """

    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        """
        MATH500 规则评判：提取模型答案并与标准答案进行数学等价性验证。

        与 AIME 评测器的区别：
        - 标准答案用 $ 包裹后使用 LatexExtractionConfig 解析
        - 模型答案使用 LaTeX + 表达式双重提取配置（与 AIME 相同）

        Args:
            solution_str: 模型生成的完整解答文本
            ground_truth: 标准答案（可能是分数、表达式等 LaTeX 格式）
            finish_generation: 模型是否已完成生成

        Returns:
            tuple: (bool, str) - (是否正确, 提取到的答案文本)
        """
        # MATH500 的标准答案需要包裹在 $ 中以启用 LaTeX 解析
        if not ground_truth.startswith("$"):
            ground_truth = f"${ground_truth}$"

        # 解析标准答案：使用 LatexExtractionConfig（与 AIME 不同）
        # 这样能正确解析分数、根号等复杂 LaTeX 数学表达式
        gold = parse(
            ground_truth,
            extraction_config=[LatexExtractionConfig()],
        )

        # 从模型输出中提取答案（与 AIME 相同的提取配置）
        answer = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            return verify(gold, answer), str(answer)

    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        """
        生成 MATH500 评测的 LLM 裁判提示词。

        提示词结构与 AIME 相同，判断模型答案是否与标准答案数学等价。

        Args:
            solution_str: 模型生成的完整解答文本
            ground_truth: 标准答案
            extract_answer: 已提取的答案（当前未使用）
            finish_generation: 模型是否已完成生成

        Returns:
            str: LLM 裁判提示词
        """
        # 先提取 </think> 之后的内容
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a math question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}""".strip()
    
class AMCEvaluator(MathEvaluator):
    """
    AMC（美国数学竞赛）评测器 (AMC Evaluator)

    AMC 是美国数学竞赛系列考试（AMC 8/10/12），题目为选择题，
    但答案通常以数值形式给出。

    评测逻辑与 MATH500Evaluator 完全相同：
    - 标准答案用 $ 包裹后使用 LatexExtractionConfig 解析
    - 模型答案使用 LaTeX + 表达式双重提取配置
    - 使用 verify() 进行数学等价性验证

    LLM 评判提示词也与 MATH500 相同。
    """

    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        """
        AMC 规则评判：与 MATH500 相同的提取和验证策略。

        Args:
            solution_str: 模型生成的完整解答文本
            ground_truth: 标准答案
            finish_generation: 模型是否已完成生成

        Returns:
            tuple: (bool, str) - (是否正确, 提取到的答案文本)
        """
        # AMC 的标准答案也需要 $ 包裹以启用 LaTeX 解析（与 MATH500 相同）
        if not ground_truth.startswith("$"):
            ground_truth = f"${ground_truth}$"

        # 解析标准答案为 LaTeX 数学表达式
        gold = parse(
            ground_truth,
            extraction_config=[LatexExtractionConfig()],
        )

        # 从模型输出中提取答案
        answer = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            # 使用 math_verify 进行数学等价性验证
            return verify(gold, answer), str(answer)

    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        """
        生成 AMC 评测的 LLM 裁判提示词（与 MATH500 相同）。

        Args:
            solution_str: 模型生成的完整解答文本
            ground_truth: 标准答案
            extract_answer: 已提取的答案（当前未使用）
            finish_generation: 模型是否已完成生成

        Returns:
            str: LLM 裁判提示词
        """
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a math question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}""".strip()


class GPQAEvaluator(MathEvaluator):
    """
    GPQA（研究生级别问答）评测器 (GPQA Evaluator)

    GPQA (Graduate-Level Google-Proof Q&A Benchmark) 是一个研究生级别的
    高难度问答题基准测试，涵盖物理、化学、生物等领域。

    与其他数学评测器的关键区别：
    - 使用 StringExtractionConfig 而非数学表达式提取
    - 这是因为 GPQA 的答案通常是文本字符串（如选项字母、简短文本答案），
      而非纯数学表达式
    - 标准答案不需要 $ 包裹

    规则评判使用字符串匹配（通过 verify() 的字符串比较模式），
    LLM 评判则判断是否为多选题的等价答案。
    """

    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        """
        GPQA 规则评判：使用字符串提取和匹配来判断答案正确性。

        与数学评测器不同，GPQA 使用 StringExtractionConfig 提取文本答案，
        而非数学表达式。这是因为 GPQA 的答案形式为文本（如选项 A/B/C/D）。

        Args:
            solution_str: 模型生成的完整解答文本
            ground_truth: 标准答案（通常为选项字母或简短文本）
            finish_generation: 模型是否已完成生成

        Returns:
            tuple: (bool, str) - (是否正确, 提取到的答案文本)
        """
        # if not ground_truth.startswith("$"):
        #     ground_truth = f"${ground_truth}$"

        # 使用 StringExtractionConfig 解析标准答案（字符串模式）
        gold = parse(
            ground_truth,
            extraction_config=[StringExtractionConfig()],
        )

        # 从模型输出中同样使用字符串提取
        answer = parse(
            solution_str,
            extraction_config=[
                StringExtractionConfig(),
            ]
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            # 使用 verify() 进行字符串等价性验证
            return verify(gold, answer), str(answer)
        
    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        """
        生成 GPQA 评测的 LLM 裁判提示词。

        与数学评测器的提示词类似，但针对多选题场景：
        判断模型答案是否与标准答案等价（如选项匹配）。

        Args:
            solution_str: 模型生成的完整解答文本
            ground_truth: 标准答案
            extract_answer: 已提取的答案（当前未使用）
            finish_generation: 模型是否已完成生成

        Returns:
            str: LLM 裁判提示词
        """
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a multiple choice question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}""".strip()


# ===== 以下注释代码为未完成/预留的代码生成评测器 =====
# MBPP (Mostly Basic Python Programming) 和 HumanEval 是代码生成基准测试
# 这些评测器使用代码执行验证而非数学表达式比较，目前尚未实现完整

# class MBPPEvaluator(Evaluator):
#     """MBPP（Python 编程基础题）评测器 - 尚未实现"""
#     def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
#         return True, "No extracted answer"
        
#     def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
#         solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
#         return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a multiple choice question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
# Model-generated answer: {solution_str}
# Reference answer: {ground_truth}""".strip()


# class HUMANEVALEvaluator(Evaluator):
#     """HumanEval（OpenAI 代码生成评测）评测器 - 尚未实现"""
#     def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
#         return True, "No extracted answer"
        
#     def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
#         solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
#         return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a multiple choice question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
# Model-generated answer: {solution_str}
# Reference answer: {ground_truth}""".strip()


# ===== 评测器注册表 (Evaluator Registry) =====
# 将数据集名称映射到对应的评测器实例
# 外部代码通过 evaluator_map[dataset_name] 获取对应评测器，
# 实现按数据集自动选择评测策略
# 支持的评测基准：
#   - aime2024/2025: AIME（美国数学邀请赛）2024/2025 年度试题
#   - gsm8k: GSM8K（小学数学应用题数据集）
#   - math500: MATH500（MATH 数据集的 500 题子集）
#   - gpqa_diamond: GPQA Diamond（研究生级别高难度问答）
#   - amc23: AMC 2023（美国数学竞赛）
evaluator_map = {
    "aime2024": AIMEEvaluator(),
    "aime2025": AIMEEvaluator(),
    "gsm8k": GSM8KEvaluator(),
    "math500": MATH500Evaluator(),
    "gpqa_diamond": GPQAEvaluator(),
    "amc23": AMCEvaluator(),
}

# ===== 全局 API 配置变量 =====
# 这些变量通过 set_client() 函数设置，供 llm_judge() 方法使用
API_BASE = None          # API 基础 URL（如 Azure OpenAI 的 endpoint）
DEPLOYMENT_NAME = None   # Azure 部署名称（仅 Azure 场景使用）
API_VERSION = None       # API 版本号（如 Azure 的 api-version）
CONSTRUCTED_URL = None   # 完整构造后的 API 请求 URL
API_KEY = None           # API 密钥（优先使用参数传入，否则从环境变量读取）
HEADERS = None           # HTTP 请求头（包含 Content-Type 和 api-key）
OPENAI_CLIENT = None     # OpenAI 客户端实例，用于 LLM 评判
MODEL_NAME = None        # LLM 裁判使用的模型名称（默认 gpt-4.1）

def set_client(api_base=None, deployment_name=None, api_version=None, api_key=None, model_name="gpt-4.1-2025-04-14"):
    """
    配置 OpenAI API 客户端，供 LLM 评判使用。

    此函数初始化全局的 API 连接参数和 OpenAI 客户端实例。
    必须在调用 llm_judge() 之前调用此函数。

    支持两种 API 接入方式：
    1. 标准 OpenAI API：仅需提供 api_key
    2. Azure OpenAI API：需提供 api_base、deployment_name、api_version 和 api_key

    Args:
        api_base: API 基础 URL（Azure 场景需要）
        deployment_name: Azure 部署名称
        api_version: API 版本号
        api_key: API 密钥，如果不提供则从环境变量 OPENAI_API_KEY 读取
        model_name: LLM 裁判模型名称，默认为 "gpt-4.1-2025-04-14"

    Side Effects:
        设置全局变量 API_BASE, DEPLOYMENT_NAME, API_VERSION, CONSTRUCTED_URL,
        API_KEY, HEADERS, MODEL_NAME, OPENAI_CLIENT
    """
    global API_BASE, DEPLOYMENT_NAME, API_VERSION, CONSTRUCTED_URL, API_KEY, HEADERS, MODEL_NAME, OPENAI_CLIENT

    API_BASE = api_base
    DEPLOYMENT_NAME = deployment_name
    API_VERSION = api_version
    # 构造 Azure OpenAI 风格的完整请求 URL
    CONSTRUCTED_URL = f"{api_base}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    # API 密钥：优先使用传入参数，否则从环境变量读取
    API_KEY = api_key or os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME = model_name
    HEADERS = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    # 如果有有效的 API 密钥，创建 OpenAI 客户端；否则客户端为 None
    if API_KEY:
        print(f"Using API key: {API_KEY}")
        OPENAI_CLIENT = OpenAI(api_key=API_KEY)
    else:
        OPENAI_CLIENT = None
    


# ===== 以下注释代码为历史遗留的独立函数实现 =====
# 这些功能已被整合到 MathEvaluator 类的方法中，保留作为参考
#
# 设计演进说明：
# 早期版本采用独立函数的方式实现评测逻辑（call_llm_judge, rule_judge, process_example），
# 每个函数处理特定任务。后来重构为面向对象的评测器类体系（MathEvaluator 及其子类），
# 将数据集特定的评判策略封装在各自的类中，通过 evaluator_map 注册表进行调度。
# 这种重构提高了代码的可扩展性——新增数据集只需创建新的评测器类并注册到 evaluator_map 即可。

# def call_llm_judge(message: list, args: argparse.Namespace) -> str:
#     """
#     Call the Qwen API with the given message.
#     调用 Qwen API 获取 LLM 裁判回复（已被 MathEvaluator.llm_judge 方法替代）
    
#     Args:
#         message (list): Message list for the API.
#         args (argparse.Namespace): Parsed arguments.
        
#     Returns:
#         str: The content of the completion response.
        
#     Raises:
#         Exception: When the API call fails.
#     """

#     try:
#         completion = client.chat.completions.create(
#             model=args.model_name,
#             messages=message,
#             temperature=args.temperature,
#             top_p=args.top_p,
#         )
#         return completion.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"API call failed: {str(e)}")
#         raise


# def rule_judge(completin, ground_truth, dataset_name: str) -> Dict[str, Any]:
#     """
#     规则评判的独立函数版本（已被各评测器类的 rule_judge 方法替代）
#     
#     通过 evaluator_map 根据数据集名称选择对应评测器进行评判。
#     如果规则评判失败，应回退到 LLM 评判（call_llm_judge）。
#     """
#     rule_judge_result = None

#     rule_judge_result = evaluator_map[dataset_name].rule_judge(completin, ground_truth)

#     if not rule_judge_result:
#         print(f"No valid answer detected | LLM judge")
#         call_llm_judge
    
        
#     return example



# def process_example(example: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
#     """
#     处理单个评测样本，包含 token 计数和评判逻辑。
#     
#     此函数展示了完整的评测流程：
#     1. 计算推理部分和答案部分的 token 数量
#     2. 先尝试规则评判 (rule_judge)
#     3. 如果规则评判失败，回退到 LLM 评判 (llm_judge)
#     4. 返回带有评判结果的样本字典
#     
#     返回的 example 字典包含以下字段：
#     - before_truncation_token_count: <think> 推理部分的 token 数
#     - after_truncation_token_count: </think> 之后答案部分的 token 数
#     - rule_judge_result: 规则评判结果
#     - llm_judge_response: LLM 裁判的原始回复
#     - llm_judge_result: LLM 评判结果
#     - final_judge_result: 最终评判结果（优先使用规则评判结果）
#     """
#     try:

#         completion_text = example.get('completion', '')
#         # 以 </think> 为分界，分离推理过程和最终答案，分别统计 token 数
#         before_truncation_text = completion_text.split("</think>")[0]
#         before_truncation_token_count = count_tokens(before_truncation_text, args)
#         after_truncation_token_count = example.get('generated_tokens', 0) - before_truncation_token_count

#         example['before_truncation_token_count'] = before_truncation_token_count
#         example['after_truncation_token_count'] = after_truncation_token_count

#         example = post_process(example)
#         if not example.get('rule_judge_result', False):
#             think_truncation = extract_after_think(completion_text) or ''

#             prompt = generate_prompt(example, think_truncation)
#             messages = format_prompt(prompt)
#             llm_judge_response = call_qwen(messages, args)
#             print(llm_judge_response)
#             example['llm_judge_response'] = llm_judge_response
#             if llm_judge_response == "YES":
#                 example['llm_judge_result'] = True
#                 example['final_judge_result'] = True
#             elif llm_judge_response == "NO":
#                 example['llm_judge_result'] = False
#                 example['final_judge_result'] = False
#             else:
#                 example['llm_judge_result'] = None
#                 example['final_judge_result'] = None
                
#         return example
#     except Exception as e:
#         print(f"Failed to process example {example.get('idx', 'unknown')}: {str(e)}")
#         example['final_judge_result'] = None
#         return example



# def main():
#     """
#     主函数：执行完整的评测流程。
#     
#     流程：
#     1. 从 JSON 文件加载评测数据
#     2. 逐个处理样本（计算 token 数 + 规则评判 + LLM 评判）
#     3. 将结果保存到 jsonlines 文件
#     4. 计算并输出评测指标：
#        - Pass@1: 每道题的正确率（对每个题目 idx 分别计算）
#        - Overall pass@1: 所有题目的平均正确率
#        - Average generated_tokens: 平均生成 token 总数
#        - Average reasoning_tokens: 平均推理部分 token 数（
</think>

 之前）
#        - Average answer_tokens: 平均答案部分 token 数（
</think>

 之后）
#     """
#     args = parse_arguments()
#     try:
#         final_results = []
#         with jsonlines.open(args.result_save_name, mode='w') as writer:
#             with open(args.data_load_name, 'r', encoding='utf-8') as f:
#                 data = json.load(f)

#             for idx, example in enumerate(data, start=1):
#                 print(f"Processed example ID: {idx}")
#                 new_example = process_example(example, args)
#                 final_results.append(new_example)

#             writer.write_all(final_results)
#             print(f"Processing complete! Valid results saved to: {args.result_save_name}")

#         # 按题目 ID 分组，用于计算每道题的 pass@1
#         grouped_data = defaultdict(list)
#         for entry in final_results:
#             grouped_data[entry['idx']].append(entry)

#         # 计算每道题的 pass@1（正确次数 / 尝试次数）
#         pass_at_1_per_idx = {}
#         for idx, entries in grouped_data.items():
#             correct_count = sum(entry['final_judge_result'] for entry in entries if entry['final_judge_result'] is True)
#             total_count = len(entries)
#             pass_at_1_per_idx[idx] = correct_count / total_count if total_count else 0

#         # 汇总指标
#         overall_pass_at_1 = sum(pass_at_1_per_idx.values()) / len(pass_at_1_per_idx)
#         average_generated_tokens = sum(entry['generated_tokens'] for entry in final_results) / len(final_results)
#         average_reasoning_tokens = sum(entry['before_truncation_token_count'] for entry in final_results) / len(final_results)
#         average_answer_tokens = sum(entry['after_truncation_token_count'] for entry in final_results) / len(final_results)

#         print(f'\nPass@1 per idx:\n{pass_at_1_per_idx}\n')
#         print(f'Overall pass@1: {overall_pass_at_1:.4f}')
#         print(f'Average generated_tokens: {average_generated_tokens:.2f}')
#         print(f'Average reasoning_tokens: {average_reasoning_tokens:.2f}')
#         print(f'Average answer_tokens: {average_answer_tokens:.2f}')

#     except json.JSONDecodeError as e:
#         print(f"Data loading failed: Invalid JSON format - {str(e)}")
#         raise


# =============================================================================
# 主入口：独立运行时的测试/演示代码 (Test/Demo Entry Point)
# =============================================================================
# 当直接运行此脚本时（python matheval.py），执行以下测试流程：
# 1. 从环境变量获取 OpenAI API 密钥
# 2. 初始化 LLM 裁判客户端（set_client）
# 3. 使用一段 AIME 数学题的模型推理输出作为测试用例
# 4. 调用 llm_judge 测试在 finish_generation=False（模型未完成生成）场景下的评判能力
#
# 测试用例说明：
# - response 变量是一段 AIME 数学竞赛题的模型 CoT 推理过程
# - 题目：求满足 a+b+c=300 且 a²b+a²c+b²a+b²c+c²a+c²b=6,000,000 的非负整数三元组数量
# - 正确答案是 601
# - 使用 evaluator_map["aime2024"] 获取 AIME 评测器
# - llm_judge 的 finish_generation=False 参数测试"未完成生成"的评判逻辑
if __name__ == '__main__':
    # 从环境变量 OPENAI_API_KEY 读取 API 密钥，如果未设置则使用空字符串
    api_key = os.getenv("OPENAI_API_KEY", "")

    # 初始化 OpenAI 客户端，用于 LLM 裁判评判
    # 这里 api_base/deployment_name/api_version 为空字符串，表示使用标准 OpenAI API（非 Azure）
    set_client(api_base="", deployment_name="", api_version="", api_key=api_key)
    response = "Okay, so I need to find the number of triples (a, b, c) of nonnegative integers such that a + b + c = 300 and the expression a\u00b2b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b equals 6,000,000. Hmm, that seems a bit complicated, but maybe I can break it down step by step.\n\nFirst, let me note down the given equations:\n\n1. a + b + c = 300\n2. a\u00b2b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b = 6,000,000\n\nI need to find all nonnegative integer triples (a, b, c) that satisfy both equations. Since the problem involves symmetric expressions in a, b, c, maybe there's a way to exploit symmetry here.\n\nLet me first if I can rewrite the second equation in a more manageable form. The expression a\u00b2b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b looks symmetric. Maybe I can factor it somehow?\n\nLet me see: Each term is of the form (variable squared times another variable). So, for example, a\u00b2b + a\u00b2c = a\u00b2(b + c), similarly for the other terms. Let me try that:\n\na\u00b2(b + c) + b\u00b2(a + c) + c\u00b2(a + b)\n\nYes, that's the same as the original expression. Now, since a + b + c = 300, maybe I can substitute (b + c) with (300 - a), and similarly for the others. Let's try that substitution.\n\nSo, substituting:\n\na\u00b2(300 - a) + b\u00b2(300 - b) + c\u00b2(300 - c)\n\nWait, hold on. Let me check that again. The original expression is't exactly that. Wait, the's see:\n\nWait, the original expression is a\u00b2(b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b. So I group terms as a\u00b2(b + c) + b\u00b2(a + c) + c\u00b2(a + b), then yes, each term is variable squared times the sum of the other two variables. Since the sum of the other two variables is 300 minus the variable. So, for example, (b + c) = 300 - a, so substituting that in, we get:\n\na\u00b2(300 - a) + b\u00b2(300 - b) + c\u00b2(300 - c)\n\nTherefore, the second equation can be rewritten as:\n\na\u00b2(300 - a) + b\u00b2(300 - b) + c\u00b2(300 - c) = 6,000,000\n\nHmm, that might be a useful form. Let me write that down:\n\nEquation 2: a\u00b2(300 - a) + b\u00b2(300 - b) + c\u00b2(300 - c) = 6,000,000\n\nNow, perhaps I can consider expanding expression in terms of the variables. Let me think if there's a way to relate this to the first equation. Since a + b + c = 300, maybe we can express some relationship between these variables here Let't immediately yet.\n\nAlternatively, maybe I can consider expanding the expression or finding another way to express it. Let me see ifAlternatively, perhaps I can consider the expression a\u00b2b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b. Let me see if this can be written in terms of (a + b + c)^3 or something similar.\n\nWait, let me recall that (a + b + c)^3 = a^3 + b^3 + c^3 + 3(a + b)(a + c)(b + c). Hmm, not sure if that helps. Alternatively, maybe expanding (a + b + c)(ab + bc + ca). Let me see:\n\n(a + b + c)(ab + bc + ca) = a^2b + a^2c + ab^2 + b^2c + a c^2 + b c^2 + 3abc\n\nWait, that's exactly our expression in the second equation plus 3abc. So:\n\n(a + b + c)(ab + bc + ca) = (a\u00b2b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b) + 3abc\n\nTherefore, the second equation can be rewritten as:\n\n(a + b + c)(ab + bc + ca) - 3abc = 6,000,000\n\nSince a know that a + b + c = 300, substitute that in:\n\n300(abab + bc + ca) - 3abc = 6,000,000\n\nHmm, that might be a better way to handle it equation. Let me write that down:\n\nEqu00(ab + bc + ca) - 3abc = 6,000,000\n\nDiv can divide both sides by 3 to simplify:\n\n100(ab + bc + ca) - abc = 2,000,000\n\nSo now, the equation becomes:\n\n100(ab + bc + ca) - abc = 2,000,000\n\nHmm, that seems a bit more manageable. Let me write that as:\n\nabc - 100(ab + bc + ca) = -2,000,000\n\nAlternatively, maybe rearranged as:\n\nabc = 100(ab + bc + ca) - 2,000,000\n\nHmm, not sure yet. Let me see if I can find another relation here.\n\n Since we have a + b + c = 300, perhaps we can denote variables in terms of two variables. say, let me set variables as a, b, and then c = 300 - a - b. Then substitute expressions in terms of a and b. But that might get complicated, but perhaps manageable.\n\nAlternatively, maybe there's a substitution or symmetry here. Let me think.\n\n the problem is symmetric in a, b, c, so perhaps the solutions are symmetric in some way. Maybe two variables are equal, or all three are equal?\n\nLet me first check if all three variables are equal. If a = b = c, then since a + b + c = 300, each would be 100. Let me check if that satisfies the second equation.\n\nIf a = b = c = 100, then compute the second expression:\n\na\u00b2b + a\u00b2c + ...\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b\n\n = 6*(100)^2*100 = 6*10003 = 6,000,000. Wait, that's exactly the required given! So (100, 100, 100) is a solution.\n\n So that's one solution. But the problem asks for all triples, so maybe might be more?\n\nHmm, so maybe there are other triples where variables are not all equal but still satisfy the equations. Let me see.\n\nAlternatively, maybe two variables are equal and the third is different. Let me suppose that two variables are equal, say a = b, and c is different. Let,2a + c = 300, so c = 300 - 2a.\n\nThen, let's plug into the second equation.\n\nFirst, let me compute ab + bc + ca. when a = b:\n\nab + bc + ca = a\u00b2 + a c + a c = a\u00b2 + 2a c\n\nSimilarly, abc = a\u00b2 c\n\nSo substitugging into the equation 100(ab + bc + ca) - abc = 100(a\u00b2 + 2a c) - a\u00b2 c = 100a\u00b2 + 200a c - a\u00b2 c\n\nBut since c = 300 - 2a, substitute that in:\n\n100a\u00b2 + 200a(300 - 2a) - a\u00b2(300 - 2a)\n\nLet me compute each term:\n\nFirst term: 100a\u00b2\n\nSecond term: 200a*(300 - 2a) = 200a*300 - 200a*2a = 60,000a - 400a\u00b2\n\nThird term: -a\u00b2*(300 - 2a) = -300a\u00b2 + 2a\u00b3\n\nSo adding all three terms together:\n\n100a\u00b2 + (60,000a - 400a\u00b2) + (-300a\u00b2 + 2a\u00b3)\n\nCombine like terms:\n\nFor00a\u00b2 - 400a\u00b2 - 300a\u00b2 + 2a\u00b3 + 60,000a\n\nThat's:\n\n(100 - 400 - 300)a\u00b2 + 2a\u00b3 + 60,000a\n\n= (-600a\u00b2) + 2a\u00b3 + 60,000a\n\nSo the entire expression is 2a\u00b3 - 600a\u00b2 + 60,000a\n\nSet this equal to 2,000,000 ( (since the equation was 100(ab + bc + ca) - abc = 2,000,000)\n\nWait:\n\n2a\u00b3 - 600a\u00b2 + 60,000a = 2,000,000\n\nDivide both sides by 2:\n\na\u00b3 - 300a\u00b2 + 30,000a = 1,000,000\n\nBring all terms to left-hand:\n\na\u00b3 - 300a\u00b2 + 30,000a - 1,000,000 = 0\n\nHmm, need to solve this cubic equation for integer a. Let me see if I can factor this or find integer roots.\n\nPossible me denote f(a) = a\u00b3 - 300a\u00b2 + 30,000a - 1,000,000\n\nWe can try possible integer roots. Since a must be a nonnegative integer such that 2a \u2264 300 (since c = 300 - 2a must be nonnegative). so a \u2264 150.\n\nPossible rational roots are factors of 1,000,000 divided by factors of 1, so possible integer roots are divisors of 1,000,000. But that's a lot. Maybe try some values.\n\nFirst, let me try a = 100. Since when all variables are 100, that's a solution. so if a = 100, then c = 300 - 200 = 100, so that's the case where all three are equal, so that should be a root.\n\nLet me check f(100):\n\n100\u00b3 - 300*(100)^2 + 30,000*100 - 1,000,000\n\n= 1,000,000 - 300*10,000 + 3,000,000 - 1,000,000\n\n= 1,000,000 - 3,000,000 + 3,000,000 - 1,000,000 = 0. Yes, so a = 100 is a root.\n\nTherefore, (a - 100) is a factor. Let's perform polynomial division or factor it out.\n\nDivide f(a) by (a - 100):\n\nUsing synthetic division:\n\nCoefficients: 1 | -300 | 30,000 | -1,000,000\n\nDivide by (a - 100), so root at 100.\n\nBring down the 1.\n\nMultiply 1 by 100: 100. Add to next coefficient: -300 + 100 = -200\n\nMultiply -200 by 100 = -20,000. Add to next coefficient: 30,000 + (-20,000) = 10,000\n\nMultiply 10,000 by 100 = 1,000,000. Add to last term: -1,000,000 + 1,000,000 = 0. Perfect.\n\nSo the polynomial factors as (a - 100)(a\u00b2 - 200a + 10,000)\n\nNow set the quadratic factor to zero:\n\na\u00b2 - 200a + 10,000 = 0\n\nLet me compute discriminant D = (200)^2 - 4*1*10,000 = 40,000 - 40,000 = 0\n\nSo discriminant is zero, so a = [200 \u00b1 0]/2 = 100. So the quadratic has a double root at a = 100.\n\nTherefore, the cubic factors as (a - 100)^3 = 0. Wait, but that would mean that the original cubic is (a - 100)^3, but let me check:\n\nWait, (a - 100)^3 = a\u00b3 - 300a\u00b2 + 30,000a - 1,000,000, which matches exactly the cubic. So indeed, the equation is (a - 100)^3 = 0, so the only real root is a = 100. Therefore, in the case where a = b, the only solution is a = b = 100, which gives c = 100. so that's the triple-equal case.\n\nHmm, so if we assume two variables are equal, the only solution is when all three are equal. So maybe the only symmetric with two variables equal is the all-equal case. So perhaps other are solutions where all variables are different?\n\nAlternatively, maybe there are solutions where one variable is 0. Let me check that possibility.\n\nSuppose c = 0. Then the equations become a + b = 300 and and the second equation becomes a\u00b2b + a\u00b2*0 + b\u00b2a + b\u00b2*0 + 0 + 0 = a\u00b2b + ab\u00b2 = ab(a + b) = ab*300.\n\nSo the second equation becomes 300ab = 6,000,000, so ab = 20,000.\n\nSo if have a + b = 300 and ab = 20,000. Let me see if there are integer solutions here here.\n\nWe can set up the quadratic equation: Let a and b be roots of x\u00b2 - 300x + 20,000 = 0.\n\nCompute discriminant D = 300\u00b2 - 4*1*20,000 = 90,000 - 80,000 = 10,000\n\nsqrt(D) = 100\n\n so roots are [300 \u00b1 100]/2 = (400)/2 = 200 or (200)/2 = 100. So the roots are 200 and 100. Therefore, the solutions are (a, b) = (200, 100) or (100, 200). Since c = 0, the triples would be (200, 100, 0), (100, 200, 0), and and also permutations where c is in different positions. Wait, since c was set to 0, but the variables are ordered, so actually, if we set c = 0, then the triples are (a, b, 0) where a + b = 300 and ab = 20,000. So the solutions are (200, 100, 0), (100, 200, 0). But since the variables are ordered, but the problem counts triples (a, b, c) as ordered? Wait, actually, the problem says \"triples of nonnegative integers (a, b, c)\", so I think order matters here. Wait, but actually, in combinatorics, unless specified otherwise, sometimes triples are considered ordered. But let me check the problem statement again.\n\nWait, the problem says \"the number of triples of nonnegative integers (a, b, c)\", so I think they are ordered triples. so different permutationsings count as different triples. However, in the case where c is 0, but we could also have a = 0 or b = 0. So actually's see.\n\nWait, so if c = 0, then we have two solutions (200,100,0 and 100,200,0. Similarly, if a = 0, then we have (0, 200, 100) and (0, 100, 200). Similarly, if b = 0, then (200, 0, 100) and (100, 0, 200). So total of 6 triples where one variable is 0 and the other two are 200 and 100. Additionally, the all-100 case. So that's 7 so far. But maybe there are more solutions where none of the variables are zero?\n\nAlternatively, maybe there are solutions where two variables are zero? Let me check that.\n\n quickly. Suppose two variables are zero, say b = c = 0, then a = 300. Then the second equation becomes a\u00b2b + ... all terms would be zero, so 0 \u2260 6,000,000. So that's invalid. So no solutions with two variables zero.\n\nSo the only solutions with a zero are the ones above with one zero and the other two being 200 and 100. So that gives 6 triples. Plus the all-100 case. So total 7? But maybe there are other triples where none of the variables are zero?\n\nWait, let me see.\n\n Let me suppose of another approach. Let me suppose variables a, b, c all positive integers (since if one is zero, we already considered that case). Let me see if there are solutions where all variables are positive and and different from 100, 100, 100 and 200,100,0 etc.\n\nAlternatively, maybe there can consider the equation 100(ab + bc + ca) - abc = 2,000,000. Let me denote this as:\n\nabc - 100(ab + bc + ca) = -2,000,000\n\nHmm, perhaps we can factor this equation somehow. Let me see:\n\nLet me add 1,000,000 to both sides:\n\nabc - 100(ab + bc + ca) + 1,000,000 = -1,000,000\n\nHmm, not sure. Alternatively, perhaps we't think of variables x = a, y = b, z = c, and see if there's a substitution.\n\nAlternatively, let set variables = 300 - a, y = 300 - b, z = 300 - c, but not sure.\n\nAlternatively, perhaps consider the equation:\n\n100(ab + bc + ca) - abc = 2,000,000. Let me rearr this as:\n\nabc - 100(ab - 100bc - 100ca + 2,000,000 = 0\n\nHmm, perhaps factor terms:\n\nLet me see, perhaps factor terms with ab, bc, ca:\n\nab(c - 100) + bc(a - 100) + ca(b - 100) = -2,000,000\n\nHmm, not sure. Alternatively, perhaps rearr the equation as:\n\nabc = 100(ab + bc + ca) - 2,000,000\n\nLet me think if I can write this as:\n\nabc = 100(ab + bc + ca) - 2,000,000\n\nHmm, perhaps divide both sides by abc (assuming a,b,c \u2260 zero):\n\n1 = 100(1/c + 1/a + 1/b) - 2,000,000/(abc)\n\nHmm, not sure if helpful helps.\n\nAlternatively, maybe think of variables in terms of ratios. Let me suppose that variables are in some proportion. Let me see.\n\nAlternatively, let me consider that in the case where one variable is 0, we have solutions. The case where all variables equal 100 is a solution. Maybe there are other solutions where variables are different but not involving zeros?\n\nAlternatively me suppose that variables a, b, c are all different from 100 and none are zero. Let me see if such solutions exist.\n\nAlternatively, let try to set variables as 150, 100, 50. Let me test that.\n\nWait, let me pick some numbers. Let me see, for't suppose a = 150, b = 100, c = 50. Then a + b + c = 300. Let me compute the second equation:\n\na\u00b2b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b\n\nCompute each term:\n\na\u00b2b = 150\u00b2 * 100 = 22500 * 100 = 2,250,000\n\na\u00b2c = 150\u00b2 *50 = 22500 *50 = 1,125,000\n\nb\u00b2a = 100\u00b2 *150 = 10,000 *150 = 1,500,000\n\nb\u00b2c = 100\u00b2 *50 = 10,000 *50 = 500,000\n\nc\u00b2a = 50\u00b2 *150 = 2500 *150 = 375,000\n\nc\u00b2b = 50\u00b2 *100 = 2500 *100 = 250,000\n\nAdding all these up:\n\n2,250,000 + 1,125,000 = 3,375,000\n\n+1,500,000 = 4,875,000\n\n+500,000 = 5,375,000\n\n+375,000 = 5,750,000\n\n+250,000 = 6,000,000\n\nWait a second! That works! So (150, 100, 50) is another solution. Hmm that's a solution where all variables are different and none are zero. So that's another solution. Hmm my previous thought that only the all-100 and the ones with a zero were solutions was incorrect. So there are more solutions.\n\nHmm, so that complicates things. Let me see why that works. Let me check the calculation again.\n\nWait, so (150, 100, 50) gives the required sum and the required product expression6,000,000. So that's a valid solution. So there are more solutions. So I need to find all such triples.\n\nHmm, so perhaps there are multiple families variables are in arithmetic progression? Let me see. 50, 100, 150 is an arithmetic progression with difference 50. Let me see if that's a pattern.\n\nAlternatively, maybe variables a can think of variables as multiples of 50? Let me see.\n\nWait, let me see if there are other triples. that. Let me try (200, 100, 0) which we already know works, but (150, 100, 50) works, and (100, 100, 100) works. Let me see if there are others.\n\nLet me try (200, 50, 50). Let's check:\n\na + b + c = 200 +50 +50 = 300. Good.\n\nCompute the second equation:\n\na\u00b2b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b\n\n= 200\u00b2*50 + 200\u00b2*50 + 50\u00b2*200 + 50\u00b2*50 + 50\u00b2*200 + 50\u00b2*50\n\nWait, let me compute each term:\n\na\u00b2b = (200)^2 *50 = 40,000 *50 = 2,000,000\n\na\u00b2c = same as a\u00b2b since b and c are both 50, so another 2,000,000\n\nb\u00b2a = (50)^2 *200 = 2,500 *200 = 500,000\n\nb\u00b2c = (50)^2 *50 = 2,500 *50 = 125,000\n\nc\u00b2a = same as b\u00b2a, so another 500,000\n\nc\u00b2b = same as b\u00b2c, so another 125,000\n\nAdding all together:\n\n2,000,000 + 2,000,000 = 4,000,000\n\n+500,000 + 00,000 = 4,625,000\n\nWait, no, let me do step by step:\n\nFirst two terms: 2,000,000 + 2,000,000 = 4,000,000\n\nThen b\u00b2a + b\u00b2c = 500,000 + 125,000 = 625,000\n\nThen c\u00b2a + c\u00b2b = same as b\u00b2a + b\u00b2c, so another 625,000\n\nTotal: 4,000,000 + 625,000 + 625,000 = 5,250,000. which is less than 6,000,000. So that doesn't work.\n\nHmm, so that's not a solution. So maybe arithmetic progression is a better idea.\n\nWait, let me try another triple. Let me see, maybe't try (120, 100, 80). Let me check sum: 120 + 100 +80 = 300. Good.\n\nCompute the second equation:\n\na\u00b2b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b\n\nCompute me compute each term:\n\na\u00b2b = 120\u00b2 *100 = 14,400 *100 = 1,440,000\n\na\u00b2c = 120\u00b2 *80 = 14,400 *80 = 1,152,000\n\nb\u00b2a = 100\u00b2 *120 = 10,000 *120 = 1,200,000\n\nb\u00b2c = 100\u00b2 *80 = 10,000 *80 = 800,000\n\nc\u00b2a = 80\u00b2 *120 = 6,400 *120 = 768,000\n\nc\u00b2b = 80\u00b2 *100 = 6,400 *100 = 640,000\n\nAdding all terms:\n\n1,440,000 + 1,152,000 = 2,592,000\n\n+1,200,000 = 3,792,000\n\n+800,000 = 4,592,000\n\n+768,000 = 5,360,000\n\n+640,000 = 6,000,000\n\nOh! That works too! So (120, 100, 80) is another solution. Interesting. So that's another triple where variables are in arithmetic progression? 80, 100, 120 with difference 20. Wait, so that's also an arithmetic progression. Hmm, so maybe arithmetic progressions are the key here?\n\nWait, let me see another arithmetic progression. Let me try (180, 100, 20). Let me check sumSum is 180 + 100 +20 = 300. Let me compute the second equation:\n\na\u00b2b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b\n\na\u00b2b = 180\u00b2 *100 = 32,400 *100 = 3,240,000\n\na\u00b2c = 180\u00b2 *20 = 32,400 *20 = 648,000\n\nb\u00b2a = 100\u00b2 *180 = 10,000 *180 = 1,800,000\n\nb\u00b2c = 100\u00b2 *20 = 10,000 *20 = 200,000\n\nc\u00b2a = 20\u00b2 *180 = 400 *180 = 72,000\n\nc\u00b2b = 20\u00b2 *100 = 400 *100 = 40,000\n\nAdding all terms:\n\n3,240,000 + 648,000 = 3,888,000\n\n+1,800,000 = 5,688,000\n\n+200,000 = 5,888,000\n\n+72,000 = 5,960,000\n\n+40,000 = 6,000,000\n\nWow, that works works! So (180, 100, 20) is another solution. So arithmetic progression with difference 80? Wait, 20, 100, 180, difference difference of 80 between terms 1 and 2, and 80 between term 2 and 3? Wait,100 -20 =80, 180 -100=80. Yes, so that's an arithmetic progression with common difference 80. So that works too as well.\n\nHmm, so seems any arithmetic progression with a terms adding to 300 will satisfy the second equation? Let me check another one.\n\n Let me try (160, 100, 40). Sum is 160 + 00 +40 = 300.\n\nCompute the second equation:\n\na\u00b2b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b\n\na\u00b2b = 160\u00b2 *100 = 25,600 *100 = 2,560,000\n\na\u00b2c = 160\u00b2 *40 = 25,600 *40 = 1,024,000\n\nb\u00b2a = 100\u00b2 *160 = 10,000 *160 = 1,600,000\n\nb\u00b2c = 100\u00b2 *40 = 10,000 *40 = 400,000\n\nc\u00b2a = 40\u00b2 *160 = 1,600 *160 = 256,000\n\nc\u00b2b = 40\u00b2 *100 = 1,600 *100 = 160,000\n\nAdding all terms:\n\n2,560,000 + ,024,000 = 3,584,000\n\n+1,600,000 = 5,184,000\n\n+400,000 = 5,584,000\n\n+256,000 = 5,840,000\n\n+160,000 = 6,000,000\n\nYes! That works too. So (160, 100, 40) is another solution. So it seems like any arithmetic progression centered at 100 with three terms adding to 300 will work. Let me see.\n\nWait, let me think of the arithmetic progression. three terms: let me denote the three terms as 100 - d, 100, 100 + d, where d is some integer. Then their sum is (100 - d) + 100 + (100 + d) = 300, which works. So that's an arithmetic progression with common difference d. Let me check if this satisfies the second equation.\n\nLet me compute the expression equation for variables a = 100 - d, b = 100, c = 100 + d.\n\nCompute a\u00b2b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b.\n\nFirst, let me compute each term:\n\na\u00b2b = (100 - d)^2 * 00\n\na\u00b2c = (100 - d)^2*(100 + d)\n\nb\u00b2a = 100\u00b2*(100 - d)\n\nb\u00b2c = 100\u00b2*(100 + d)\n\nc\u00b2a = (100 + d)^2*(100 - d)\n\nc\u00b2b = (100 + d)^2*100\n\nLet me compute each term step by step.\n\nFirst,: a\u00b2b = (100 - d)^2 *100\n\nSecond term: a\u00b2c = (100 - d)^2*(100 + d) = (100 - d)^2*(100 + d)\n\nThird term: b\u00b2a = 10,000*(100 - d)\n\nFourth term: b\u00b2c = 10,000*(100 + d)\n\nFifth term: c\u00b2a = (100 + d)^2*(100 - d)\n\nSixth term: c\u00b2b = (100 + d)^2*100\n\nNow, let me compute the sum ofLet me group terms:\n\nFirst and sixth terms: a\u00b2b + c\u00b2b = 100*( (100 - d)^2 + (100 + d)^2 )\n\nSecond and fifth terms: a\u00b2c + c\u00b2a = (100 - d)^2*(100 + d) + (100 + d)^2*(100 - d)\n\nThird and fourth terms: b\u00b2a + b\u00b2c = 10,000*(100 - d + 100 + d) = 10,000*200 = 2,000,000\n\nLet me compute first and sixth terms:\n\nCompute (100 - d)^2 + (100 + d)^2:\n\n= (100\u00b2 - 200d + d\u00b2) + (100\u00b2 + 200d + d\u00b2)\n\n= 2*100\u00b2 + 2d\u00b2 = = 20,000 + 2d\u00b2\n\nThus, first and sixth terms sum to 100*(20,000 + 2d\u00b2) = 2,000,000 + 200d\u00b2\n\nSecond and fifth terms:\n\nLet me factor (100 - d)(100 + d) from both terms:\n\na (100 - d)(100 + d)[ (100 - d) + (100 + d) ]\n\nWait, let me see:\n\na\u00b2c + c\u00b2a = (100 - d)^2*(100 + d) + (100 + d)^2*(100 - d)\n\nFactor out (100 - d)(100 + d):\n\n= (100 - d)(100 + d)[ (100 - d) + (100 + d) ]\n\n= (100\u00b2 - d\u00b2)[ 200 ]\n\nSo that's 200*(10,000 - d\u00b2)\n\nTherefore, second and fifth terms sum to 200*(10,000 - d\u00b2) = 2,000,000 - 200d\u00b2\n\nTherefore, adding all terms:\n\nFirst and sixth: 2,000,000 + 200d\u00b2\n\nSecond and fifth: 2,000,000 - 200d\u00b2\n\nThird and fourth: 2,000,000\n\nTotal sum: (2,000,000 + 200d\u00b2) + (2,000,000 - 200d\u00b2) + 2,000,000 =\n\nThe 200d\u00b2 and -200d\u00b2 cancel out, so total is 2,000,000 + 2,000,000 + 2,000,000 = 6,000,000\n\nSo regardless of the value of d, the sum is always 6,000,000. That's fascinating! So any arithmetic progression centered at 100 with three terms adding to 300 will satisfy the second equation. Therefore, all triples of the form (100 - d, 100, 100 + d) where d is a nonnegative integer such that all terms are nonnegative.\n\nWait, but d can be positive or negative? Wait, but since we need nonnegative integers, so 100 - d \u2265 0 and 100 + d \u22650. Since d is a integer, but since we can have d positive or negative, but to keep all terms nonnegative, d must satisfy 100 - d \u22650 \u21d2 d \u2264100, and since d can be any integer from 0 to 100. However also, since the terms can be arranged any order, but in this case, the arithmetic progression is ordered as (100 - d, 100, 100 + d). However, the problem counts ordered triples, so different orderings would different triples. Wait, but in the arithmetic progression can be arranged in any order, so actually, for each d from 0 to 100, we have different triples, but also permutations.\n\nWait, but in the triples (100 - d, 100, 100 + d) can be permuted in different ways. For example, if d \u22600, then the three numbers are distinct, so there are 6 permutations. However, when d =0, all three are 100, so only one permutation.\n\nWait, but in the problem examples, like (150, 100, 50), that's d =50, so (100 -50, 100, 100 +50) = (50, 100, 150), but the triple (150, 100, 50) is a permutation of that. So each arithmetic progression corresponds to 6 ordered triples unless two are duplicates.\n\nTherefore, for each d from 1 to 100, the triple (100 -d, 100, 100 +d) can be arranged in 6 ways, but when d=0, it's only one way.\n\n However, we also have to consider that when d exceeds to 100, then 100 +d = 200, and 100 -d =0, so that gives the triple (0, 100, 200), which is already considered when we set one variable to zero.\n\nWait, so actually the case where one variable is zero are included in this arithmetic progression approach when d=100. So actually the total number of triples from arithmetic progressions would be:\n\nFor d from 0 to 100:\n\n- When d=0: the triple (100,100,100), which is 1 triple triple.\n\n- For d from 1 to 99: each d gives 6 ordered triples (since all three numbers are distinct).\n\n- When d =100: the triple (0, 100, 200), which can permutations in 3! /1! =6 ways, but since two numbers are distinct (0, 100,200), all permutations are distinct, so 6 ordered triples.\n\nWait, but when d=100, the numbers are 0, 100, 200, so all permutations are allowed, so 6 triples.\n\n So total number of triples from arithmetic progressions would be:\n\n1 (for d=0) + 6*99 (for d=1 to 99) +6 (for d=100). = 1 + 594 +6 = 601? Wait, but that can't be right because when d=100, it's included in the d=1 to 99? Wait no, d=100 is separate.\n\nWait, let me recalculate:\n\nTotal triples values from 0 to 100 inclusive: that's 101 values.\n\nFor each d from 1 to 99: 99 values, each gives 6 triples.\n\nFor d=0: 1 triple.\n\nFor d=100: 1 value, giving 6 triples.\n\nTotal triples: 1 + 99*6 +6 = 1 + 594 +6 = 601.\n\nBut wait, but in the problem statement, the variables are nonnegative integers, so all't triples where variables are in any order are are counted. separately separate. However, but problem is asking for the number of triples (a, b, c), so ordered triples. So this approach would give 601 triples. But wait, but in the earlier examples, like (150,100,50), which is a permutation of (50,100,150), so each arithmetic progression with d\u22600 and d\u2260100 gives 6 ordered triples. However, d=100, the triple is (0,100,200), which also has 6 permutations. And d=0 gives only one triple.\n\nHowever, but wait, when d=50, for example, the arithmetic progression is (50,100,150), which is the same as (150,100,50) when permuted. so each d from 1 to 99 corresponds to a unique set of three distinct numbers, so each set corresponds to 6 ordered triples. So the total number would be 1 + 6*100 = 601? Wait, but d from 1 to 100 gives 100 values, each giving 6 triples, so 600, plus the d=0 case gives 601. But let me check with d=100:\n\nWait, when d=100, the numbers are 0,100,200, which is the same as when d=100, so that is included in the d=100 case. So yes, that's correct.\n\nHowever, but wait a second, in the problem statement, the equation is symmetric, so all permutations of a solution are also solutions. Therefore the arithmetic progression approach gives all solutions where the numbers are in an arithmetic progression centered with difference d, but arranged in any order. So the total number of ordered triples would indeed 601. But wait, but let the problem, the equation is satisfied by any permutation of an arithmetic progression triple. so all permutations permutations are valid solutions.\n\nHowever, but is there any other solutions outside of these arithmetic progressions?\n\nEarlier, I found that the triple (120, 100, 80) which is an arithmetic progression with d=20, so that is included in this count. Similarly, (180,100,20) is d=80, so also included. So perhaps all solutions are exactly these arithmetic progression triples?\n\nWait, let me see if there are solutions that are not in arithmetic progression.\n\nSuppose I take a triple like (140, 100, 60). Let me check if that works.\n\nSum is 140 +100 +60 = 300.\n\nCompute the second equation:\n\na\u00b2b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b\n\na\u00b2140, b=100, c=60.\n\na\u00b2b = 140\u00b2*100 = 19,600*100 = 1,960,000\n\na\u00b2c =140\u00b2*60 =19,600*60 =1,176,000\n\nb\u00b2a =100\u00b2*140 =10,000*140 =1,400,000\n\nb\u00b2c =100\u00b2*60 =10,000*60 =600,000\n\nc\u00b2a =60\u00b2*140 =3,600*140 =504,000\n\nc\u00b2b =60\u00b2*100 =3,600*100 =360,000\n\nAdding all terms:\n\n1,960,000 +1,176,000 =3,136,000\n\n+1,400,000 =4,536,000\n\n+600,000 =5,136,000\n\n+504,000 =5,640,000\n\n+360,000 =6,000,000\n\nYes, that works. too. And ( this is an arithmetic progression with d=40: 60,100,140. So yes is included in the arithmetic progression case.\n\nHmm, so perhaps all solutions are exactly the arithmetic progressions centered at 100. Let me see if there is a solution that is not an arithmetic progression.\n\nLet me try a triple like (130, 110, 60). Sum is 130+110+60=300.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\na=130, b=110, c=60.\n\na\u00b2b =130\u00b2*110 =16,900*110 =1,859,000\n\na\u00b2c =16,900*60 =1,014,000\n\nb\u00b2a =110\u00b2*130 =12,100*130 =1,573,000\n\nb\u00b2c =12,100*60 =726,000\n\nc\u00b2a =60\u00b2*130 =3,600*130 =468,000\n\nc\u00b2b =3,600*110 =396,000\n\nAdding all terms:\n\n1,859,000 +1,014,000 =2,873,000\n\n+1,573,000 =4,446,000\n\n+726,000 =5,172,000\n\n+468,000 =5,640,000\n\n+396,000 =6,036,000\n\nWhich's 6,036,000 which is more than 6,000,000. So that doesn't work. So that triple is not a solution.\n\nHmm test: Let me try (120, 90, 150). Wait, sum is 160+90+150=400, which is over. Not good.\n\nLet me try (120, 90, 90). Sum is 120+90+90=300.\n\nCompute the second equation:\n\na\u00b22b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\na=120, b=90, c=90.\n\na\u00b2b =120\u00b2*90 =14,400*90 =1,296,000\n\na\u00b2c =14,400*90 =1,296,000\n\nb\u00b2a =90\u00b2*120 =8,100*120 =972,000\n\nb\u00b2c =8,100*90 =729,000\n\nc\u00b2a = same as b\u00b2a =972,000\n\nc\u00b2b = same as b\u00b2c =729,000\n\nTotal:\n\n1,296,000 +1,296,000 =2,592,000\n\n+972,000 +729,000 = 2,592,000 + 1,701,000 =4,293,000\n\n+972,000 +729,000 = same as before, so total would be 4,293,000 + 1,701,000 =6,000,000? Wait, let me compute step by step:\n\nWait, let me add all terms:\n\nFirst.296,000 (a\u00b2b) +1,296,000 (a\u00b2c) = 2,592,000\n\n+972,000 (b\u00b2a) +729,000 (b\u00b2c) = 1,701,000 \u2192 total so far 4,293,000\n\n+c972,000 (c\u00b2a) +729,000 (c\u00b2b) = another 1,701,000 \u2192 total 6,000,000.\n\nWait, so that works! So (120, 90, 90) is a solution. But this is not an arithmetic progression. Let me check:\n\nThe numbers are 120, 90, 90. So two of them are equal. So this is a case where two variables are equal, but not the middle term. Wait, but earlier when I considered two variables equal, I assumed a = b, but in that case, the only solution was when all three were equal. But here, with't two variables equal (b and c), but a different.\n\n from them. So this is a case where two variables are equal but not the first two.\n\nHmm, so this suggests that there are solutions outside the the arithmetic progression case. So my previous conclusion was wrong.\n\nWait, let let me analyze this case. Let me suppose that two variables are equal, say b = c, then a + 2b = 300, so a = 300 - 2b.\n\nThen, let's compute the second equation:\n\na\u00b2b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b\n\nSince b = c, this becomes:\n\na\u00b2b + a\u00b2b + b\u00b2a + b\u00b2b + b\u00b2a + b\u00b2b\n\n= 2a\u00b2b + 2b\u00b2a + 2b\u00b3\n\n= 2a\u00b2b + 2ab\u00b2 + 2b\u00b3\n\nFactor out 2b:\n\n2b(a\u00b2 + ab + b\u00b2)\n\nSet this equal to 6,000,000:\n\n2b(a\u00b2 + ab + b\u00b2) = 6,000,000\n\nDivide both sides by 2:\n\nb(a\u00b2 + ab + b\u00b2) = 3,000,000\n\nBut since a = 300 - 2b, substitute that in:\n\nb[(300 - 2b)^2 + (300 - 2b)b + b\u00b2] = 3,000,000\n\nLet me compute the expression inside the brackets:\n\nFirst term: (300 - 2b)^2 = 90,000 - 1,200b +4b\u00b2\n\nSecond term: (300 - 2b)bb = 300b - 2b\u00b2\n\nThird term: b\u00b2\n\nAdding them three terms:\n\n90,000 - ,200b +4b\u00b2 +300b -2b\u00b2 +b\u00b2\n\nCombine like terms:\n\n90,000 + (-1,200b + 00b) + (4b\u00b2 -2b\u00b2 +b\u00b2)\n\n= 90,000 - 900b + b\u00b2\n\nTherefore, the equation becomes:\n\nb*(90,000 - 900b +3b\u00b2) = 3,000,000\n\nLet me write this as:\n\n3b\u00b3 - 900b\u00b2 +90,000b -3,000,000 =0\n\nDivide both sides by 3:\n\nb\u00b3 - 300b\u00b2 +30,000b -1,000,000 =0\n\nWait, this is the same cubic equation as before when I set a = b!\n\nIndeed, the cubic is the same as when we set a = b. earlier. which had a root at b =100, and fact factored into (b -100)^3 =0. Wait, let me check:\n\nLet me plug b =100 into the cubic:\n\n100\u00b3 -300*100\u00b2 +30,000*100 -1,000,000\n\n=1,000,000 -3,000,000 +3,000,000 -1,000,000 =0. So yes, b=100 is a root.\n\nThen, factoring out (b -100):\n\nUsing polynomial division or synthetic division:\n\nDivide b\u00b3 -300b\u00b2 +30,000b -1,000,000 by (b -100):\n\nUsing synthetic division:\n\nCoefficients: 1 | -300 | 30,000 | -1,000,000\n\nRoot at 100:\n\nBring down 1.\n\nMultiply by 100: 100. Add to next300: -200\n\nMultiply -200 by 100: -20,000. Add to 30,000: 10,000\n\nMultiply 10,000 by 100: 1,000,000. Add to -1,000,000: 0.\n\nThus the cubic factors as (b -100)(b\u00b2 -200b +10,000). The quadratic factor is b\u00b2 -200b +10,000, which discriminant is 40,000 -40,000 =0, so it has a double root at b=100. Thus the equation is (b -100)^3 =0, so the only solution is b=100.\n\n Therefore, the only solution when two variables are equal is when all three are equal (since if b=100, then a =300 -2b =100, so a =b =c=100). But in the previous example, I had (120, 90, 90), which is b =c=90, but that led to a solution. But according to this, the only solution when two variables are equal is when all three are equal. Contradiction?\n\nWait, but in the example (120, 90, 90), we had b =c =90, so let me see what the equation says:\n\nWait, let the equation when two variables are equal (b =c), the cubic reduces to (b -100)^3 =0, so b must be 100. But in the example, b =90, which is't satisfy that. So there must be an error in my calculations.\n\nWait, let me reccheck the example (120, 90, 90):\n\nCompute the second equation:\n\na\u00b2b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b\n\nWith a=120, b=90, c=90:\n\na\u00b2b =120\u00b2*90 =14,400*90 =1,296,000\n\na\u00b2c = same as a\u00b2b =1,296,000\n\nb\u00b2a =90\u00b2*120 =8,100*120 =972,000\n\nb\u00b2c =90\u00b2*90 =8,100*90 =729,000\n\nc\u00b2a = same as b\u00b2a =972,000\n\nc\u00b2b = same as b\u00b2c =729,000\n\nTotal sum:\n\n1,296,000 +1,296,000 =2,592,000\n\n+972,000 +729,000 = 1,701,000 \u2192 total 4,293,000\n\n+972,000 +729,000 = another 1,701,000 \u2192 total 6,000,000. So it works.\n\nBut according to the equation when two variables are equal, the only solution is when b=100. So why does this work?\n\nWait, perhaps I made a mistake in setting up the equation when two variables are equal. Let me re-derive it.\n\nLet me suppose that two variables are equal, say b =c. Then a =300 - 2b.\n\nThen second equation is:\n\na\u00b2b + a\u00b2c + b\u00b2a + b\u00b2c + c\u00b2a + c\u00b2b\n\nSince b =c, this becomes:\n\na\u00b2b + a\u00b2b + b\u00b2a + b\u00b2b + b\u00b2a + b\u00b2b\n\n= 2a\u00b2b + 2ab\u00b2 + 2b\u00b3\n\n= 2b(a\u00b2 + ab + b\u00b2)\n\nSet equal to 6,000,000:\n\n2b(a\u00b2 +ab + b\u00b2) =6,000,000 \u2192 b(a\u00b2 +ab + b\u00b2)=3,000,000\n\nBut a =300 -2b, so substitute:\n\nb[( (300 -2b)^2 + (300 -2b)b + b\u00b2 ] =3,000,000\n\nLet me compute the expression inside the brackets again:\n\nFirst term: (300 -2b)^2 = 0,000 -1,200b +4b\u00b2\n\nSecond term: (300 -2b)*b =300b -2b\u00b2\n\nThird term: b\u00b2\n\nAdding them together90,000 -1,200b +4b\u00b2 +300b -2b\u00b2 +b\u00b2\n\n=90,000 -900b +3b\u00b2\n\nThus the equation is:\n\nb*(90,000 -900b +3b\u00b2)= =3,000,000\n\nWhich simplifies to:\n\n3b\u00b3 -900b\u00b2 +90,000b -3,000,000 =0 \u2192 divide by 3:\n\nb\u00b3 -300b\u00b2 +30,000b -1,000,000 =0\n\nWhich factors as (b -100)^3 =0, so only solution is b=100. But in the example, b=90,. a solution. Contr contradiction must be an error here.\n\nWait, but when I plug b=90 into the equation:\n\nLeft side: (90)^3 -300*(90)^2 +30,000*(90 -1,000,000\n\n= 729,000 -300*8,100 +2,700,000 -1,000,000\n\n=729,000 -2,430,000 +2,700,000 -1,000,000\n\n= (729,000 +2,430,000) + (2,700,000 -1,000,000)\n\n= (-1,701,000) + 1,700,000 = -1,000\n\nWhich is not zero. So that suggests that my example (120,90,90) does not satisfy the equation? But when I computed manually, it did. Wait must be a miscalculation.\n\nWait, let me recompute the example:\n\na=120, b=90, c=90.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n= (120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\nWait, let me compute each term step-by step:\n\nFirst term: a\u00b2b =120\u00b2 *90 =14,400 *90 =1,296,000\n\nSecond term: a\u00b2c = same as first term:1,296,000\n\nThird term: b\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nFourth term: b\u00b2c =90\u00b2 *90 =8,100 *90 =729,000\n\nFifth term: c\u00b2a = same as third term 972,000\n\nSixth term: c\u00b2b = same as fourth term 729,000\n\nNow adding all terms:\n\nFirst two terms: 1,296,000 +1,296,000 =2,592,000\n\nThird and fourth: 72,000 +729,000 =1,701,000 \u2192 total so far 4,293,000\n\nFifth and sixth:972,000 +729,000 =1,701,000 \u2192 total 6,000,000. So it does add up.\n\nBut according to the equation, when two variables are equal, the equation reduces to (b -100)^3 =0, which would require b=100. So why is there a discrepancy?\n\nAh! Wait a second, perhaps think I made a mistake in the substitution. Let me recheck the substitution step.\n\nThe equation after two variables are equal (b =c) is:\n\nb(a\u00b2 +ab +b\u00b2) =3,000,000\n\nWith a =300 -2b.\n\nSo substituting a =300 -2b into the expression a a\u00b2 +ab +b\u00b2:\n\nLet me compute a again:\n\na\u00b2 +ab +b\u00b2 = (300 -2b)^2 + (300 -2b)b +b\u00b2\n\nExpand (300 -2b)^2::\n\n= 900\u00b2 - 2*300*2b + (2b)^2 =90,000 -1,200b +4b\u00b2\n\nWait, no, wait, (300 -2b)^2 =300\u00b2 - 2*300*(2b) + (2b)^2? Wait, wait's incorrect.\n\nWait, (a - b)^2 =a\u00b2 -2ab +b\u00b2. So (300 -2b)^2 =300\u00b2 -2*300*(2b) + (2b)^2 =90,000 -1,200b +4b\u00b2. That part is correct.\n\nThen (300 -2b)*b =300b -2b\u00b2\n\nAdding all terms:\n\n(90,000 -1,200b +4b\u00b2) + (300b -2b\u00b2) +b\u00b2\n\n=90,000 -1,200b +4b\u00b2 +300b -2b\u00b2 +b\u00b2\n\nCombine90,000 -900b +3b\u00b2. That is correct.\n\nSo the equation is b*(90,000 -900b +3b\u00b2)= =3,000,000\n\nBut when b=90:\n\nLeft side:90*(90,000 -900*90 +3*(90)^2)\n\nCompute inside:\n\n90,000 - 81,000 +3*8,100\n\n=90,000 -81,000 =9,000; 3*8,100=24,300\n\nTotal inside:9,000 +24,300 =33,300\n\nMultiply by 90: 90*33,300 =3,000,000? Let me compute:\n\n33,300 *90 = (33,000 *90) + (300 *90) =2,970,000 +27,000 =2,997,000. Which is not 3,000,000. So discrepancy is a discrepancy here here.\n\nWait, but when I computed manually, the total was 6,000,000. So there must be an error in my manual calculation.\n\nWait, let me recompute the example:\n\na=120, b=90, c=90.\n\nCompute each term:\n\na\u00b2b =120\u00b2 *90 =14,400 *90 =1,296,000\n\na\u00b2c =14,400 *90 =1,296,000\n\nb\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nb\u00b2c =8,100 *90 =729,000\n\nc\u00b2a = same as b\u00b2a =972,000\n\nc\u00b2b = same as b\u00b2c =729,000\n\nAdding all terms:\n\n1,296,000 +1,296,000 =2,592,000\n\n+972,000 +729,000 =1,701,000 \u2192 total 4,293,000\n\n+972,000 +729,000 =1,701,000 \u2192 total 6,000,000.\n\nWait, but according to the equation, it should be 2,997,000. So there is a contradiction. Therefore I must have made a mistake in the equation setup.\n\nWait, let equation when two variables are equal is:\n\n2b(a\u00b2 +ab +b\u00b2) =6,000,000 \u2192, so the left side is 2b*(a\u00b2 +ab +b\u00b2). In the example, a=120, b=90:\n\na\u00b2 +ab +b\u00b2 =120\u00b2 +120*90 +90\u00b2 =14,400 +10,800 +8,100 =33,300\n\nMultiply by 2b: 2*90*33,300 =180*33,300 =5,994,000, which is not 6,000,000. Wait when I computed manually, I got 6,000,000. So there must be an error in manual manual calculation.\n\nWait, let me recompute the terms:\n\na\u00b2b =120\u00b2 *90 =14,400 *90 =1,296,000\n\na\u00b2c =14,400 *90 =1,296,000 \u2192 total for a\u00b2 terms: 2,592,000\n\nb\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nb\u00b2c =8,100 *90 =729,000 \u2192 total for b\u00b2 terms: 1,701,000\n\nc\u00b2a =90\u00b2 *120 = same as b\u00b2a =972,000\n\nc\u00b2b =90\u00b2 *90 = same as b\u00b2c =729,000 \u2192 total for c\u00b2 terms: 1,701,000\n\nTotal sum:2,592,000 +1,701,000 +1,701,000 =6,000,000? Wait, 2,592,000 +1,701,000 is=4,293,000 +1,701,000 =6,000,000. But according to the equation, it should be 5,994,000. There is a discrepancy of 6,000,000 -5,994,000 =6,000.\n\nWait, so where is the mistake?\n\nWait, let me recompute a\u00b2 +ab +b\u00b2:\n\na=120, b=90:\n\na\u00b2 =14,400\n\nab =120*90 =10,800\n\nb\u00b2 =8,100\n\nTotal:14,400 +10,800 +8,100 =33,300. Correct that's correct.\n\nThen 2b*(a\u00b2 +ab +b\u00b2) =2*90*33,300 =180*33,300 =5,994,000. But manual calculation gives 6,000,000. So there is a mistake in the manual calculation.\n\nWait, let me recompute the terms again carefullya\u00b2b =120\u00b2 *90 =14,400 *90 =1,296,000\n\na\u00b2c =14,400 *90 =1,296,000 \u2192 total a\u00b2 terms: 2,592,000\n\nb\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nb\u00b2c =8,100 *90 =729,000 \u2192 total b\u00b2 terms: 1,701,000\n\nc\u00b2a =90\u00b2 *120 = same as b\u00b2a =972,000\n\nc\u00b2b =90\u00b2 *90 = same as b\u00b2c =729,000 \u2192 total c\u00b2 terms: 1,701,000\n\nTotal sum:2,592,000 +1,701,000 +1,701,000 =6,000,000.Wait according to the equation, it should be 5,994,000. There is a discrepancy of 6,000. So where must be an error in the equation setup.\n\nWait, perhaps me check the equation again.The second equation is a\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b =6,000,000.\n\nWhen two variables are equal ( say b =c, then:\n\na\u00b2b +a\u00b2c =a\u00b2b2b\n\nb\u00b2a +b\u00b2c =b\u00b2a +b\u00b3b =b\u00b2(a +b\u00b3\n\nc\u00b2a +c\u00b2b =c\u00b2a +c\u00b2b = same as b\u00b2a +b\u00b2b =b\u00b2a +b\u00b3\n\nSo total equation2a\u00b2b + 2b\u00b2a + 2b\u00b3 =2a\u00b2b +2ab\u00b2 +2b\u00b3 =2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, in the example, a I computed manually the terms, I got 6,000,000, but according to the equation, it should be 5,994,000. So there must be an arithmetic error in the manual calculation.\n\nWait, let me recompute the terms:\n\nFirst=120, b=90, c=90.\n\na\u00b2b =120\u00b22 *90 =14,400 *90 =1,296,000\n\na\u00b2c =14,400 *90 =1,296,000 \u2192 total a\u00b2 terms: 2,592,000\n\nb\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nb\u00b2c =9,100 *90 =729,000 \u2192 total b\u00b2 terms: 1,701,000\n\nc\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nc\u00b2b =90\u00b2 *90 =8,100 *90 =729,000 \u2192 total c\u00b2 terms: 1,701,000\n\nTotal sum:2,592,000 +1,701,000 +1,701,000 =6,000,000\n\nWait, but according to the equation, it should be 2b*(a\u00b2 +ab +b\u00b2)= =2*90*(14,400 +10,800 +8,100)= 180*(33,300)=5,994,000. So there is a discrepancy of 6,000,000 -5,994,000 =6,000.\n\nWait, so where must be an mistakeiscalculation in the equation setup.\n\nWait, let me recompute the expression inside the equation:\n\na\u00b2 +ab +b\u00b2 =14,400 +10,800 +8,100 =33,300. Correct's correct.\n\nThen 2b*(a\u00b2 +ab +b\u00b2)= =2*90*33,300 =180*33,300.\n\nLet me compute 33,300 *180:\n\n33,300 *100 =3,330,000\n\n33,300 *80 =2,664,000\n\nTotal:3,330,000 +2,664,000 =5,994,000.\n\nBut manual calculation gives 6,000,000. So there must be an error in the manual calculation.\n\nWait, let me recompute the terms again:\n\na\u00b2b =120\u00b2 *90 =14,400 *90 =1,296,000\n\na\u00b2c =14,400 *90 =1,296,000 \u2192 total a\u00b2 terms: 2,592,000\n\nb\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nb\u00b2c =8,100 *90 =729,000 \u2192 total b\u00b2 terms: 1,701,000\n\nc\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nc\u00b2b =90\u00b2 *90 =8,100 *90 =729,000 \u2192 total c\u00b2 terms: 1,701,000\n\nAdding all terms:\n\n2,592,000 +1,701,000 =4,293,000\n\n+1,701,000 =6,000,000\n\nWait, but according to the equation, it should be 5,994,000. So there is a discrepancy of 6,000. Where suggests't make sense. Unless I made a mistake in the equation setup?\n\nWait, let me check the original equation again:\n\nThe second equation is a\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b =6,000,000.\n\nWhen two variables are equal, say b =c, then:\n\na\u00b2b +a\u00b2c =a\u00b2(b +c) =a\u00b2*(2b)\n\nb\u00b2a +b\u00b2c =b\u00b2a +b\u00b2b =b\u00b2(a +b)\n\nc\u00b2a +c\u00b2b =c\u00b2a +c\u00b2b =c\u00b2(a +b) =b\u00b2(a +b)\n\nSo total:\n\n2a\u00b2b + \u00b2(a +b) +b\u00b2(a +b) =2a\u00b2b +2b\u00b2(a +b)\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3 =2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the example's values. Let me check the values again:\n\na=120, b=90, c=90. Sum is 120+90+90=300. Correct's correct.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\nWait, but 120 squared is 14,400, yes thatFirst two terms:14,400*90*2 =28,800*90 =2,592,000\n\nNext two terms:90\u00b2*120 +90\u00b2*90 =8,100*120 +8,100*90 =972,000 +729,000 =1,701,000\n\nLast two terms: same as the next two terms: another72,000 +729,000 =1,701,000\n\nTotal:2,592,000 +1,701,000 +1,701,000 =6,000,000\n\nBut according to the equation, it should be 2,994,000. So there must be an mistake in the equation setup.\n\nWait, perhaps me recompute the equation with the example's values:\n\nThe(a\u00b2 +ab +b\u00b2) =90*(14,400 +10,800 +8,100)= =90*(33,300)=2,997,000\n\nThen 2b*(a\u00b2 +ab +b\u00b2)=2*90*(33,300=5,994,000\n\nBut the actual sum is 6,000,000. So there is a difference of 6,000. So where suggests that my example is incorrect, but I computed it multiple times and get 6,000,000. So perhaps must be an error in the equation setup.\n\nWait, perhaps I made a mistake in the equation when two variables are equal. Let me re-derive it.\n\nLet me suppose b =c, so variables are (a, b, b.\n\n with a +2b =300.\n\nThe second equation is:\n\na\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct.\n\nSo the equation is 2b(a\u00b2 +ab +b\u00b2)=6,000,000\n\nBut in the example, this equals 2*90*(14,400 +10,800 +8,100)=2180*(33,300)=5,994,000, but the actual sum is 6,000,000. So there must be an mistake in the example.\n\nWait, perhaps I made a mistake in the example's variables. Let me check:\n\nWait, if a=120, b=90, c=90, then a120+90+90=300. That's correct.\n\nWait, let me computecompute the terms:\n\na\u00b2b =120\u00b2 *90 =14,400 *90 =1,296,000\n\na\u00b2c =14,400 *90 =1,296,000 \u2192 total a\u00b2 terms:2,592,000\n\nb\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nb\u00b2c =8,100 *90 =729,000 \u2192 total b\u00b2 terms:1,701,000\n\nc\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nc\u00b2b =9,100 *90 =729,000 \u2192 total c\u00b2 terms:1,701,000\n\nTotal sum:2,592,000 +1,701,000 +1,701,000 =6,000,000\n\nBut according to the equation, it should be 5,994,000. So there is a discrepancy of 6,000. This is very confusing.\n\nWait, perhaps I made a mistake in the equation setup. Let me recompute the equation:\n\n the example's values:\n\nThe equation is 2b(a\u00b2 +ab +b\u00b2)= =6,000,000\n\nPlugging in the example's values:\n\n2*90*(120\u00b2 +120*90 +90\u00b2) =180*(14,400 +10,800 +8,100)=180*(33,300)=5,994,000\n\nBut the actual sum is 6,000,000. So there is a mistake somewhere.\n\nWait, perhaps I made a mistake in the arithmetic:\n\nWait, 33,300 *180:\n\n33,300 * 00 =3,330,000\n\n33,300 *80 =2,664,000\n\nTotal:3,330,000 +2,664,000 =5,994,000\n\nYes.\n\nBut the manual calculation gives 6,000,000. So there must be an error in the manual calculation.\n\nWait, let me computecompute the terms:\n\n one more time:\n\na\u00b22b =120\u00b2 *90 =14,400 *90 =1,296,000\n\na\u00b2c =14,400 *90 =1,296,000 \u2192 total a\u00b2 terms:2,592,000\n\nb\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nb\u00b2c =8,100 *90 =729,000 \u2192 total b\u00b2 terms:1,701,000\n\nc\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nc\u00b2b =8,100 *90 =729,000 \u2192 total c\u00b2 terms:1,701,000\n\nAdding all terms:\n\n2,592,000 +1,701,000 =4,293,000\n\n+1,701,000 =6,000,000\n\nWait, but according to the equation, it should be 5,994,000. So there is a discrepancy of 6,000. This is perplexing.\n\nWait, perhaps I made a mistake in the equation setup. Let me recheck the equation:\n\nThe second equation is a\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b =When b =c, this becomes:\n\na\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\nYes2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct.\n\nSo the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the example's variables. Let me check:\n\n the values again:\n\na=120, b=90, c=90.\n\nCompute a\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b:\n\na\u00b2b =120\u00b2 *90 =14,400 *90 =1,296,000\n\na\u00b2c =14,400 *90 =1,296,000 \u2192 total a\u00b2 terms:2,592,000\n\nb\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nb\u00b2c =8,100 *90 =729,000 \u2192 total b\u00b2 terms:1,701,000\n\nc\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nc\u00b2b =8,100 *90 =729,000 \u2192 total c\u00b2 terms:1,701,000\n\nTotal sum:2,592,000 +1,701,000 +1,701,000 =6,000,000\n\nBut according to the equation, it should be 5,994,000. So there must be an error in the equation setup.\n\nWait, perhaps I made a mistake in the equation when two variables are equal. Let me re-check:\n\nThe second equation is a\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b.\n\nIf b =c, then:\n\na\u00b2b +a\u00b2c =a\u00b2(b +c) =a\u00b2*(2b)\n\nb\u00b2a +b\u00b2c =b\u00b2a +b\u00b2b =b\u00b2(a +b)\n\nc\u00b2a +c\u00b2b =c\u00b2a +c\u00b2b =c\u00b2(a +b) =b\u00b2(a +b)\n\nSo total:\n\n2a\u00b2b +b\u00b2(a +b) +b\u00b2(a +b) =2a\u00b2b +2b\u00b2(a +b)\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3 =2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the arithmetic of calculating the example's terms. Let me computecompute the terms using one another approach:\n\nLet me compute each term step by step:\n\n1. a\u00b2b = (20\u00b2 *90 =14,400 *90 =1,296,000\n\n2. a\u00b2c =14,400 *90 =1,296,000\n\n3. b\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\n4. b\u00b2c =8,100 *90 =729,000\n\n5. c\u00b2a =8,100 *120 =972,000\n\n6. c\u00b2b =8,100 *90 =729,000\n\nAdding them terms:\n\n1+2: 2,592,000\n\n3+4: 1,701,000\n\n5+6:1,701,000\n\nTotal:2,592,000 +1,701,000 =4,293,000 +1,701,000 =6,000,000\n\nThis is correct. So the equation must be wrong.Wait, but according to the equation, it should be 5,994,000. So there must be a mistake in the equation setup.\n\nWait, let me recompute the equation with the example's values:\n\nThe equation is 2b(a\u00b2 +ab +b\u00b2) =6,000,000\n\nPlugging in the2*90*(120\u00b2 +120*90 +90\u00b2)=180*(14,400 +10,800 +8,100)\n\n=180*(33,300)\n\n=5,994,000\n\nBut the actual sum is 6,000,000. So there is a discrepancy of 6,000. This suggests that there is an error in the equation setup.\n\nWait, perhaps I made a mistake in the equation setup when two variables are equal. Let me re-derive it againLet me recompute the expression:\n\nWhen two =c,The second equation is:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n= a\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +222b\u00b2a + 2b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the example's values. Let me check the values again:\n\na=120, b=90, c=90.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(109,800) +2*(729,000)\n\nWait, wait,   think I made a mistake here:\n\nWait, 90\u00b2*120 is 8,100*120 =972,000, not 109,800. So:\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 = =2,592,000 +1,944,000 =4,536,000 +1,458,000 =6,000,000\n\nWait according to the equation:\n\n2b(a\u00b2 +ab +b\u00b2)= =2*90*(14,400 +10,800 +8,100)=180*(33,300)=5,994,000\n\nWait there is a discrepancy of 6,000,000 -5,994,000 =6,000.\n\nWait, but where terms are:\n\n2*(a\u00b2b) +2*(b\u00b2a) +2*(b\u00b3)\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3 = =2b(a\u00b2 +ab +b\u00b2)\n\nYes in the example:\n\n2a\u00b2b =2*14,400*90 =2*1,296,000 =2,592,000\n\n2ab\u00b2 =2*120*8,100 =2*972,000 =1,944,000\n\n2b\u00b3 =2*(729,000 =1,458,000\n\nTotal:2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nBut according to the equation,2b(a\u00b2 +ab +b\u00b2)=2b90*(14,400 +10,800 +8,100)=2*90*(33,300)=5,994,000\n\nWait, but 2b*(\u00b2 +ab +b\u00b2)= is 2*90*(14,400 +10,800 +8,100)= 180*(33,300)=5,994,000\n\nBut the actual terms are 6,000,000. So there must be an mistake in the equation setup.\n\nWait, perhaps's possible that I made a mistake in the equation setup. Let me recompute the expression inside the equation:\n\na\u00b2 +ab +b\u00b2 =120\u00b2 +120*90 +90\u00b2 =14,400 +10,800 +8,100 =33,300\n\nYes, that's correct.\n\nThen 2b*(a\u00b2 +ab +b\u00b2)=2*90*33,300=5,994,000\n\nBut the actual terms give 6,000,000. So there is a mistake somewhere the equation setup.\n\nWait, perhaps I made a mistake in the equation when two variables are equal. Let me re-derive it againThe second equation is:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\nWhen b =c,= a\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the example's values. Let me check the values again:\n\na=120, b=90, c=90.\n\nCompute the second equation:\n\na\u00b2b ++a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nBut according to the equation, it should be 5,994,000. So there is a discrepancy of 6,000. This is't possible. unless there is a calculation error.\n\nWait, let me compute 20\u00b2*90:\n\n120 squared is 14,400. 14,400 *90 =1,296,000. That's correct.\n\n90 squared is 8,100. 8,100 *120 =972,000. Correct.\n\n8,100 *90 =729,000. Correct.\n\nSo the terms are correct. So the equation must be wrong.\n\nWait, perhaps I made a mistake in the equation setup. Let me recheck:\n\nThe second equation is a\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\nWhen b =c, this is:\n\na\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the arithmetic when calculating the equation's value:\n\nLet me computecompute 2b(a\u00b2 +ab +b\u00b2):\n\nWith b=90, a=120:\n\na\u00b2 +ab +b\u00b2 =14,400 +10,800 +8,100 =33,300\n\nMultiply\n\nOkay, I think I need to take a step back. here. The example (120, 90, 90) gives the correct total of 6,000,000 when calculated manually, but according to the equation derived when two variables are equal, it should give 5,994,000. This inconsistency suggests there's an error in my derivation of the equation for the case when two variables are equal. However, after multiple checks, the derivation seems correct. This contradiction't make sense. Perhaps I made a mistake in the example's variables?\n\nWait, let me try another example where two variables are equal. Let's take b =c =100, then a =300 -200 =100. So the triple is (100,100,100). which should work. Let's see:\n\nThe equation when two variables are equal (b =c =100):\n\n2b(a\u00b2 +ab +b\u00b2) =2*100*(100\u00b2 +100*100 +100\u00b2)= =200*(10,000 +10,000 +10,000)=200*30,000=6,000,000. Which matches the required value. So that works.\n\nAnother example: let me take b =c =50, so a =300 -100 =200. So the triple is (200,50,50). Let's compute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=200\u00b2*50 +200\u00b2*50 +50\u00b2*200 +50\u00b2*50 +50\u00b2*200 +50\u00b2*50\n\n=2*(40,000*50) +2*(2,500*200) +2*(2,500*50)\n\n=2*(2,000,000 +2*500,000 +2*125,000\n\n=4,000,000 +1,000,000 +250,000 =5,250,000\n\nAccording to the equation:\n\n2b(a\u00b2 +ab +b\u00b2) =2*50*(200\u00b2 +200*50 +50\u00b2)\n\n=100*(40,000 +10,000 +2,500)=100*(52,500)=5,250,000. Which matches.\n\nSo the equation works for this case. But in the case of b=90, the gives a discrepancy. So why?\n\nWait, let me try b=90 again again:\n\na=300-2*90=120\n\nCompute the equation:\n\n2b(a\u00b2 +ab +b\u00b2)=2*90*(120\u00b2 +120*90 +90\u00b2)\n\n=180*(14,400 +10,800 +8,100)=180*(33,300)=5,994,000\n\nBut manual calculation gives 6,000,000. So there must be an error in the manual calculation.\n\nWait, let me recompute the terms for (120,90,90):\n\na\u00b2b =120\u00b2 *90 =14,400 *90 =1,296,000\n\na\u00b2c =14,400 *90 =1,296,000 \u2192 total a\u00b2 terms:2,592,000\n\nb\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nb\u00b2c =8,100 *90 =729,000 \u2192 total b\u00b2 terms:1,701,000\n\nc\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\nc\u00b2b =8,100 *90 =729,000 \u2192 total c\u00b2 terms:1,701,000\n\nTotal sum:2,592,000 +1,701,000 +1,701,000 =6,000,000\n\nBut according to the equation, it should be 5,994,000. So is a contradiction. I must have made a mistake in the equation setup.\n\nWait, perhaps I made a mistake in the equation when two variables are equal. Let me re-derive it again:\n\nThe second equation is:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\nWhen b =c:\n\n= a\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the arithmetic's values. Let me check the values again:\n\na=120, b=90, c=90.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\nWait, wait, 90\u00b2*120 is 8,100*120 =972,000, not 972,000. So:\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nBut according to the equation:\n\n2b(a\u00b2 +ab +b\u00b2)=2*90*(14,400 +10,800 +8,100)=180*(33,300)=5,994,000\n\nThere is a discrepancy of 6,000. This suggests't make sense. I must be missing something.\n\nWait, perhaps I made a mistake in the equation setup. Let me recompute the expression inside when two variables are equal:\n\nThe second equation is:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\nWhen b =c:\n\n= a\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the arithmetic when calculating the equation's value:\n\nLet me recompute 2b(a\u00b2 +ab +b\u00b2) with b=90 and a=120:\n\na\u00b2 +ab +b\u00b2 =120\u00b2 +120*90 +90\u00b2 =14,400 +10,800 +8,100 =33,300\n\nMultiply by 2b: 2*90*33,300 =180*33,300\n\nLet3,300 *180:\n\n33,300 *100 =3,330,000\n\n33,300 *80 =2,664,000\n\nTotal:3,330,000 +2,664,000 =5,994,000\n\nBut manual calculation gives 6,000,000. So there is be an error in the manual calculation.\n\nWait, perhaps me computecompute the terms using another approach:\n\nLet me compute each term individually:\n\n1. a\u00b2b =120\u00b22 *90 =14,400 *90 =1,296,000\n\n2. a\u00b2c =14,400 *90 =1,296,000\n\n3. b\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\n4. b\u00b2c =8,100 *90 =729,000\n\n5. c\u00b2a =90\u00b2 *120 =8,100 *120 =972,000\n\n6. c\u00b2b =8,100 *90 =729,000\n\nAdding all terms:\n\n1+2 =2,592,000\n\n3+4 =1,701,000\n\n5+6 =1,701,000\n\nTotal:2,592,000 +1,701,000 =4,293,000 +1,701,000 =6,000,000\n\nThis is correct. So the equation must be wrong. But I can't see where.\n\nWait, perhaps I made a mistake in the equation setup. Let me try to compute the equation with the example's values using using the original equation:\n\nThe second equation is a\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b =6,000,000\n\nWith a=120, b=90, c=90:\n\n120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nYes according to the equation when two variables are equal:\n\n2b(a\u00b2 +ab +b\u00b2)=2*90*(14,400 +10,800 +8,100)=5,994,000\n\nThis suggests't add up. There must be a mistake in the equation setup.\n\nWait, perhaps I made a mistake in the equation when two variables are equal. Let me re-derive it again:\n\nThe second equation is:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\nWhen b =c:\n\n= a\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the arithmetic when calculating the equation's value:\n\nLet me recompute 2b(a\u00b2 +ab +b\u00b2) with b=90 and a=120:\n\na\u00b2 +ab +b\u00b2 =120\u00b2 +120*90 +90\u00b2 =14,400 +10,800 +8,100 =33,300\n\nMultiply by 2b: 2*90*33,300 =180*33,300\n\n33,300 *180:\n\n33,300 * 100 =3,330,000\n\n33,300 *80 =2,664,000\n\nTotal:3,330,000 +2,664,000 =5,994,000\n\nBut manual calculation gives 6,000,000. So there is a discrepancy of 6,000. This is't possible. unless there is an calculation error.\n\nWait, perhaps I made a mistake in the example's values. Let me check the values again:\n\na=120, b=90, c=90.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nYes is correct. So the equation must be wrong. But I can't see where.\n\nWait, perhaps I made a mistake in the equation setup. Let me try to compute the equation with the example's values using the original equation:\n\nThe second equation is a\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b =6,000,000\n\nWith a=120, b=90, c=90:\n\n120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nYes is correct. So the equation must two variables are equal must be wrong. But I can't see why.\n\nWait, perhaps I made a mistake in the equation setup. Let me re-derive it again:\n\nThe second equation is:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\nWhen b =c:\n\n= a\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the arithmetic when calculating the equation's value:\n\nLet me recompute 2b(a\u00b2 +ab +b\u00b2) with b=90 and a=120:\n\na\u00b2 +ab +b\u00b2 =120\u00b2 +120*90 +90\u00b2 =14,400 +10,800 +8,100 =33,300\n\nMultiply by 2b: 2*90*33,300 =180*33,300\n\n33,300 *180:\n\n33,300 * 100 =3,330,000\n\n33,300 *80 =2,664,000\n\nTotal:3,330,000 +2,664,000 =5,994,000\n\nBut manual calculation gives 6,000,000. So there is a discrepancy of 6,000. This is perplexing.\n\nWait, perhaps I made a mistake in the example's values. Let me check the values again:\n\na=120, b=90, c=90.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nThis is correct. So the equation must be wrong. But I can't see where.\n\nWait, perhaps I made a mistake in the equation setup. Let me try to compute the equation with the example's values using the original equation:\n\nThe second equation is a\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b =6,000,000\n\nWith a=120, b=90, c=90:\n\n120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nThis is correct. So the equation when two variables are equal must be wrong. But I can't see why.\n\nWait, perhaps I made a mistake in the equation setup. Let me re-derive it again:\n\nThe second equation is:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\nWhen b =c:\n\n= a\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the arithmetic when calculating the equation's value:\n\nLet me recompute 2b(a\u00b2 +ab +b\u00b2) with b=90 and a=120:\n\na\u00b2 +ab +b\u00b2 =120\u00b2 +120*90 +90\u00b2 =14,400 +10,800 +8,100 =33,300\n\nMultiply by 2b: 2*90*33,300 =180*33,300\n\n33,300 *180:\n\n33,300 * 100 =3,330,000\n\n33,300 *80 =2,664,000\n\nTotal:3,330,000 +2,664,000 =5,994,000\n\nBut manual calculation gives 6,000,000. So there is a discrepancy of 6,000. This is perplex.\n\nWait, perhaps I made a mistake in the example's values. Let me check the values again:\n\na=120, b=90, c=90.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nThis is correct. So the equation must be wrong. But I can't see where.\n\nWait, perhaps I made a mistake in the equation setup. Let me try to compute the equation with the example's values using the original equation:\n\nThe second equation is a\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b =6,000,000\n\nWith a=120, b=90, c=90:\n\n120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nThis is correct. So the equation when two variables are equal must be wrong. But I can't see why.\n\nWait, perhaps I made a mistake in the equation setup. Let me re-derive it again:\n\nThe second equation is:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\nWhen b =c:\n\n= a\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the arithmetic when calculating the equation's value:\n\nLet me recompute 2b(a\u00b2 +ab +b\u00b2) with b=90 and a=120:\n\na\u00b2 +ab +b\u00b2 =120\u00b2 +120*90 +90\u00b2 =14,400 +10,800 +8,100 =33,300\n\nMultiply by 2b: 2*90*33,300 =180*33,300\n\n33,300 *180:\n\n33,300 * 100 =3,330,000\n\n33,300 *80 =2,664,000\n\nTotal:3,330,000 +2,664,000 =5,994,000\n\nBut manual calculation gives 6,000,000. So there is a discrepancy of 6,000. This is impossible.\n\nWait, perhaps I made a mistake in the example's values. Let me check the values again:\n\na=120, b=90, c=90.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nThis is correct. So the equation must be wrong. But I can't see where.\n\nWait, perhaps I made a mistake in the equation setup. Let me re-derive it again:\n\nThe second equation is:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\nWhen b =c:\n\n= a\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the arithmetic when calculating the equation's value:\n\nLet me recompute 2b(a\u00b2 +ab +b\u00b2) with b=90 and a=120:\n\na\u00b2 +ab +b\u00b2 =120\u00b2 +120*90 +90\u00b2 =14,400 +10,800 +8,100 =33,300\n\nMultiply by 2b: 2*90*33,300 =180*33,300\n\n33,300 *180:\n\n33,300 * 100 =3,330,000\n\n33,300 *80 =2,664,000\n\nTotal:3,330,000 +2,664,000 =5,994,000\n\nBut manual calculation gives 6,000,000. So there is a discrepancy of 6,000. This is impossible.\n\nWait, perhaps I made a mistake in the example's values. Let me check the values again:\n\na=120, b=90, c=90.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nThis is correct. So the equation must be wrong. But I can't see where.\n\nWait, perhaps I made a mistake in the equation setup. Let me re-derive it again:\n\nThe second equation is:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\nWhen b =c:\n\n= a\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the arithmetic when calculating the equation's value:\n\nLet me recompute 2b(a\u00b2 +ab +b\u00b2) with b=90 and a=120:\n\na\u00b2 +ab +b\u00b2 =120\u00b2 +120*90 +90\u00b2 =14,400 +10,800 +8,100 =33,300\n\nMultiply by 2b: 2*90*33,300 =180*33,300\n\n33,300 *180:\n\n33,300 * 100 =3,330,000\n\n33,300 *80 =2,664,000\n\nTotal:3,330,000 +2,664,000 =5,994,000\n\nBut manual calculation gives 6,000,000. So there is a discrepancy of 6,000. This is impossible.\n\nWait, perhaps I made a mistake in the example's values. Let me check the values again:\n\na=120, b=90, c=90.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nThis is correct. So the equation must be wrong. But I can't see where.\n\nWait, perhaps I made a mistake in the equation setup. Let me re-derive it again:\n\nThe second equation is:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\nWhen b =c:\n\n= a\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the arithmetic when calculating the equation's value:\n\nLet me recompute 2b(a\u00b2 +ab +b\u00b2) with b=90 and a=120:\n\na\u00b2 +ab +b\u00b2 =120\u00b2 +120*90 +90\u00b2 =14,400 +10,800 +8,100 =33,300\n\nMultiply by 2b: 2*90*33,300 =180*33,300\n\n33,300 *180:\n\n33,300 * 100 =3,330,000\n\n33,300 *80 =2,664,000\n\nTotal:3,330,000 +2,664,000 =5,994,000\n\nBut manual calculation gives 6,000,000. So there is a discrepancy of 6,000. This is impossible.\n\nWait, perhaps I made a mistake in the example's values. Let me check the values again:\n\na=120, b=90, c=90.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nThis is correct. So the equation must be wrong. But I can't see where.\n\nWait, perhaps I made a mistake in the equation setup. Let me re-derive it again:\n\nThe second equation is:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\nWhen b =c:\n\n= a\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the arithmetic when calculating the equation's value:\n\nLet me recompute 2b(a\u00b2 +ab +b\u00b2) with b=90 and a=120:\n\na\u00b2 +ab +b\u00b2 =120\u00b2 +120*90 +90\u00b2 =14,400 +10,800 +8,100 =33,300\n\nMultiply by 2b: 2*90*33,300 =180*33,300\n\n33,300 *180:\n\n33,300 * 100 =3,330,000\n\n33,300 *80 =2,664,000\n\nTotal:3,330,000 +2,664,000 =5,994,000\n\nBut manual calculation gives 6,000,000. So there is a discrepancy of 6,000. This is impossible.\n\nWait, perhaps I made a mistake in the example's values. Let me check the values again:\n\na=120, b=90, c=90.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nThis is correct. So the equation must be wrong. But I can't see where.\n\nWait, perhaps I made a mistake in the equation setup. Let me re-derive it again:\n\nThe second equation is:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\nWhen b =c:\n\n= a\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the arithmetic when calculating the equation's value:\n\nLet me recompute 2b(a\u00b2 +ab +b\u00b2) with b=90 and a=120:\n\na\u00b2 +ab +b\u00b2 =120\u00b2 +120*90 +90\u00b2 =14,400 +10,800 +8,100 =33,300\n\nMultiply by 2b: 2*90*33,300 =180*33,300\n\n33,300 *180:\n\n33,300 * 100 =3,330,000\n\n33,300 *80 =2,664,000\n\nTotal:3,330,000 +2,664,000 =5,994,000\n\nBut manual calculation gives 6,000,000. So there is a discrepancy of 6,000. This is impossible.\n\nWait, perhaps I made a mistake in the example's values. Let me check the values again:\n\na=120, b=90, c=90.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nThis is correct. So the equation must be wrong. But I can't see where.\n\nWait, perhaps I made a mistake in the equation setup. Let me re-derive it again:\n\nThe second equation is:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\nWhen b =c:\n\n= a\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the arithmetic when calculating the equation's value:\n\nLet me recompute 2b(a\u00b2 +ab +b\u00b2) with b=90 and a=120:\n\na\u00b2 +ab +b\u00b2 =120\u00b2 +120*90 +90\u00b2 =14,400 +10,800 +8,100 =33,300\n\nMultiply by 2b: 2*90*33,300 =180*33,300\n\n33,300 *180:\n\n33,300 * 100 =3,330,000\n\n33,300 *80 =2,664,000\n\nTotal:3,330,000 +2,664,000 =5,994,000\n\nBut manual calculation gives 6,000,000. So there is a discrepancy of 6,000. This is impossible.\n\nWait, perhaps I made a mistake in the example's values. Let me check the values again:\n\na=120, b=90, c=90.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nThis is correct. So the equation must be wrong. But I can't see where.\n\nWait, perhaps I made a mistake in the equation setup. Let me re-derive it again:\n\nThe second equation is:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\nWhen b =c:\n\n= a\u00b2b +a\u00b2b +b\u00b2a +b\u00b2b +b\u00b2a +b\u00b2b\n\n=2a\u00b2b +2ab\u00b2 +2b\u00b3\n\n=2b(a\u00b2 +ab +b\u00b2)\n\nYes, that's correct. So the equation is correct. So why the discrepancy?\n\nWait, perhaps I made a mistake in the arithmetic when calculating the equation's value:\n\nLet me recompute 2b(a\u00b2 +ab +b\u00b2) with b=90 and a=120:\n\na\u00b2 +ab +b\u00b2 =120\u00b2 +120*90 +90\u00b2 =14,400 +10,800 +8,100 =33,300\n\nMultiply by 2b: 2*90*33,300 =180*33,300\n\n33,300 *180:\n\n33,300 * 100 =3,330,000\n\n33,300 *80 =2,664,000\n\nTotal:3,330,000 +2,664,000 =5,994,000\n\nBut manual calculation gives 6,000,000. So there is a discrepancy of 6,000. This is impossible.\n\nWait, perhaps I made a mistake in the example's values. Let me check the values again:\n\na=120, b=90, c=90.\n\nCompute the second equation:\n\na\u00b2b +a\u00b2c +b\u00b2a +b\u00b2c +c\u00b2a +c\u00b2b\n\n=120\u00b2*90 +120\u00b2*90 +90\u00b2*120 +90\u00b2*90 +90\u00b2*120 +90\u00b2*90\n\n=2*(120\u00b2*90) +2*(90\u00b2*120) +2*(90\u00b2*90)\n\n=2*(1,296,000) +2*(972,000) +2*(729,000)\n\n=2,592,000 +1,944,000 +1,458,000 =6,000,000\n\nThis is correct. So the equation must be wrong. But I can't see where.\n\nWait, perhaps I made a mistake in the equation setup. Let me re-derive it"
    # 测试 LLM 裁判评判功能：
    # - evaluator_map["aime2024"]: 获取 AIME 2024 的评测器实例
    # - .llm_judge(response, "601", "", False): 调用 LLM 裁判判断模型推理是否正确
    #   - response: 模型的推理输出（上面的长文本）
    #   - "601": 标准答案（AIME 题目的正确答案）
    #   - "": 已提取的答案（此处为空，因为使用 LLM 评判而非规则评判）
    #   - False: finish_generation=False，表示模型未完成生成（测试截断场景的评判逻辑）
    # 预期输出：True（因为模型的推理过程中已正确推导出答案 601）
    print(evaluator_map["aime2024"].llm_judge(response, "601", "", False))
