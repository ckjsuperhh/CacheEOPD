"""
setup.py - EOPD (Entropy-gated On-Policy Distillation) 项目的后备安装脚本。

当 pyproject.toml 无法正常工作时，使用此文件作为回退安装入口。
该脚本定义了 verl 包的核心依赖、可选依赖（如 vllm、sglang、mcore 等）
以及包的元数据信息。EOPD 是基于 verl 框架的 fork，用于 LLM 蒸馏研究。
"""

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# setup.py 是 pyproject.toml 不工作时的后备安装脚本
import os
from pathlib import Path

from setuptools import find_packages, setup

# 获取当前文件所在目录作为版本文件夹的基准路径
version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

# 从 verl/version/version 文件中读取版本号
with open(os.path.join(version_folder, "verl/version/version")) as f:
    __version__ = f.read().strip()

# 核心安装依赖列表 —— 安装 verl/EOPD 时必需的第三方库
install_requires = [
    "accelerate",          # HuggingFace 分布式训练加速库
    "codetiming",          # 代码执行计时工具
    "datasets",            # HuggingFace 数据集加载库
    "dill",                # Python 对象序列化（比 pickle 更强大）
    "hydra-core",          # 配置管理框架，用于管理训练超参数
    "numpy<2.0.0",         # 数值计算库（限制版本以避免兼容性问题）
    "pandas",              # 数据处理与分析库
    "peft",                # HuggingFace 参数高效微调库（LoRA 等）
    "pyarrow>=19.0.0",     # Apache Arrow 的 Python 绑定，用于 Parquet 文件处理
    "pybind11",            # C++ 与 Python 互操作绑定工具
    "pylatexenc",          # LaTeX 编码/解码工具，用于数学公式处理
    "ray[default]>=2.41.0", # Ray 分布式计算框架，verl 的核心调度引擎
    "torchdata",           # PyTorch 数据加载工具库
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0", # 张量字典数据结构，用于 RL 数据传递
    "transformers",        # HuggingFace Transformers，LLM 模型加载与推理
    "wandb",               # Weights & Biases 实验跟踪与可视化平台
    "packaging>=20.0",     # Python 包版本解析工具
    "tensorboard",         # TensorBoard 训练日志可视化工具
]

# 以下为可选依赖组，通过 pip install verl[组名] 安装
TEST_REQUIRES = ["pytest", "pre-commit", "py-spy", "pytest-asyncio", "pytest-rerunfailures"]  # 测试相关依赖
PRIME_REQUIRES = ["pyext"]                # PRIME 算法所需的扩展依赖
GEO_REQUIRES = ["mathruler", "torchvision", "qwen_vl_utils"]  # 几何/视觉推理相关依赖
GPU_REQUIRES = ["liger-kernel", "flash-attn"]                  # GPU 加速依赖（Flash Attention 等）
MATH_REQUIRES = ["math-verify"]           # 数学答案验证库，用于数学推理任务的奖励计算
VLLM_REQUIRES = ["tensordict>=0.8.0,<=0.10.0,!=0.9.0", "vllm>=0.13.0"]  # CacheEOPD V1 connector 依赖
SGLANG_REQUIRES = [
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "sglang[srt,openai]==0.5.6",          # SGLang 推理引擎（替代 vLLM 的选项）
    "torch==2.9.1",                        # SGLang 需要固定的 PyTorch 版本
]
TRL_REQUIRES = ["trl<=0.9.6"]            # HuggingFace TRL（Transformer Reinforcement Learning）库
MCORE_REQUIRES = ["mbridge"]             # Megatron-Core 桥接库，用于 Megatron-LM 并行策略
TRANSFERQUEUE_REQUIRES = ["TransferQueue==0.1.5.dev3"]  # TransferQueue 数据传输队列（异步训练用）

# 可选依赖字典，键为安装时的 extras 名称
extras_require = {
    "test": TEST_REQUIRES,           # pip install verl[test]
    "prime": PRIME_REQUIRES,         # pip install verl[prime]
    "geo": GEO_REQUIRES,             # pip install verl[geo]
    "gpu": GPU_REQUIRES,             # pip install verl[gpu]
    "math": MATH_REQUIRES,           # pip install verl[math]
    "vllm": VLLM_REQUIRES,           # pip install verl[vllm] - 使用 vLLM 作为推理引擎
    "sglang": SGLANG_REQUIRES,       # pip install verl[sglang] - 使用 SGLang 作为推理引擎
    "trl": TRL_REQUIRES,             # pip install verl[trl]
    "mcore": MCORE_REQUIRES,         # pip install verl[mcore] - 使用 Megatron-Core 并行
    "transferqueue": TRANSFERQUEUE_REQUIRES,  # pip install verl[transferqueue]
}


# 读取 README.md 作为 PyPI 上的长描述
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# 调用 setuptools 的 setup 函数完成包的安装配置
setup(
    name="verl",                     # 包名（在 PyPI 上注册为 verl）
    version=__version__,             # 版本号（从 version 文件动态读取）
    package_dir={"": "."},           # 包的根目录为当前目录
    packages=find_packages(where="."), # 自动发现所有 Python 包
    url="https://github.com/volcengine/verl",  # 项目仓库地址
    license="Apache 2.0",            # 开源许可证
    author="Bytedance - Seed - MLSys",  # 作者（字节跳动 Seed 团队）
    author_email="zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk",
    description="verl: Volcano Engine Reinforcement Learning for LLM",  # 简短描述
    install_requires=install_requires,  # 核心依赖
    extras_require=extras_require,      # 可选依赖
    package_data={
        "": ["version/*"],             # 包含版本文件
        "verl": ["trainer/config/*.yaml"],  # 包含训练器 YAML 配置文件
    },
    include_package_data=True,          # 启用包数据包含
    long_description=long_description,  # 长描述（README.md 内容）
    long_description_content_type="text/markdown",  # 长描述格式为 Markdown
)
