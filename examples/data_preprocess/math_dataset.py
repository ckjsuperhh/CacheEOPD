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
"""
MATH 数据集预处理脚本。

文件作用：
    下载并预处理 MATH-lighteval 数据集（数学推理），转换为 verl 框架所需的 parquet 格式。
    这是 EOPD/OPD 训练的核心训练数据来源（论文 §5.1 使用 MATH 17k 训练数据）。

命令行参数：
    --local_dir: （已弃用）本地保存目录，请改用 --local_save_dir
    --local_save_dir: 预处理数据集保存路径，默认 /root/verl/data/math
    --local_dataset_path: 本地原始数据集路径（离线模式时使用）
    --hdfs_dir: HDFS 保存目录（可选，用于集群环境）

输出格式 (parquet)：
    - data_source: 数据集来源标识
    - prompt: [{"role": "user", "content": 问题 + 指令}]
    - ability: "math"
    - reward_model: {"style": "rule", "ground_truth": 从 \\boxed{} 中提取的答案}
    - extra_info: {"split": train/test, "index": 序号}

执行流程：
    1. 加载 MATH-lighteval 数据集（train + test）
    2. 为每道题拼接指令 "Let's think step by step and output the final answer within \\boxed{}."
    3. 从 solution 中用 last_boxed_only_string 提取标准答案
    4. 保存为 train.parquet 和 test.parquet

依赖：
    - datasets: HuggingFace 数据集库
    - verl.utils.reward_score.math_reward: 数学答案提取工具

在 EOPD 中的角色：
    训练数据预处理的第一步，生成训练集供 on_policy_it.sh 使用。
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    """
    从 LaTeX 格式的解题过程中提取最终答案。
    使用 last_boxed_only_string 定位最后一个 \\boxed{}，
    再用 remove_boxed 去除 \\boxed 标记，得到纯答案文本。
    """
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="/root/verl/data/math", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(
            local_dataset_path,
        )
    else:
        dataset = datasets.load_dataset(
            data_source,
        )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            question = question + " " + instruction_following

            answer = example.pop("solution")
            solution = extract_solution(answer)
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    # Save one example as JSON for reference
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    example = test_dataset[0]
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
