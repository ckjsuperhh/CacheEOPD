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
MATH-500 评测数据集预处理脚本。

文件作用：
    下载并预处理 HuggingFaceH4/MATH-500 数据集（MATH 测试集的 500 题子集），
    转换为 verl 框架所需的 parquet 格式。该数据集是 EOPD 评测的 6 个 benchmark 之一。

命令行参数：
    --local_dir: （已弃用）本地保存目录
    --local_save_dir: 保存路径，默认 /root/verl/data/math500
    --local_dataset_path: 本地原始数据集路径（离线模式）
    --hdfs_dir: HDFS 保存目录（可选）

数据来源：HuggingFaceH4/MATH-500（仅有 test split，500 道题）

输出格式 (parquet)：
    - data_source: "HuggingFaceH4/MATH-500"
    - prompt: [{"role": "user", "content": 问题 + \\boxed{} 指令}]
    - reward_model: {"style": "rule", "ground_truth": 标准答案字符串}

在 EOPD 评测中的角色：
    评测流程第一步——预处理 MATH-500 测试数据，
    供 generate_offline_vllm.py 生成模型回复，再由 score_avg_pass_at_k.py 评分。
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="/root/verl/data/math500", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "HuggingFaceH4/MATH-500"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(
            local_dataset_path,
        )
    else:
        dataset = datasets.load_dataset(
            data_source,
        )

    # This dataset only has a test split
    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            question = question + " " + instruction_following

            solution = example.pop("answer")
            
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    hdfs_dir = args.hdfs_dir
    
    os.makedirs(local_dir, exist_ok=True)

    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    
    # Save one example as JSON for reference
    example = test_dataset[0]
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=2)
        
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

