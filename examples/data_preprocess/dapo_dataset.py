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
DAPO-Math-17k 数据集预处理脚本。

文件作用：
    下载并预处理 open-r1/DAPO-Math-17k-Processed 数据集，
    转换为 verl 框架所需的 parquet 格式。
    这是 DAPO（Decoupled Alignment Policy Optimization）方法使用的数学训练数据。

命令行参数：
    --local_dir / --local_save_dir: 保存路径
    --local_dataset_path: 本地原始数据集路径（离线模式）
    --hdfs_dir: HDFS 保存目录（可选）

数据来源：open-r1/DAPO-Math-17k-Processed
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
    parser.add_argument(
        "--local_save_dir", default="/root/verl/data/dapo_math", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()

    data_source = "open-r1/DAPO-Math-17k-Processed"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    
    # Load subset 'en', split 'train'
    dataset = datasets.load_dataset(
        data_source,
        "en",
        split="train"
    )

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # The source dataset uses 'prompt' for the problem statement
            question = example["prompt"]

            question = question + " " + instruction_following

            # The source dataset uses 'solution' for the answer, and it is already the ground truth
            solution = example["solution"]
            
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    # Process only train dataset
    train_dataset = dataset.map(function=make_map_fn("train"), with_indices=True)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    hdfs_dir = args.hdfs_dir
    
    # Ensure local directory exists
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    
    # Save one example as JSON for reference
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
