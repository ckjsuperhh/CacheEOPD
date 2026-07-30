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
Preprocess the dataset to parquet format.

R1 评估数据预处理模块。

本模块将多个评估基准的数据集从 HuggingFace 下载并转换为 verl 框架所需的 parquet 格式。
支持的数据集包括：
  - AIME 2024: 美国数学邀请赛（数学推理）
  - GPQA Diamond: 研究生级科学问答（多选题）
  - CNMO 2024: 中国数学奥林匹克（中英文数学）
  - LiveCodeBench: 代码生成基准（编程能力）

每个样本包含：data_source, prompt, ability, reward_model（含 ground_truth）, extra_info
"""

import argparse
import os
from functools import partial

from datasets import concatenate_datasets, load_dataset

from verl.utils.hdfs_io import copy, makedirs


def example_map_fn(example, idx, process_fn, data_source, ability, split):
    """
    通用的数据映射函数，将原始样本转换为 verl 框架标准格式。

    参数:
        example: 原始数据样本
        idx: 样本索引
        process_fn: 数据预处理函数，返回 (question, solution) 元组
        data_source: 数据来源标识符
        ability: 能力标签（如 "Math", "Code", "English"）
        split: 数据集划分（如 "test"）
    返回:
        标准化的数据字典
    """
    question, solution = process_fn(example)
    data = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx},
    }
    return data


def build_aime2024_dataset():
    """构建 AIME 2024 评估数据集（美国数学邀请赛，数学推理能力）。"""
    def process_aime2024(example):
        return example["Problem"], str(example["Answer"])

    data_source = "Maxwell-Jia/AIME_2024"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    map_fn = partial(
        example_map_fn, process_fn=process_aime2024, data_source=data_source, ability="English", split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_gpqa_dimond_dataset():
    """
    构建 GPQA Diamond 评估数据集（研究生级科学多选题）。

    将正确答案随机插入到四个选项中，生成标准的多选题格式。
    """
    import random

    GPQA_QUERY_TEMPLATE = (
        "Answer the following multiple choice question. The last line of your response should be of the following "
        "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before "
        "answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    )

    def process_gpqa_diamond(example):
        choices = [example["Incorrect Answer 1"], example["Incorrect Answer 2"], example["Incorrect Answer 3"]]
        random.shuffle(choices)
        gold_index = random.randint(0, 3)
        choices.insert(gold_index, example["Correct Answer"])
        query_prompt = GPQA_QUERY_TEMPLATE.format(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=example["Question"]
        )
        gold_choice = "ABCD"[gold_index]
        return query_prompt, gold_choice

    data_source = "Idavidrein/gpqa"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset = load_dataset(data_source, "gpqa_diamond", split="train")
    map_fn = partial(
        example_map_fn, process_fn=process_gpqa_diamond, data_source=data_source, ability="Math", split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_cnmo2024_dataset():
    """构建 CNMO 2024 评估数据集（中国数学奥林匹克，包含中英文两个版本并合并）。"""
    def process_cnmo2024(example):
        return example["question"], example["answer"]

    data_source = "opencompass/LiveMathBench"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset_en = load_dataset(data_source, "v202412_CNMO_en", split="test")
    map_fn_en = partial(
        example_map_fn, process_fn=process_cnmo2024, data_source="opencompass/cnmo2024_en", ability="Math", split="test"
    )
    dataset_en = dataset_en.map(map_fn_en, with_indices=True, remove_columns=dataset_en.column_names)

    dataset_zh = load_dataset(data_source, "v202412_CNMO_cn", split="test")
    map_fn_zh = partial(
        example_map_fn, process_fn=process_cnmo2024, data_source="opencompass/cnmo2024_zh", ability="Math", split="test"
    )
    dataset_zh = dataset_zh.map(map_fn_zh, with_indices=True, remove_columns=dataset_zh.column_names)

    dataset = concatenate_datasets([dataset_en, dataset_zh])
    return dataset


def build_livecodebench_dataset():
    """
    构建 LiveCodeBench 代码生成评估数据集。

    使用 2024年8月至2025年1月期间的竞赛编程题目。
    将公开和私有测试用例合并并压缩存储，用于代码正确性验证。
    """
    import base64
    import json
    import pickle
    import zlib

    def process_livecodebench(example):
        # Construct Query Prompt
        # From https://github.com/LiveCodeBench/LiveCodeBench/blob/998c52d394b836f15fff3b9a29866191108ff81b/lcb_runner/prompts/code_generation.py#L140
        query_prompt = (
            f"You will be given a question (problem specification) and will generate a correct Python program "
            f"that matches the specification and passes all tests.\n\nQuestion: {example['question_content']}\n\n"
        )
        if example["starter_code"]:
            query_prompt += (
                f"You will use the following starter code to write the solution to the problem and enclose your "
                f"code within delimiters.\n```python\n{example['starter_code']}\n```"
            )
        else:
            query_prompt += (
                "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test "
                "on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
                "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
                "```python\n# YOUR CODE HERE\n```"
            )

        # Construct test cases
        public_test_cases = json.loads(example["public_test_cases"])
        try:
            private_test_cases = json.loads(example["private_test_cases"])
        except Exception as e:
            print(f"Error loading private test cases: {e}")
            private_test_cases = json.loads(
                pickle.loads(zlib.decompress(base64.b64decode(example["private_test_cases"].encode("utf-8"))))
            )
        full_test_cases = public_test_cases + private_test_cases

        metadata = json.loads(example["metadata"])
        test_cases = {
            "inputs": [t["input"] for t in full_test_cases],
            "outputs": [t["output"] for t in full_test_cases],
            "fn_name": metadata.get("func_name", None),
        }
        text_cases_compressed = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(test_cases)))).decode("utf-8")
        return query_prompt, text_cases_compressed

    data_source = "livecodebench/code_generation_lite"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="test")
    # R1 Evaluation use LiveCodeBench 24.08-25.01
    dataset = dataset.filter(lambda line: "2024-08-00T00:00:00" <= line["contest_date"] < "2025-01-00T00:00:00")
    map_fn = partial(
        example_map_fn, process_fn=process_livecodebench, data_source=data_source, ability="Code", split="test"
    )

    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names, num_proc=8)
    return dataset


TASK2DATA = {
    "aime2024": build_aime2024_dataset,
    "gpqa_diamond": build_gpqa_dimond_dataset,
    "cnmo2024": build_cnmo2024_dataset,
    "livecodebench": build_livecodebench_dataset,
}
SUPPORTED_TASKS = TASK2DATA.keys()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/r1")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--tasks", default="all")

    args = parser.parse_args()

    if args.tasks.lower() == "all":
        args.tasks = SUPPORTED_TASKS
    else:
        args.tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        for task in args.tasks:
            if task not in SUPPORTED_TASKS:
                raise NotImplementedError(f"{task} has not been supported.")

    datasets = []
    for task in args.tasks:
        datasets.append(TASK2DATA[task]())
    test_dataset = concatenate_datasets(datasets)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
