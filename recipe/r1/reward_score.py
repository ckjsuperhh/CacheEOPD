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
R1 奖励评分路由模块。
根据数据来源（data_source）自动分发到对应的评分函数：
AIME/CNMO 使用数学评分，GPQA 使用多选题评分，LiveCodeBench 使用代码评分。
"""


def reward_func(data_source, solution_str, ground_truth, extra_info=None):
    if data_source in ["Maxwell-Jia/AIME_2024", "opencompass/cnmo2024_en", "opencompass/cnmo2024_zh"]:
        from recipe.r1.tasks import math_reward

        return math_reward.compute_score(solution_str, ground_truth)
    elif data_source == "Idavidrein/gpqa":
        from recipe.r1.tasks import gpqa

        return gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["livecodebench/code_generation_lite", "livecodebench/code_generation"]:
        from recipe.r1.tasks import livecodebench

        return livecodebench.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError
