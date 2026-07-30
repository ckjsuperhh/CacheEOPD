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
FlowRL Ray Trainer that extends RayPPOTrainer with FlowRL-specific components.

FlowRL Ray 训练器模块。

该模块扩展了 verl 框架的 RayPPOTrainer，增加了 FlowRL 特有的组件。
主要区别在于使用了 FlowRL 特有的优势函数估计方法（advantage estimator），
通过流平衡原理来匹配奖励分布，从而改进策略优化过程。
"""
"""
FlowRL Ray 分布式训练器。
继承自 RayPPOTrainer，使用 FlowRL 特有的优势估计方法（flowrl_adv_estimator），
通过流平衡（Flow Balance）匹配奖励分布进行策略优化。
"""

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class RayFlowRLTrainer(RayPPOTrainer):
    """
    FlowRL trainer that uses the FlowRL advantage estimator.
    The main difference is in the advantage estimation which is registered
    as 'flowrl' in flowrl_adv_estimator.py

    FlowRL 训练器，使用 FlowRL 特有的优势函数估计方法。
    继承自 RayPPOTrainer，主要区别在于优势估计部分，
    使用 'flowrl' 注册的优势估计器（定义在 flowrl_adv_estimator.py 中）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
