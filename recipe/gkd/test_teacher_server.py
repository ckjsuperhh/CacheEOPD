# Copyright 2025 Individual Contributor: furunding
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
教师模型服务端连通性测试脚本。

通过 TeacherClient 向本地教师服务端发送随机 token，
验证返回的 top-k logprobs 和 indices 的形状和数据类型是否正确。
用于调试和验证教师服务端部署是否正常工作。
"""

import random

import torch
from teacher import TeacherClient


def main():
    teacher_client = TeacherClient("127.0.0.1", 15555)
    tokens = [[random.randint(1, 99999) for _ in range(100)] for _ in range(2)]
    tokens[0][40] = 128858
    _, teacher_topk_logps, teacher_topk_indices = teacher_client.submit(tokens).result()
    assert all(logps.shape == (100, 256) for logps in teacher_topk_logps)
    assert all(logps.dtype == torch.float32 for logps in teacher_topk_logps)
    assert all(indices.shape == (100, 256) for indices in teacher_topk_indices)
    assert all(indices.dtype == torch.int32 for indices in teacher_topk_indices)
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    main()
