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
基础配置类（BaseConfig）的 CPU 单元测试。

测试 BaseConfig 的 __getitem__ 方法行为，包括：
- 正常路径：访问已存在的属性
- 异常路径1：访问不存在的属性，应抛出 AttributeError
- 异常路径2：使用非字符串类型作为键，应抛出 TypeError
"""

import pytest

from verl.base_config import BaseConfig


@pytest.fixture
def base_config_mock():
    """测试夹具：创建一个带有测试属性的 BaseConfig 模拟实例。"""
    mock_config = BaseConfig()
    mock_config.test_attr = "test_value"
    return mock_config


def test_getitem_success(base_config_mock):
    """测试 __getitem__ 访问已存在的属性（正常路径）。"""
    assert base_config_mock["test_attr"] == "test_value"  # 验证可以正确获取已存在的属性值


def test_getitem_nonexistent_attribute(base_config_mock):
    """测试 __getitem__ 访问不存在的属性（异常路径1），应抛出 AttributeError。"""
    with pytest.raises(AttributeError):
        _ = base_config_mock["nonexistent_attr"]


def test_getitem_invalid_key_type(base_config_mock):
    """测试 __getitem__ 使用非法键类型（异常路径2），应抛出 TypeError。"""
    with pytest.raises(TypeError):
        _ = base_config_mock[123]  # type: ignore # 传入整数类型键，应触发 TypeError
