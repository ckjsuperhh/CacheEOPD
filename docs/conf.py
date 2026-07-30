"""
conf.py - Sphinx 文档构建配置文件。

本文件是 verl/EOPD 项目在线文档（Read the Docs）的 Sphinx 配置。
定义了项目名称、作者、版权信息、文档扩展（MyST Parser、autodoc 等）、
HTML 主题（sphinx_rtd_theme）以及源文件格式等选项。
文档地址：https://verl.readthedocs.io/
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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- 项目信息配置 -----------------------------------------------------

project = "verl"                                              # 项目名称
copyright = "2024 ByteDance Seed Foundation MLSys Team"        # 版权声明
author = "Guangming Sheng, Chi Zhang, Yanghua Peng, Haibin Lin"  # 作者列表


# -- 通用配置 ---------------------------------------------------
# 主目录树文档（入口文件为 index.rst）
master_doc = "index"

# Sphinx 扩展模块列表
extensions = [
    "myst_parser",               # MyST 解析器，支持在 .rst 中嵌入 Markdown
    "sphinx.ext.autodoc",        # 自动从 Python 代码生成文档
    "sphinx.ext.autosummary",    # 自动生成模块摘要页面
    "sphinx.ext.autosectionlabel", # 自动为章节生成交叉引用标签
    "sphinx.ext.napoleon",       # 支持 Google/NumPy 风格的 docstring
    "sphinx.ext.viewcode",       # 在文档中添加"查看源码"链接
]

# MyST-Parser 设置（Markdown 扩展语法）
myst_enable_extensions = [
    "dollarmath",  # 启用 $...$ 和 $$...$$ 数学公式语法
    "amsmath",  # 启用 amsmath 数学环境
]

# 使用 Google 风格 docstring（而非 NumPy 风格）
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# 源文件后缀映射
source_suffix = {
    ".rst": "restructuredtext",  # reStructuredText 格式
    ".md": "markdown",           # Markdown 格式
}

# 模板目录（相对于本文件所在目录）
templates_path = ["_templates"]

# 文档语言设为英文
language = "en"

# 排除不需要处理的文件/目录模式
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- HTML 输出选项 -------------------------------------------------

# 使用 Read the Docs 官方主题
html_theme = "sphinx_rtd_theme"

# 自定义静态文件目录（CSS/JS 等）
html_static_path = ["_static"]

# 引入自定义 JavaScript 文件
html_js_files = [
    "js/runllm-widget.js",      # RunLLM 交互组件
    "js/resizable-sidebar.js",  # 可调整大小的侧边栏
]

# 引入自定义 CSS 文件（全宽布局等）
html_css_files = [
    "custom.css",
]

# 排除文档目录中的 README 文件（避免重复内容）
exclude_patterns += ["README.md", "README_vllm0.7.md"]

# 抑制重复引用和 MyST 相关的警告
suppress_warnings = ["ref.duplicate", "ref.myst"]
