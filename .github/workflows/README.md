<!--
  workflows/README.md - CI Workflow 开发指南
  中文摘要：本文件说明了如何为 verl 项目添加新的 GitHub Actions CI Workflow。
  介绍了两种 Runner 选项：固定 Runner（指定 GPU 机器）和 Vemlp 动态 Runner（弹性启动）。
  提供了使用 Vemlp 动态 Runner 的模板，包括 setup/cleanup job 的创建方式、
  环境变量配置和模型/数据集预下载路径等信息。
-->
### Adding a New Workflow

When adding a new workflow for continuous integration (CI), you have two runner options: a fixed runner or a machine from the vemlp.

- **Fixed Runner**: To use a fixed runner, specify it in your workflow using the `runs-on` keyword, like `runs-on: [L20x8]`. 
- **Vemlp Runner**: Opting for a Vemlp machine allows you to launch tasks elastically. 

Here is a template to assist you. This template is designed for using Vemlp machines. Currently, for each workflow, you need to create a `setup` and a `cleanup` job. When using this template, the main parts you need to modify are the `IMAGE` environment variable and the specific `job steps`.

```yaml
name: Your Default Workflow

on:
  push:
    branches:
      - main
      - v0.*
  pull_request:
    branches:
      - main
      - v0.*
    paths:
      - "**/*.py"
      - ".github/workflows/template.yml"

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}

permissions:
  contents: read

env:
  IMAGE: "your vemlp image" # e.g. "verl-ci-cn-beijing.cr.volces.com/verlai/verl:sgl056.latest"
  DYNAMIC_RUNNER_URL: "https://sd10g3clalm04ug7alq90.apigateway-cn-beijing.volceapi.com/runner" # public veFaas api

jobs:
  setup:
    if: github.repository_owner == 'volcengine'
    runs-on: ubuntu-latest
    outputs:
      runner-label: ${{ steps.create-runner.outputs.runner-label }}
      task-id: ${{ steps.create-runner.outputs.task-id }}
    steps:
      - uses: actions/checkout@v4
      - id: create-runner
        uses: volcengine/vemlp-github-runner@v1 
        with:
          mode: "create"
          faas-url: "${{ env.DYNAMIC_RUNNER_URL }}"
          image: "${{ env.DEFAULT_IMAGE }}"

  your_job:
    needs: setup
    runs-on: ["${{ needs.setup.outputs.runner-label || 'default-runner' }}"]
    steps:
      xxxx # your jobs

  cleanup:
    runs-on: ubuntu-latest
    needs: [setup, your_job]
    if: always()
    steps:
      - id: destroy-runner
        uses: volcengine/vemlp-github-runner@v1
        with:
          mode: "destroy"
          faas-url: "${{ env.DYNAMIC_RUNNER_URL }}"
          task-id: "${{ needs.setup.outputs.task-id }}"
```

### Model and Dataset
To avoid CI relies on network, we pre-download dataset on a NFS on the CI machine. The path for models are \${HOME}/models and the path for dataset is \${HOME}/models/hf_data.