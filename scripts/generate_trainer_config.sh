#!/usr/bin/env bash
# =============================================================================
# 训练器配置文件自动生成脚本
#
# 文件作用：
#   将 verl/trainer/config/ 下的 Hydra 层级配置（ppo_trainer.yaml 等）
#   展平（flatten）为单一 YAML 文件，方便用户参考所有可用配置项。
#   生成的文件仅供阅读参考，不会被训练流程直接使用。
#
# 执行流程：
#   1. 遍历 CONFIG_SPECS 中定义的每个配置规格
#   2. 调用 scripts/print_cfg.py 将 Hydra 配置展平为单文件
#   3. 添加文件头注释，写入 verl/trainer/config/_generated_*.yaml
#   4. 用 git diff 检查生成文件是否与仓库中的版本一致
#      - 如果不一致，说明配置有变更但生成文件未更新，退出报错
#
# 依赖脚本：
#   - scripts/print_cfg.py: Hydra 配置展平工具
#
# 生成的文件：
#   - verl/trainer/config/_generated_ppo_trainer.yaml
#   - verl/trainer/config/_generated_ppo_megatron_trainer.yaml
# =============================================================================
set -euox pipefail


# 定义配置规格列表，格式为 "配置名:输出文件名:额外参数"
# - ppo_trainer: 标准 PPO 训练器配置
# - ppo_megatron_trainer: 使用 Megatron 后端的 PPO 训练器配置
CONFIG_SPECS=(
    "ppo_trainer:_generated_ppo_trainer.yaml:"
    "ppo_megatron_trainer:_generated_ppo_megatron_trainer.yaml:--config-name=ppo_megatron_trainer.yaml"
)

generate_config() {
    local config_name="$1"
    local output_file="$2"
    local config_arg="$3"
    
    local target_cfg="verl/trainer/config/${output_file}"
    local tmp_header=$(mktemp)
    local tmp_cfg=$(mktemp)
    
    echo "# This reference configration yaml is automatically generated via 'scripts/generate_trainer_config.sh'" > "$tmp_header"
    echo "# in which it invokes 'python3 scripts/print_cfg.py --cfg job ${config_arg}' to flatten the 'verl/trainer/config/${config_name}.yaml' config fields into a single file." >> "$tmp_header"
    echo "# Do not modify this file directly." >> "$tmp_header"
    echo "# The file is usually only for reference and never used." >> "$tmp_header"
    echo "" >> "$tmp_header"
    
    python3 scripts/print_cfg.py --cfg job ${config_arg} > "$tmp_cfg"
    
    cat "$tmp_header" > "$target_cfg"
    sed -n '/^actor_rollout_ref/,$p' "$tmp_cfg" >> "$target_cfg"
    
    rm "$tmp_cfg" "$tmp_header"
    
    echo "Generated: $target_cfg"
}

for spec in "${CONFIG_SPECS[@]}"; do
    IFS=':' read -r config_name output_file config_arg <<< "$spec"
    generate_config "$config_name" "$output_file" "$config_arg"
done

for spec in "${CONFIG_SPECS[@]}"; do
    IFS=':' read -r config_name output_file config_arg <<< "$spec"
    target_cfg="verl/trainer/config/${output_file}"
    if ! git diff --exit-code -- "$target_cfg" >/dev/null; then
        echo "✖ $target_cfg is out of date. Please regenerate via 'scripts/generate_trainer_config.sh' and commit the changes."
        exit 1
    fi
done

echo "All good"
exit 0
