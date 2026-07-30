# ============================================================
# 脚本用途：评估模型生成的数学推理响应
# ============================================================
# 使用 verl 的 main_eval 模块，加载自定义评分函数（compute_score_data_source），
# 对 AIME24/AIME25 测试集上的生成结果进行评分。
# ============================================================
#!/usr/bin/env bash

# Evaluation
python3 -m verl.trainer.main_eval \
    data.path=$HOME/data/gen/qwen_8b_gen_test.parquet \
    custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
    custom_reward_function.name=compute_score_data_source
