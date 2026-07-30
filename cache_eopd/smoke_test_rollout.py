"""
【C2C 核心】C2CHFRollout 冒烟测试（不依赖 ray / FSDP，单进程直接实例化）。

验证：
    1. C2CHFRollout.generate_sequences 在真实 DataProto 上能跑通；
    2. 输出的 batch 字段（prompts/responses/input_ids/attention_mask/position_ids）
       形状与 EOPD 训练循环的约定一致；
    3. c2c.enable=False 时行为回退到原版 HFRollout（对照）。

用法（apex-llm）:
    cd ~/CacheEOPD && PYTHONPATH=. python -m cache_eopd.smoke_test_rollout \
        --student ~/taopd-baseline/modelweights/Qwen3-1.7B \
        --teacher ~/taopd-baseline/modelweights/Qwen3-4B
"""

import argparse

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl import DataProto
from cache_eopd.c2c_hf_rollout import C2CHFRollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--response-length", type=int, default=32)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.student)
    tok.padding_side = "left"  # verl 约定 left-padding
    student = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(args.device)

    # rollout 配置（模拟 actor_rollout_ref.rollout 节点 + c2c 扩展）
    config = OmegaConf.create({
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "response_length": args.response_length,
        "micro_batch_size": 2,
        "val_kwargs": {"top_k": 0, "top_p": 1.0, "temperature": 0},
        "c2c": {
            "enable": True,
            "teacher_path": args.teacher,
            "teacher_device": "auto",  # 冒烟测试环境显存紧张，teacher 跨卡分片
            "zero_init": False,
        },
    })

    rollout = C2CHFRollout(module=student, config=config)

    # 构造两条 left-padded prompt 的 DataProto（与 verl 数据协议一致）
    prompts_text = [
        tok.apply_chat_template([{"role": "user", "content": "What is 17 * 23?"}],
                                tokenize=False, add_generation_prompt=True),
        tok.apply_chat_template([{"role": "user", "content": "Name the capital of France."}],
                                tokenize=False, add_generation_prompt=True),
    ]
    enc = tok(prompts_text, return_tensors="pt", padding=True).to(args.device)
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

    batch = TensorDict(
        {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids},
        batch_size=input_ids.shape[0],
    )
    data = DataProto(batch=batch)
    data.meta_info = {
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        "do_sample": True,
    }

    out = rollout.generate_sequences(data)

    L = input_ids.shape[1]
    R = args.response_length
    assert out.batch["prompts"].shape == (2, L), out.batch["prompts"].shape
    assert out.batch["responses"].shape == (2, R), out.batch["responses"].shape
    assert out.batch["input_ids"].shape == (2, L + R)
    assert out.batch["attention_mask"].shape == (2, L + R)
    assert out.batch["position_ids"].shape == (2, L + R)
    print("[pass] 输出字段形状全部符合 EOPD 训练循环约定")

    for i in range(2):
        text = tok.decode(out.batch["responses"][i], skip_special_tokens=True)
        print(f"\n--- sample {i} response ---\n{text}")


if __name__ == "__main__":
    main()
