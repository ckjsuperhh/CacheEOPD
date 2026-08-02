# CacheEOPD 阶段性实验汇报

## 实验转向：从一直带着做，到教学与考试交替

最初的方案是在 student 训练和 rollout 中始终注入 teacher KV，等价于老师全程带着
学生完成每一步。这个方案造成了明显的训练-评测不一致：训练时 student 依赖 teacher KV，
独立考试时却只能使用自己的 KV。

在 student-only 评测下，早期 always-fused 实验的代表性结果为：base student=`181/300`，
plain step100/200/300=`189/177/182`，always-fused step100/200/300=`185/179/170`。
尤其 step300 的 fused student 明显退化。这说明“始终让老师带着”可能让 student 学会了
依赖外部 cache，而不是稳定地形成自己的 rollout 能力；此前 teacher-assisted 的
`192/300` 也不能当作独立 student 能力。

这个失败引导了后续设计：教学时可以让老师手把手纠正，但必须反复让学生独立完成任务，
再接受下一轮教学。于是我们提出 mixed 和 anneal 两种交替策略：mixed 固定概率地在
micro-batch 间切换 teacher KV 与 student KV，anneal 则逐步减少 teacher KV，让训练后期
更接近“考试”。

## 目标

CacheEOPD 将 C2C 的 teacher KV 投影融合到 EOPD 的 student rollout 中。本阶段进一步
把 projector 冻结，只训练 student，并比较三种训练方式：

- plain：始终使用 student 自己的 KV；
- mixed：每个 micro-batch 以 `p=0.5` 注入 teacher KV；
- anneal：teacher KV 注入概率从 1.0 退火到 0.0。

训练配置为 Qwen3-0.6B student、Qwen3-4B teacher、1368 条有效 teacher trajectory，前
64 条作为 holdout，300 steps、`grad_accum=8`、`lr=1e-5`，每 50 步保存。最终评测固定为
GSM8K-COT 300 题、greedy、batch size 8、`max-new-tokens=512`。

## 关键评测协议

所有最终 accuracy 都是 student-only：评测时不加载 teacher、不加载 projector、不注入
teacher KV，只加载 student checkpoint 并使用其原生 KV cache。这一点排除了“评测依赖
teacher KV”造成的虚高。

## 主要结果

base student 为 `181/300`。EOPD dense baseline 和 prior-v7 projector mixed 结果为：

| step | EOPD dense | mixed seed41717 | mixed seed41718 | mixed seed41719 |
|---:|---:|---:|---:|---:|
| 50 | 188 | 181 | 188 | 191 |
| 100 | 175 | 180 | 184 | 190 |
| 150 | 183 | 186 | 186 | 189 |
| 200 | 185 | **192** | 189 | 186 |
| 250 | 190 | 190 | **199** | 195 |
| 300 | 186 | 188 | 198 | 195 |

三个 mixed seed 的均值为：step150 `187.00 ± 1.41`、step200 `189.00 ± 2.45`、step250
`194.67 ± 3.68`、step300 `193.67 ± 4.19`（总体标准差）。anneal 的最佳结果为
`189/300`。

为测试 projector 的通用性，使用相同 mixed 配置替换官方 C2C fuser：

| projector | step150 | step200 | step250 | step300 |
|---|---:|---:|---:|---:|
| prior-v7 projector | 186 | **192** | 190 | 188 |
| official fuser | 187 | 186 | **194** | 176 |

## 结论

实验支持 mixed 方法“可行且有局部收益”：它在不同 seed 和两种 projector 上都能在某个
checkpoint 区间超过 base，并且较优区间集中在 step200--300。官方 fuser 在 step250
达到 `194/300`，说明收益不完全依赖于重新训练 projector。

但实验不支持“KV 注入稳定提高最终上限”或“step200 必然最好”：不同 seed 的峰值分别在
step200、step250 和 step250/300，官方 fuser 在 step300 还明显退化。因此后续大规模实验
应保留密集 checkpoint，并围绕 mixed 注入概率、注入时机和 early stopping 做消融。

完整过程、逐题 paired gain/loss、踩坑和远端产物见 [PROGRESS.md](PROGRESS.md)。核心训练
入口为 `train_student_distill.py`，独立评测入口为 `eval_student_batch.py`。
