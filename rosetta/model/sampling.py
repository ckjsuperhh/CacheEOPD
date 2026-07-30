"""
采样模块 (Sampling Module)
==========================

本文件提供 LLM 文本生成过程中使用的 **token 采样（解码策略）** 工具函数。

在 C2C (Cache-to-Cache) 框架中，该模块主要用于 **第二阶段推理 (Stage 2)**：
  - Stage 1: Sharer 模型处理 prompt，产生 KV-Cache
  - Stage 2: Receiver 模型基于融合后的 KV-Cache 自回归生成回答；
    每一步生成时，模型输出 logits，然后调用本模块的 `sample_token` 函数
    将 logits 转换为下一个 token ID。

核心函数:
  - sample_token: 支持 temperature、top-k、top-p (nucleus) 三种解码策略的
    token 采样函数，可灵活组合使用，也支持贪心解码 (greedy decoding)。

与其他模块的关系:
  - 被 rosetta/model/ 下的推理 (inference) 代码调用，作为 LLM 自回归生成的
    解码步骤。
  - 本身不依赖 Projector、Fuser 等 C2C 特有模块，是通用的采样工具。
"""

import torch
import torch.nn.functional as F
from typing import Union

def sample_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1) -> Union[int, torch.Tensor]:
    """
    从 logits 中采样生成 token，支持 temperature、top-k、top-p 三种解码策略的组合。
    Sample a token from logits using temperature, top-p, and top-k sampling.

    解码策略说明:
      - temperature (温度): 控制 softmax 前的缩放系数。
          temperature > 1 → 分布更平滑（更随机）
          temperature < 1 → 分布更尖锐（更确定）
          temperature → 0 → 退化为贪心解码 (argmax)
      - top_k: 只保留概率最高的 k 个 token，其余置零。-1 表示不过滤。
      - top_p (nucleus sampling): 保留累计概率刚好达到 p 的最小 token 集合，
        动态确定保留的 token 数量，比固定 top_k 更灵活。

    参数 (Args):
        logits (torch.Tensor): 模型输出的 token logits，形状为 [vocab_size]（单条）
            或 [batch_size, vocab_size]（批量）。
        temperature (float): 采样温度，>0。值越大采样越随机，默认为 1.0（标准 softmax）。
        top_p (float): nucleus 采样的概率阈值，范围 (0, 1]。默认 1.0 表示不做 top-p 过滤。
        top_k (int): top-k 采样的 k 值。默认 -1 表示不做 top-k 过滤。

    返回 (Returns):
        Union[int, torch.Tensor]:
            - 输入为单条 (1D) 时返回 int（采样的 token ID）
            - 输入为批量 (2D) 时返回 torch.Tensor（每个样本对应的 token ID）

    关键算法流程:
        1. 用 temperature 对 logits 缩放后做 softmax，得到概率分布
        2. 若 top_k != -1，只保留概率最高的 k 个 token，重新归一化
        3. 若 top_p < 1.0，按概率降序排列，保留累计概率 ≤ top_p 的 token，重新归一化
        4. 从过滤后的分布中用 multinomial 采样得到 token
    """
    # ---- 输入校验 (Input validation) ----
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    
    if logits.dim() not in [1, 2]:
        raise ValueError("logits must have shape [vocab_size] or [batch_size, vocab_size]")
        
    # 统一处理：将 1D 输入 [vocab_size] 扩展为 2D [1, vocab_size]，方便后续批量处理
    # Handle single dimension input
    is_single_input = logits.dim() == 1
    if is_single_input:
        logits = logits.unsqueeze(0)
    
    batch_size = logits.shape[0]
    
    # ---- 贪心解码 (Greedy decoding) ----
    # 当 temperature 为 0 或极小时，直接使用 argmax 取概率最大的 token，
    # 此时不引入任何随机性，结果完全确定。
    # For greedy sampling (temperature=0), just return argmax
    if temperature == 0 or temperature <= 1e-5:
        tokens = torch.argmax(logits, dim=-1)
        return tokens.item() if is_single_input else tokens
    
    # ---- 带温度的 softmax (Temperature-scaled softmax) ----
    # 将 logits 除以 temperature 后做 softmax：
    #   probs = softmax(logits / temperature)
    # temperature 越大，分布越平滑（高熵）；越小，分布越尖锐（低熵）
    # Convert to probabilities
    probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
    
    # ---- Top-k 过滤 (Top-k filtering) ----
    # 只保留概率最大的 k 个 token，其余 token 的概率置零，然后重新归一化。
    # 这确保了采样只在 top_k 个候选中进行。
    # Apply top-k filtering first (if specified)
    if top_k != -1:
        # 取出概率最高的 k 个值及其索引
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
        
        # 创建布尔掩码，仅在 top_k 个位置为 True
        # Create a mask to zero out non-top-k probabilities
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(-1, top_k_indices, True)
        
        # 将不在 top_k 中的 token 概率置零
        # Zero out non-top-k probabilities
        probs = probs * mask.float()
        
        # 重新归一化，使概率总和为 1
        # Renormalize probabilities
        probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # ---- Top-p (Nucleus) 采样 (Top-p / nucleus sampling) ----
    # 按概率降序排列，累加概率直到超过 top_p 阈值，
    # 只保留累计概率刚好达到 top_p 的最小 token 集合。
    # 与 top_k 不同，top_p 动态决定保留多少个 token，更灵活。
    # Apply top-p (nucleus) sampling
    if top_p < 1.0:
        # 将概率按降序排列
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        
        # 计算累计概率（前缀和）
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 创建掩码：累计概率 ≤ top_p 的位置保留，超出部分被截断
        # 注意：由于先累加再比较，被截断位置的前一个 token 会被保留，
        # 这保证了累计概率至少达到 top_p。
        # Create a mask for probabilities to keep
        # Values above top_p threshold are masked out
        mask = cumulative_probs <= top_p
        
        # 始终保留概率最大的那个 token（即使它单独已超过 top_p），
        # 确保至少有一个 token 可被采样。
        # Always keep at least one token
        mask[:, 0] = True
        
        # 将不在 nucleus 集合中的 token 概率置零
        # Zero out masked positions to exclude them from sampling
        sorted_probs = sorted_probs * mask.float()
        
        # 重新归一化
        # Renormalize probabilities
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        
        # 从过滤后的分布中采样一个 token（multinomial 按行独立采样）
        # Sample from the filtered distribution
        sampled_indices = torch.multinomial(sorted_probs, num_samples=1)
        
        # 将采样到的排序后索引映射回原始词汇表索引
        # Map back to original vocabulary indices
        tokens = torch.gather(sorted_indices, dim=-1, index=sampled_indices)
        tokens = tokens.squeeze(-1)  # 去掉 num_samples=1 引入的多余维度 / Remove sample dimension
    else:
        # ---- 无 top-p 过滤时直接从概率分布中采样 ----
        # Direct sampling if no top-p filtering
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    # 如果输入是单条 (1D)，返回 Python int；否则返回 batch 维度的 tensor
    return tokens.item() if is_single_input else tokens
