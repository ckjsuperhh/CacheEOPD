"""
Core utilities for Cache-to-Cache (C2C) operations.

C2C（Cache-to-Cache）核心工具模块。

本模块提供了 C2C 框架中与"共享者（sharer）"索引和位掩码（bitmask）之间
相互转换的核心工具函数。在 C2C 框架中，多个 LLM 通过 KV-Cache 进行通信，
每个参与共享的模型被分配一个从 1 开始的索引编号，系统使用位掩码来高效地
表示"哪些共享者参与了当前的缓存共享操作"。

核心函数：
    - sharers_to_mask: 将共享者索引列表转换为位掩码
    - mask_to_sharers: 将位掩码还原为共享者索引列表
    - all_sharers_mask: 生成选中所有共享者的位掩码
    - format_sharer_mask: 将位掩码格式化为可读字符串

与其他模块的关系：
    - 被 rosetta 包中涉及 KV-Cache 共享策略选择的模块调用
    - 位掩码机制用于控制哪些模型的 KV-Cache 可以被投影/共享给目标模型
    - 在模型推理阶段，根据掩码决定从哪些"共享者"模型中获取缓存信息

位掩码编码规则（bitmask encoding convention）：
    - 第 i 个共享者（1-based）对应位掩码的第 (i-1) 位
    - mask = -1 表示"无投影（no projection）"
    - mask = 0  表示"自身投影（self projection）"
    - mask > 0  表示共享者位掩码，每一位代表一个共享者
"""

from typing import List


def sharers_to_mask(sharer_indices: List[int]) -> int:
    """
    Convert a list of sharer indices to a bitmask.
    将共享者索引列表转换为位掩码整数。

    在 C2C 框架中，每个参与 KV-Cache 共享的模型有一个从 1 开始的唯一索引。
    本函数将这些索引编码为一个位掩码，方便在系统中以单个整数传递共享配置。

    编码方式：索引 i 对应位掩码的第 (i-1) 位（从 0 开始的最低位）。
    例如：索引 1 -> 第 0 位，索引 3 -> 第 2 位。

    Args:
        sharer_indices: List of 1-based sharer indices (e.g., [1, 2, 3])
            共享者索引列表，索引从 1 开始（例如 [1, 2, 3]）

    Returns:
        Bitmask integer (e.g., [1, 2] -> 3, [1, 3] -> 5, [1, 2, 3] -> 7)
        位掩码整数

    Example:
        >>> sharers_to_mask([1])      # 001 = 1
        1
        >>> sharers_to_mask([2])      # 010 = 2
        2
        >>> sharers_to_mask([1, 2])   # 011 = 3
        3
        >>> sharers_to_mask([1, 3])   # 101 = 5
        5
    """
    # 初始化掩码为 0（所有位均为 0，表示没有选中任何共享者）
    mask = 0
    for idx in sharer_indices:
        # 使用位或运算将第 (idx-1) 位设为 1
        # 例如 idx=1 -> 1 << 0 = 1 (二进制 001)
        #      idx=3 -> 1 << 2 = 4 (二进制 100)
        mask |= (1 << (idx - 1))
    return mask


def mask_to_sharers(mask: int) -> List[int]:
    """
    Convert a bitmask to a list of sharer indices.
    将位掩码还原为共享者索引列表（sharers_to_mask 的逆操作）。

    通过逐位检查位掩码中哪些位为 1，还原出对应的共享者索引。
    这是 sharers_to_mask 的逆函数，两者构成一对编解码函数。

    Args:
        mask: Bitmask integer
            位掩码整数

    Returns:
        List of 1-based sharer indices
        从 1 开始的共享者索引列表；若 mask <= 0 则返回空列表

    Example:
        >>> mask_to_sharers(1)   # 001 -> [1]
        [1]
        >>> mask_to_sharers(3)   # 011 -> [1, 2]
        [1, 2]
        >>> mask_to_sharers(5)   # 101 -> [1, 3]
        [1, 3]
        >>> mask_to_sharers(7)   # 111 -> [1, 2, 3]
        [1, 2, 3]
    """
    # mask <= 0 包括两种特殊语义：-1 表示"无投影"，0 表示"自身投影"
    # 这两种情况都不涉及外部共享者，因此返回空列表
    if mask <= 0:
        return []
    sharers = []
    # idx 从 1 开始，因为共享者索引是 1-based 的
    idx = 1
    while mask:
        # 检查最低位是否为 1：若是，则当前 idx 对应的共享者被选中
        if mask & 1:
            sharers.append(idx)
        # 右移一位，检查下一个位
        mask >>= 1
        idx += 1
    return sharers


def all_sharers_mask(num_sharers: int) -> int:
    """
    Get bitmask that selects all sharers.
    生成一个选中所有共享者的位掩码。

    当系统需要"所有模型都参与 KV-Cache 共享"时，调用此函数获取全 1 的掩码。
    例如 3 个共享者 -> 二进制 111 -> 十进制 7。

    算法原理：(1 << n) - 1 可以快速生成低 n 位全为 1 的整数。
    例如 1 << 3 = 8 (1000)，8 - 1 = 7 (0111)。

    Args:
        num_sharers: Number of sharers
            共享者的总数量

    Returns:
        Bitmask with all bits set (e.g., 3 sharers -> 7 = 111)
        所有位均置 1 的位掩码整数
    """
    # (1 << num_sharers) 产生第 num_sharers 位为 1、其余为 0 的数
    # 减 1 后，低 num_sharers 位全部变为 1
    return (1 << num_sharers) - 1


def format_sharer_mask(mask: int) -> str:
    """
    Format a sharer mask as a human-readable string.
    将共享者位掩码格式化为人类可读的字符串，主要用于日志输出和调试。

    位掩码的特殊值约定（mask value convention）：
        - mask < 0 (通常为 -1)：表示"无投影"，即不使用任何外部共享者的 KV-Cache
        - mask == 0：表示"自身投影"，即只使用目标模型自己的 KV-Cache
        - mask > 0：正常的共享者位掩码，每一位对应一个被选中的共享者

    Args:
        mask: Bitmask integer (-1=no projection, 0=self projection, >0=sharer bitmask)
            位掩码整数（-1=无投影，0=自身投影，>0=共享者位掩码）

    Returns:
        Formatted string like "sharers [1, 2]" or "no projection"
        格式化后的可读字符串，例如 "sharers [1, 2]" 或 "no projection"
    """
    # 负数掩码：表示无投影模式（no projection），不使用任何外部共享缓存
    if mask < 0:
        return "no projection"
    # 零值掩码：表示自身投影模式（self projection），仅使用模型自身的缓存
    if mask == 0:
        return "self projection"
    # 正值掩码：解码为共享者索引列表并格式化输出
    sharers = mask_to_sharers(mask)
    return f"sharers {sharers}"
