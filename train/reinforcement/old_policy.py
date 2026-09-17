"""Extracted training components; function bodies preserve the legacy implementation."""

def return_decay(step, decay_type):
    """
    Old model的EMA decay系数（DiffusionNFT Line 169-189）

    decay_type:
      0: 始终为0（不使用old model）
      1: 线性增长到0.5
      2: 前75步为0，之后增长到0.999（推荐）
    """
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.001, 0.999
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)
