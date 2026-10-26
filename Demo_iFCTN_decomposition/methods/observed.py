import numpy as np

def run_observed(F, Omega, opts=None):
    """
    最简单的基准：直接返回观测值（缺失处补0）
    """
    # 这里 F 已经是缺失处为0的数据了
    return F, {}