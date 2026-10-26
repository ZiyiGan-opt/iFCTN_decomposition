import torch


def kr(A, B):
    """
    实现 MATLAB 的 kr(A, B) - Khatri-Rao Product (列向 Kronecker 积)
    相当于: for j in range(r): col_j = kron(A[:,j], B[:,j])

    参数:
        A: (p, r) Tensor
        B: (q, r) Tensor
    返回:
        (p*q, r) Tensor
    """
    p, r1 = A.shape
    q, r2 = B.shape

    if r1 != r2:
        raise ValueError(f'The numbers of columns must be the same. A:{A.shape}, B:{B.shape}')

    # === 核心逻辑 (利用广播代替循环) ===
    # 1. A 变型为 (p, 1, r)
    # 2. B 变型为 (1, q, r)
    # 3. 乘法广播得到 (p, q, r)
    # 4. Reshape 合并前两维 -> (p*q, r)

    # 注意：Matlab 的 kron(a, b) 是 b 的元素在内存中变化得快（block structure）
    # PyTorch 的 reshape 默认行优先，A.unsqueeze(1) * B.unsqueeze(0) 会生成
    # 类似于 A[i]*B[j] 的结构。
    # 最终 reshape(p*q, r) 会让第二个维度的索引(对应B)变化得快，符合 Kronecker 定义。

    res = A.unsqueeze(1) * B.unsqueeze(0)  # Shape: (p, q, r)
    return res.reshape(p * q, r1)  # Shape: (p*q, r)


def kron(A, B):
    """
    实现 MATLAB 的 kron(A, B) - Standard Kronecker Product

    参数:
        A: (p, n) Tensor
        B: (q, m) Tensor
    返回:
        (p*q, n*m) Tensor
    """
    p, n = A.shape
    q, m = B.shape

    # === 核心逻辑 ===
    # A (p, 1, n, 1) * B (1, q, 1, m) -> (p, q, n, m)
    # Reshape -> (p*q, n*m)
    res = A.unsqueeze(1).unsqueeze(3) * B.unsqueeze(0).unsqueeze(2)
    return res.reshape(p * q, n * m)