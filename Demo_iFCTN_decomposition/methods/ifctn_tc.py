import torch
import numpy as np

def solve_TR_subproblem_gpu(B, S, A1, B1, C, D, rho, device='cuda'):
    """
    通用 TR 子问题求解器 (GPU 版)

    参数约定:
    B: 数据张量, 必须调整为 (Batch, Dim_A, Dim_B) 的形状
    S: 权重矩阵 (Dim_A, Dim_B)
    A1: 左侧因子矩阵 (Rank_A, Dim_A)
    B1: 右侧因子矩阵 (Rank_B, Dim_B)
    C:  更新 X1 时的辅助矩阵 (Rank_B, Batch) -> 注意这里传入时需转置
    D:  更新 X2 时的辅助矩阵 (Rank_A, Batch) -> 注意这里传入时需转置
    rho: 标量
    """

    # 确保输入是 Tensor
    def to_dev(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device) if not isinstance(x, torch.Tensor) else x

    S, A1, B1, C, D = map(to_dev, [S, A1, B1, C, D])

    # B 必须已经在外部 permute 好，并转为 (Batch, Dim_A, Dim_B)
    # 例如：如果 Batch 是 I1，则 shape 为 (I1, I2, I3)
    if isinstance(B, np.ndarray):
        B_cube = torch.tensor(B, dtype=torch.float32, device=device)
    else:
        B_cube = B

    # 获取维度信息
    N, Dim_A, Dim_B = B_cube.shape
    Rank_A = A1.shape[0]
    Rank_B = B1.shape[0]

    # 初始化 X2 (使用 D 作为初始猜测)
    X2_curr = D

    # ================= Update X1 =================
    # 目标求解 X1 (Rank_B, Batch)

    # V = A1' * X2  => (Dim_A, Rank_A) * (Rank_A, Batch) -> (Dim_A, Batch)
    V = torch.matmul(A1.T, X2_curr)

    # T = S .* V (广播)
    # S: (Dim_A, Dim_B), V: (Dim_A, Batch)
    # 我们需要 T 为 (Batch, Dim_A, Dim_B) 以匹配 B_cube
    # Einstein Summation: n=Batch, a=Dim_A, b=Dim_B
    # S(a,b) * V(a,n) -> T(n,a,b)
    # 【注意】这里必须用字母，不能用数字
    T = torch.einsum('ab, an -> nab', S, V)

    # Lambda = sum(T.^2, 1)' => 在 Matlab 中是 sum over Dim_A (rows)
    # PyTorch T is (N, A, B). Sum over dim 1 (A). Result (N, B).
    Lambda = torch.sum(T ** 2, dim=1)

    # z = sum(T .* B_cube, 1)' => Sum over dim 1 (A). Result (N, B).
    z = torch.sum(T * B_cube, dim=1)

    # 构建 M (左侧矩阵)
    # M_k = B1 * diag(Lambda_k) * B1' + rho * I
    # B1: (Rank_B, Dim_B). Lambda: (N, Dim_B).
    # 广播计算: (N, Rank_B, Dim_B) = B1(1, R, B) * Lambda(N, 1, B)
    B1_weighted = B1.unsqueeze(0) * Lambda.unsqueeze(1)

    # M: (N, Rank_B, Rank_B)
    M = torch.matmul(B1_weighted, B1.T.unsqueeze(0))

    # + rho * I
    M += rho * torch.eye(Rank_B, device=device).unsqueeze(0)

    # 构建 RHS (右侧向量)
    # rhs = B1 * z + rho * C
    # B1 (R_B, B) * z^T (B, N) -> (R_B, N). 转置后 (N, R_B)
    # 这里直接用 Batch 乘法: z (N, B) @ B1.T (B, R_B) -> (N, R_B)
    rhs = torch.matmul(z, B1.T) + rho * C.T  # C 是 (R_B, N) -> C.T 是 (N, R_B)

    # Cholesky Solve
    # M: (N, R, R), rhs: (N, R). Result X1_T: (N, R)
    L = torch.linalg.cholesky(M)
    X1_T = torch.cholesky_solve(rhs.unsqueeze(2), L).squeeze(2)
    X1 = X1_T.T  # 转回 (Rank_B, Batch)

    # ================= Update X2 =================
    # 目标求解 X2 (Rank_A, Batch)

    # U = B1' * X1 => (Dim_B, Rank_B) * (Rank_B, Batch) -> (Dim_B, Batch)
    U = torch.matmul(B1.T, X1)

    # T = S .* U
    # S: (Dim_A, Dim_B), U: (Dim_B, Batch)
    # Einstein Summation: S(a,b) * U(b,n) -> T(n,a,b)
    T = torch.einsum('ab, bn -> nab', S, U)

    # Gamma = sum(T.^2, 2) => Sum over Dim_B (dim 2). Result (N, A)
    Gamma = torch.sum(T ** 2, dim=2)

    # zeta = sum(T .* B_cube, 2) => (N, A)
    zeta = torch.sum(T * B_cube, dim=2)

    # N = A1 * diag(Gamma) * A1' + rho * I
    # A1 (R_A, A). Gamma (N, A).
    A1_weighted = A1.unsqueeze(0) * Gamma.unsqueeze(1)  # (N, R_A, A)
    N_mat = torch.matmul(A1_weighted, A1.T.unsqueeze(0))  # (N, R_A, R_A)
    N_mat += rho * torch.eye(Rank_A, device=device).unsqueeze(0)

    # rhs = A1 * zeta + rho * D
    # A1 (R_A, A) * zeta (N, A)^T -> (R_A, N)
    # Batch: zeta (N, A) @ A1.T (A, R_A) -> (N, R_A)
    rhs = torch.matmul(zeta, A1.T) + rho * D.T

    # Solve
    L_N = torch.linalg.cholesky(N_mat)
    X2_T = torch.cholesky_solve(rhs.unsqueeze(2), L_N).squeeze(2)
    X2 = X2_T.T  # (Rank_A, Batch)

    return X1, X2


def iFCTN_TC_gpu(F, Omega, opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    tol = opts.get('tol', 1e-4)
    maxit = opts.get('maxit', 100)
    rho = opts.get('rho', 0.1)
    # R 对应 [R12, R13, R23]
    R_ranks = opts.get('R', [10, 10, 10])

    F_gpu = torch.as_tensor(F, dtype=torch.float32, device=device)
    Omega_gpu = torch.as_tensor(Omega, dtype=torch.bool, device=device)

    X = F_gpu.clone()
    X[~Omega_gpu] = 0

    I1, I2, I3 = F.shape
    R12, R13, R23 = R_ranks

    # 初始化因子
    A1 = torch.rand(R12, I2, device=device)
    A2 = torch.rand(R23, I2, device=device)
    B1 = torch.rand(R13, I3, device=device)
    B2 = torch.rand(R23, I3, device=device)
    C1 = torch.rand(R12, I1, device=device)
    C2 = torch.rand(R13, I1, device=device)

    loss_history = []

    print(f"Start FCTN-TC | Size: {F.shape} | Ranks: {R_ranks}")

    for k in range(maxit):
        Xold = X.clone()

        # ================= Update G1 (C1, C2) =================
        # 目标: 更新 C1(R12, I1) 和 C2(R13, I1)
        # Batch 维度: I1。数据矩阵: (I2, I3)。
        # 输入 B 应为 (I1, I2, I3) -> 不需要 permute，直接连续化
        B_in = X.contiguous()

        # S = A2' * B2 -> (I2, I3)
        S = torch.matmul(A2.T, B2)

        # 调用 Solver:
        # Batch=I1, Dim_A=I2, Dim_B=I3
        # 输入对应: A1(R12, I2), B1(R13, I3), C->C1, D->C2
        # 注意：C1, C2 传入 solver 时对应 C, D 参数
        C2, C1 = solve_TR_subproblem_gpu(B_in, S, A1, B1, C2, C1, rho, device=device)

        # ================= Update G2 (A2, A1) =================
        # 目标: 更新 A2(R23, I2) 和 A1(R12, I2)
        # Batch 维度: I2。数据矩阵: (I1, I3)。
        # 我们构建 S = C2' * B1 -> (I1, R13) * (R13, I3) -> (I1, I3)
        # 所以 Dim_A=I1, Dim_B=I3
        # 输入 B 应为 (I2, I1, I3) -> 需要 permute(1, 0, 2)
        B_in = X.permute(1, 0, 2).contiguous()

        S = torch.matmul(C2.T, B1)

        # 调用 Solver:
        # Batch=I2, Dim_A=I1, Dim_B=I3
        # 输入对应: C1(R12, I1), B2(R23, I3)
        # 这里的 A2, A1 对应返回的 X1, X2
        A2, A1 = solve_TR_subproblem_gpu(B_in, S, C1, B2, A2, A1, rho, device=device)

        # ================= Update G3 (B2, B1) =================
        # 目标: 更新 B2(R23, I3) 和 B1(R13, I3)
        # Batch 维度: I3。
        # S = C1' * A1 -> (I1, R12) * (R12, I2) -> (I1, I2)
        # 所以 Dim_A=I1, Dim_B=I2
        # 输入 B 应为 (I3, I1, I2) -> 需要 permute(2, 0, 1)
        B_in = X.permute(2, 0, 1).contiguous()

        S = torch.matmul(C1.T, A1)

        # 调用 Solver:
        # Batch=I3, Dim_A=I1, Dim_B=I2
        # 输入对应: C2(R13, I1), A2(R23, I2)
        B2, B1 = solve_TR_subproblem_gpu(B_in, S, C2, A2, B2, B1, rho, device=device)

        # ================= Update X =================
        # 重构 Tensor: sum_{r} C1(r1,i) C2(r3,i) A1(r1,j) A2(r2,j) B1(r3,k) B2(r2,k)
        # 索引: i(I1), j(I2), k(I3)
        # 秩索引: a(R12), b(R23), c(R13)

        X_recon = torch.einsum('ai, ci, aj, bj, ck, bk -> ijk', C1, C2, A1, A2, B1, B2)

        X = (X_recon + rho * Xold) / (1.0 + rho)
        X[Omega_gpu] = F_gpu[Omega_gpu]

        # Check convergence
        diff = torch.norm(X - Xold)
        norm_x = torch.norm(Xold)
        rse = (diff / norm_x).item()
        loss_history.append(rse)

        if (k + 1) % 5 == 0 or k == 0:
            print(f"Iter {k + 1:3d} | RSE: {rse:.6f}")

        if rse < tol:
            print("Converged.")
            break

    return X.cpu().numpy(), loss_history
