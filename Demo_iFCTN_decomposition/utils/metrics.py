# import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as psnr_func
# from skimage.metrics import structural_similarity as ssim_func
#
#
# def compute_quality(X_true, X_recon, unsampled_mask=None):
#     """计算 PSNR, SSIM, RSE"""
#     # 归一化到 [0, 1] 避免数值范围问题
#     if X_true.max() > 1:
#         X_true = X_true / X_true.max()
#     if X_recon.max() > 1:
#         X_recon = X_recon / X_recon.max()
#
#     rse = np.linalg.norm(X_recon.ravel() - X_true.ravel()) / np.linalg.norm(X_true.ravel())
#
#     n_bands = X_true.shape[2]
#     psnr_list = []
#     ssim_list = []
#
#     for k in range(n_bands):
#         p = psnr_func(X_true[:, :, k], X_recon[:, :, k], data_range=1.0)
#         s = ssim_func(X_true[:, :, k], X_recon[:, :, k], data_range=1.0)
#         psnr_list.append(p)
#         ssim_list.append(s)
#
#     return np.mean(psnr_list), np.mean(ssim_list), rse

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func


def compute_quality(X_true, X_recon):
    """
    计算 PSNR, SSIM, RSE。
    完全复刻 MATLAB quality_ybz 的逻辑：
    1. 如果维度 > 3，将后续维度展平。
    2. 对每一张切片 (Slice) 计算 PSNR/SSIM 然后取平均。
    """
    # 确保输入是 NumPy 数组
    if hasattr(X_true, 'cpu'): X_true = X_true.cpu().numpy()
    if hasattr(X_recon, 'cpu'): X_recon = X_recon.cpu().numpy()

    # 1. RSE (Relative Standard Error) - 全局计算
    numerator = np.linalg.norm(X_true - X_recon)
    denominator = np.linalg.norm(X_true)
    rse = numerator / denominator

    # 2. Reshape 处理 (复刻 MATLAB: if length(Nway)>3 ...)
    shape = X_true.shape
    if len(shape) > 3:
        # 保持前两维 (I1, I2)，合并后面所有维
        # 例如 (10, 20, 5, 5) -> (10, 20, 25)
        new_dim3 = np.prod(shape[2:])
        X_true_flat = X_true.reshape(shape[0], shape[1], new_dim3)
        X_recon_flat = X_recon.reshape(shape[0], shape[1], new_dim3)
    else:
        X_true_flat = X_true
        X_recon_flat = X_recon

    # 3. 逐切片计算 PSNR 和 SSIM
    N_slices = X_true_flat.shape[2]
    psnr_list = []
    ssim_list = []

    # 数据范围推断 (0-1 还是 0-255)
    # 既然你之前的代码做了归一化，这里 data_range=1.0 比较安全
    # 如果数据是 0-255，请改为 255
    data_range = X_true.max() - X_true.min()
    if data_range == 0: data_range = 1.0

    for k in range(N_slices):
        img_true = X_true_flat[:, :, k]
        img_rec = X_recon_flat[:, :, k]

        # --- PSNR ---
        p = psnr_func(img_true, img_rec, data_range=data_range)
        psnr_list.append(p)

        # --- SSIM ---
        try:
            # win_size 必须为奇数，且小于图像最小边长
            min_side = min(img_true.shape)
            win_size = min(7, min_side)
            if win_size % 2 == 0: win_size -= 1
            if win_size < 3: win_size = 1  # 极小情况

            s = ssim_func(img_true, img_rec, data_range=data_range, win_size=win_size)
        except Exception:
            # 如果实在太小算不了 SSIM，就填 NaN 或 0
            s = 0
        ssim_list.append(s)

    return np.mean(psnr_list), np.mean(ssim_list), rse