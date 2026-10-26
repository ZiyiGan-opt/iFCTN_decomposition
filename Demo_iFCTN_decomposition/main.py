# import os
# import time
# import numpy as np
# import matplotlib.pyplot as plt

# from utils.data_loader import load_data
# from utils.metrics import compute_quality
# from methods.ifctn_tc import iFCTN_TC_gpu


# DATA_PATH = os.path.join("data", "cave_face_ms_256.mat")
# MISSING_RATIO = 0.6
# RHO = 0.1

# RGB_BANDS = (28, 18, 8)  # 对应论文里 Band 29/19/9
# GAMMA = 0.8


# def rgb_from_msi(X, bands=RGB_BANDS, gamma=GAMMA):
#     X = np.asarray(X, dtype=np.float32)
#     max_band = X.shape[2] - 1
#     bands = [min(max(b, 0), max_band) for b in bands]

#     rgb = np.stack(
#         [X[:, :, bands[0]], X[:, :, bands[1]], X[:, :, bands[2]]],
#         axis=-1,
#     )
#     rgb = np.clip(rgb, 0.0, 1.0)
#     return rgb ** gamma


# def visualize_rgb(observed_tensor, recon_tensor, observed_psnr, recon_psnr):
#     panels = [
#         (f"Observed\n{observed_psnr:.3f} dB", rgb_from_msi(observed_tensor)),
#         (f"iFCTN\n{recon_psnr:.3f} dB", rgb_from_msi(recon_tensor)),
#     ]

#     fig, axes = plt.subplots(1, 2, figsize=(8, 4))

#     for ax, (title, image) in zip(axes, panels):
#         ax.imshow(image)
#         ax.set_title(title)
#         ax.axis("off")

#     plt.tight_layout()
#     plt.show()


# def main():
#     print(f"Loading data from {DATA_PATH}...")
#     X = load_data(DATA_PATH)
#     if X is None:
#         return

#     X = X.astype(np.float32)
#     X = X / X.max()
#     I1, I2, I3 = X.shape

#     np.random.seed(42)
#     num_fibers = I1 * I2
#     num_missing = int(np.round(num_fibers * MISSING_RATIO))
#     flat_indices = np.random.choice(num_fibers, num_missing, replace=False)

#     mask_2d = np.ones((I1, I2), dtype=bool)
#     mask_2d.reshape(-1)[flat_indices] = False
#     Omega = np.repeat(mask_2d[:, :, np.newaxis], I3, axis=2)

#     F = np.zeros_like(X)
#     F[Omega] = X[Omega]

#     observed_psnr, observed_ssim, observed_rse = compute_quality(X, F)
#     print(
#         f"Observed | PSNR: {observed_psnr:.3f} dB | "
#         f"SSIM: {observed_ssim:.4f} | RSE: {observed_rse:.4f}"
#     )

#     opts = {
#         "tol": 1e-5,
#         "maxit": 1000,
#         "rho": RHO,
#         "R": [6, 6, 6],
#     }

#     print("Running iFCTN...")
#     t0 = time.time()
#     X_recon, _ = iFCTN_TC_gpu(F, Omega, opts)
#     cost_time = time.time() - t0

#     X_recon = np.asarray(X_recon, dtype=np.float32)
#     X_recon = np.clip(X_recon, 0.0, 1.0)

#     recon_psnr, recon_ssim, recon_rse = compute_quality(X, X_recon)
#     print(
#         f"iFCTN   | PSNR: {recon_psnr:.3f} dB | "
#         f"SSIM: {recon_ssim:.4f} | RSE: {recon_rse:.4f} | Time: {cost_time:.2f}s"
#     )

#     visualize_rgb(F, X_recon, observed_psnr, recon_psnr)


# if __name__ == "__main__":
#     main()

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from utils.data_loader import load_data
from utils.metrics import compute_quality
from methods.ifctn_tc import iFCTN_TC_gpu


DATA_PATH = os.path.join("data", "cave_face_ms_256.mat")
MISSING_RATIO = 0.6
RHO = 0.1

RGB_BANDS = (28, 18, 8)  # 对应论文里 Band 29/19/9
GAMMA = 0.8


def rgb_from_msi(X, bands=RGB_BANDS, gamma=GAMMA):
    X = np.asarray(X, dtype=np.float32)
    max_band = X.shape[2] - 1
    bands = [min(max(b, 0), max_band) for b in bands]

    rgb = np.stack(
        [X[:, :, bands[0]], X[:, :, bands[1]], X[:, :, bands[2]]],
        axis=-1,
    )
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb ** gamma


def visualize_rgb(observed_tensor, recon_tensor, observed_psnr, observed_time, recon_psnr, recon_time):
    panels = [
        (f"Observed\nPSNR={observed_psnr:.3f} dB\nTime={observed_time:.2f}s", rgb_from_msi(observed_tensor)),
        (f"iFCTN\nPSNR={recon_psnr:.3f} dB\nTime={recon_time:.2f}s", rgb_from_msi(recon_tensor)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    print(f"Loading data from {DATA_PATH}...")
    X = load_data(DATA_PATH)
    if X is None:
        return

    X = X.astype(np.float32)
    X = X / X.max()
    I1, I2, I3 = X.shape

    observed_t0 = time.time()

    np.random.seed(2)
    num_fibers = I1 * I2
    num_missing = int(np.round(num_fibers * MISSING_RATIO))
    flat_indices = np.random.choice(num_fibers, num_missing, replace=False)

    mask_2d = np.ones((I1, I2), dtype=bool)
    mask_2d.reshape(-1)[flat_indices] = False
    Omega = np.repeat(mask_2d[:, :, np.newaxis], I3, axis=2)

    F = np.zeros_like(X)
    F[Omega] = X[Omega]

    observed_time = time.time() - observed_t0

    observed_psnr, observed_ssim, observed_rse = compute_quality(X, F)
    print(
        f"Observed | PSNR: {observed_psnr:.3f} dB | "
        f"SSIM: {observed_ssim:.4f} | RSE: {observed_rse:.4f} | Time: {observed_time:.2f}s"
    )

    opts = {
        "tol": 1e-5,
        "maxit": 1000,
        "rho": RHO,
        "R": [6, 6, 6],
    }

    print("Running iFCTN...")
    t0 = time.time()
    X_recon, _ = iFCTN_TC_gpu(F, Omega, opts)
    recon_time = time.time() - t0

    X_recon = np.asarray(X_recon, dtype=np.float32)
    X_recon = np.clip(X_recon, 0.0, 1.0)

    recon_psnr, recon_ssim, recon_rse = compute_quality(X, X_recon)
    print(
        f"iFCTN   | PSNR: {recon_psnr:.3f} dB | "
        f"SSIM: {recon_ssim:.4f} | RSE: {recon_rse:.4f} | Time: {recon_time:.2f}s"
    )

    visualize_rgb(F, X_recon, observed_psnr, observed_time, recon_psnr, recon_time)


if __name__ == "__main__":
    main()