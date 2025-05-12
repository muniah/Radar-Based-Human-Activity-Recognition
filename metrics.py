import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

def compute_global_metrics(ref, test):
    """
    Compute global metrics between reference and test images (2D arrays).

    Args:
        ref (2D np.array): Reference image (noise-free).
        test (2D np.array): Test image (noisy or denoised).

    Returns:
        metrics (dict): Dictionary containing MSE, MAE, PSNR, Correlation, SSIM.
    """
    
    ref_flat = ref.flatten()
    test_flat = test.flatten()
    mse = np.mean((ref_flat - test_flat) ** 2)
    mae = np.mean(np.abs(ref_flat - test_flat))
    psnr = 10.0 * np.log10(np.max(ref)**2 / mse) if mse != 0 else np.inf
    corr, _ = pearsonr(ref_flat, test_flat)
    data_range = np.max(ref) - np.min(ref)
    ssim_val = ssim(ref, test, data_range=data_range)
    return {"MSE": mse, "MAE": mae, "PSNR": psnr, "Correlation": corr, "SSIM": ssim_val}