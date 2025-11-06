import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def _compute_single_image_stats(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = img.astype(np.float32) / 255.0
    h, w, c = img.shape
    n_pixels = h * w
    sum_ = img.sum(axis=(0, 1))
    sum_sq = (img ** 2).sum(axis=(0, 1))
    return sum_, sum_sq, n_pixels


def compute_mean_std(df, img_col='img_path', sample_size=None, parallel=-1):
    """
    Calculate channel-wise mean and std over all dataset images.
    
    Args:
        df: pd.DataFrame containing image paths
        img_col: dataframe key containing image paths
        sample_size: only use n images
        parallel: amount processes (-1 = all CPU-cores, 0 = sequential)
    
    Returns:
        (mean, std): np.array with 3 values each (B, G, R)
    """
    image_paths = np.unique(df[img_col])
    if sample_size is not None:
        image_paths = image_paths[:sample_size]

    # --- Sequential ---
    if not parallel:
        n_pixels = 0
        channel_sum = np.zeros(3)
        channel_sum_sq = np.zeros(3)
        for path in tqdm(image_paths, desc="Calculate Mean/Std", unit="img"):
            res = _compute_single_image_stats(path)
            if res is None:
                continue
            sum_, sum_sq, n_pix = res
            n_pixels += n_pix
            channel_sum += sum_
            channel_sum_sq += sum_sq
        mean = channel_sum / n_pixels
        std = np.sqrt(channel_sum_sq / n_pixels - mean ** 2)
        return mean, std

    # --- Parallel ---
    n_workers = multiprocessing.cpu_count() if parallel == -1 else parallel
    n_pixels = 0
    channel_sum = np.zeros(3)
    channel_sum_sq = np.zeros(3)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_compute_single_image_stats, path) for path in image_paths]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Parallel ({n_workers} cores)"):
            res = f.result()
            if res is None:
                continue
            sum_, sum_sq, n_pix = res
            n_pixels += n_pix
            channel_sum += sum_
            channel_sum_sq += sum_sq

    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sum_sq / n_pixels - mean ** 2)
    return mean, std
