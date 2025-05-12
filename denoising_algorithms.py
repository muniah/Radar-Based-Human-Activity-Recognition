import numpy as np
from scipy.signal import stft
from scipy.fft import fft

def algo_EBD(signal_data, optimal_range, Th=3, fs_slow=1000.0):
    """
    Entropy-based denoising algorithm: Denoise using adaptive STFT thresholding.
    Instead of plotting directly, returns the thresholded STFT in complex form,
    plus frequency/time arrays so you can plot a spectrogram that matches your
    'raw/noisy' style.

    Parameters:
      signal_data: shape (num_chirps, num_samples)
      optimal_range: (start, end) for the selected range bins
      Th: threshold factor
      fs_slow: slow-time sampling rate (1000 Hz since 1 ms per chirp)

    Returns:
      denoised_stft_accum: complex 2D array (freq_bins x time_frames),
                           after thresholding & averaging across the selected bins
      f: frequency bins (in Hz)
      t: time bins (in s)
    """
    range_fft = fft(signal_data.T, axis=1)
    start, end = optimal_range
    denoised = None
    count = 0
    for rb in range(start, end):
        slow_time = range_fft[:, rb]
        f, t, Zxx = stft(slow_time, fs=fs_slow, nperseg=128, noverlap=64, return_onesided=False)
        Zxx_denoised = np.where(np.abs(Zxx) > Th * np.mean(np.abs(Zxx)), Zxx, 0)
        denoised = Zxx_denoised if denoised is None else denoised + Zxx_denoised
        count += 1
    return denoised / max(count, 1), f, t