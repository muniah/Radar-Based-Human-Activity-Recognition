import numpy as np
from scipy.signal import stft
from scipy.fft import fft, fftshift
from scipy.stats import entropy

def compute_full_fft(signal_data):
    """
    Compute the magnitude of the Fast Fourier Transform (FFT) along the sample axis.

    Parameters:
    - signal_data (ndarray): 2D numpy array of shape (num_chirps, num_samples)
                             representing the input signal data.

    Returns:
    - ndarray: 2D numpy array of the same shape as input, containing the magnitude
               of the FFT computed along each row of the input signal data.
    """

    return np.abs(fft(signal_data, axis=1))

def compute_stft(signal_data, fs=1e3):
    """
    Compute Short-Time Fourier Transform (STFT) of the given signal.

    Parameters:
        signal_data (ndarray): 2D numpy array of shape (num_chirps, num_samples)
                              representing the input signal data.
        fs (float): Sampling frequency.

    Returns:
        tuple: (f, t, Zxx, Zxx_magnitude)

        - f (ndarray): Frequency bins.
        - t (ndarray): Time bins.
        - Zxx (ndarray): Complex STFT result.
        - Zxx_magnitude (ndarray): Magnitude of STFT in dB.
    """
    f, t, Zxx = stft(signal_data, fs=fs, return_onesided=False)
    return f, t, Zxx, 10 * np.log10(np.abs(Zxx) + 1e-12)

def compute_entropy(Zxx, tau_index):
    """
    Calculate the Shannon entropy of the magnitude spectrum at a specific time index.

    Parameters:
        Zxx (ndarray): Complex STFT result.
        tau_index (int): Time index for which the entropy is computed.

    Returns:
        float: Shannon entropy of the magnitude spectrum.
    """

    mag = np.abs(Zxx[:, tau_index])
    P = mag / np.sum(mag)
    P = P[P > 0]
    return -np.sum(P * np.log2(P))

def compute_average_entropy(Zxx, t):
    """
    Compute the average Shannon entropy of the magnitude spectrum across all time bins.

    Parameters:
        Zxx (ndarray): Complex STFT result.
        t (ndarray): Time bins.

    Returns:
        float: Average Shannon entropy of the magnitude spectrum across all time bins.
    """
    return np.mean([compute_entropy(Zxx, i) for i in range(len(t))])

def get_optimal_range_bin(data_matrix, r_values):
    """
    Find the optimal range-bin for a given set of range intervals.

    Parameters:
        data_matrix (ndarray): 2D numpy array of shape (num_chirps, num_samples)
                              representing the input signal data.
        r_values (list): List of range intervals for defining r_q.

    Returns:
        tuple: Optimal range-bin interval (start, end) indices.
    """
    
    num_chirps, num_samples = data_matrix.shape
    fft_mag = compute_full_fft(data_matrix)
    P_max = [np.argmax(fft_mag[i, :]) for i in range(num_chirps)]
    idx_max = np.argmax(np.bincount(P_max))
    min_entropy = float('inf')
    best_r_opt = None
    for r in r_values:
        r_q = np.arange(max(0, idx_max - r), min(num_samples, idx_max + r))
        signal_range = data_matrix[:, r_q]
        for chirp in signal_range:
            f, t, Zxx, _ = compute_stft(chirp)
            avg_entropy = compute_average_entropy(Zxx, t)
            if avg_entropy < min_entropy:
                min_entropy = avg_entropy
                best_r_opt = r_q
    return (best_r_opt[0], best_r_opt[-1])