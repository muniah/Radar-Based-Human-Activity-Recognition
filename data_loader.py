import scipy.io
import numpy as np

def load_radar_data(file_path):
    """
    Load radar data from a .mat file.

    Parameters
    ----------
    file_path : str
        The path to the .mat file containing the radar data.

    Returns
    -------
    numpy.ndarray
        The received time domain signal data extracted from the file.
        Shape: (num_chirps, num_samples).
    """

    data = scipy.io.loadmat(file_path)
    return data['received_time_domain_signal']

def add_awgn(data, snr_db):
    """
    Add white Gaussian noise to 'data' to achieve the desired SNR in dB.
    
    Parameters
    ----------
    data : numpy array
        The data to be noised. Can be real or complex.
    snr_db : float
        The desired signal-to-noise ratio in decibels.
    
    Returns
    -------
    noisy_data : numpy array
        The noisy data, with the same shape as 'data'.
    """
    sig_power = np.mean(np.abs(data)**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_var = sig_power / snr_linear
    noise = np.sqrt(noise_var) * np.random.randn(*data.shape)
    return data + noise