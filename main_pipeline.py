from data_loader import load_radar_data, add_awgn
from spectrogram_utils import get_optimal_range_bin
from denoising_algorithms import algo_EBD
from visualization import plot_spectrogram
import numpy as np

def run_pipeline(file_path, snr_list=[-15, 0, 10]):
    """
    Runs the entire pipeline for a given radar data file.

    Args:
        file_path (str): Path to the radar data file.
        snr_list (list, optional): List of SNRs to generate noisy spectrograms for. Defaults to [-15, 0, 10].

    Returns:
        None
    """
    data = load_radar_data(file_path)
    for snr in snr_list:
        noisy_data = add_awgn(data, snr)
        optimal_range = get_optimal_range_bin(noisy_data, r_values=[1])
        denoised, f, t = algo_EBD(noisy_data, optimal_range, Th=3)
        plot_spectrogram(f, t, np.abs(denoised), title=f"Denoised Spectrogram (SNR={snr} dB)")