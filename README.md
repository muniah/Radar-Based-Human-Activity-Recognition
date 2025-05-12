# Radar-Based Human Activity Recognition (HAR)

This project provides a modular Python implementation of a radar signal processing pipeline that generates micro-Doppler spectrograms from time-domain radar returns. It includes denoising techniques, entropy-based optimal range-bin selection, and a CNN-based neural network model for spectrogram classification.

## Project Structure 

├── data_loader.py               # Load .mat radar data, add AWGN noise
├── spectrogram_utils.py         # STFT, FFT, entropy, range-bin selection
├── denoising_algorithms.py      # EBD and Adaptive Thresholding
├── visualization.py             # Spectrogram plotting functions
├── classifier.py                # PyTorch CNN model for classification
├── main_pipeline.py             # Example runner script

## Features

✅ Load radar returns from .mat files

✅ Add synthetic white Gaussian noise at varying SNR levels

✅ Compute spectrograms via STFT

✅ Select optimal range-bin using entropy minimization

✅ Denoise using:

Entropy-Based Denoising (EBD)

Adaptive Thresholding (ADTh)

✅ Visualize noisy and denoised spectrograms

✅ Classify spectrograms using a lightweight CNN model

## Project Highlights

* Reimplementation and benchmarking of three recent denoising techniques for micro-Doppler spectrograms:

  * Adaptive Preprocessing (APr)
  * Adaptive Thresholding (ATh)
  * Entropy-Based Denoising (EBD)

* Evaluation under varying levels of noise using both:

  * Conventional error-based metrics (MSE, RMSE, PSNR)
  * Perceptual metrics (SSIM)

* Insight into the limitations of error metrics in low-SNR scenarios, and the need for perceptually aligned evaluation metrics in training deep learning denoisers.

* Static Activity Recognition via Range–Angle (RA) feature maps:

  * Novel Quality Scoring algorithm using “lumps + subpeaks” for no-reference map evaluation.
  * Temporal Tracking algorithm for enforcing consistency in cluttered environments.
  * Grad-CAM visualizations for model interpretability.

* A simple PyTorch CNN model is provided in classifier.py for supervised classification of spectrograms (e.g., walking vs running).

## Datasets

* Dynamic Activity Dataset: 19,800 radar samples covering 11 activities across multiple angles and SNR levels.
* Static Activity Dataset: Collected using a 60 GHz FMCW radar (Infineon BGT60TR13C) in:

  * Laboratory environment
  * Long-Term Care facility

## Architecture

* Dynamic activity models trained on micro-Doppler spectrograms using SVM classifiers.
* Static activity models trained on 3D CNNs with Range-Azimuth maps as input.
* Temporal attention enhanced via soft and hard region masks guided by tracking.

## Visual Examples

* Denoising results of spectrograms across SNR levels
* RA map attention heatmaps using Grad-CAM
* Bounding box overlays and generated masks for static feature maps

## Evaluation Metrics
Included (in metrics.py):
Mean Squared Error (MSE)
Mean Absolute Error (MAE)
PSNR
Structural Similarity Index (SSIM)
Correlation Coefficient

## Key Contributions

1. Holistic evaluation framework for radar spectrogram denoising
2. A no-reference quality scoring algorithm for static RA maps
3. Temporal tracking to improve spatial coherence in static recognition
4. Demonstrated improved classification performance and interpretability across HAR regimes