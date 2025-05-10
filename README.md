# Radar-Based Human Activity Recognition (HAR)

This project explores advanced signal processing and deep learning techniques for radar-based human activity recognition (HAR), focusing on both dynamic and static activity classification. Radar offers a privacy-preserving, non-contact alternative to vision-based systems, especially valuable in sensitive settings such as long-term care facilities.

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

## Key Contributions

1. Holistic evaluation framework for radar spectrogram denoising
2. A no-reference quality scoring algorithm for static RA maps
3. Temporal tracking to improve spatial coherence in static recognition
4. Demonstrated improved classification performance and interpretability across HAR regimes