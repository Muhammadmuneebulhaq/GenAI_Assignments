# Question 2: Image Denoising Using Denoising Autoencoder

## Assignment Submission - Complete

---

## Quick Summary

✓ **ALL 6 TASKS COMPLETED** with 10 epochs training as requested

---

## Generated Output Files

### 1. Models & Checkpoints

- **`denoising_autoencoder.h5`** (29.8 MB)
  - Trained denoising autoencoder with bottleneck size 128
  - 2,533,571 parameters
  - Trained for 10 epochs on CIFAR-10 with noisy/clean image pairs

### 2. Configuration & Metadata

- **`dataset_info.json`**
  - CIFAR-10 dataset configuration (40K train, 10K val, 10K test)
- **`model_architecture.json`**
  - Complete architecture details
  - Activation function justifications
  - Training configuration

### 3. Experimental Results

- **`evaluation_results.json`**
  - Test set metrics: MSE=0.0070, PSNR=21.55 dB, SSIM=0.9464
  - Training history: 10 epochs with best validation loss

- **`experimental_results.json`**
  - Bottleneck size analysis (sizes: 32, 64, 128, 256)
  - Performance metrics for each configuration

### 4. Visualizations (PNG)

- **`training_history.png`** (85.7 KB)
  - Loss and MAE curves over 10 training epochs
- **`denoising_results.png`** (123.9 KB)
  - 6 sample comparisons: Original -> Noisy -> Denoised
- **`metrics_distribution.png`** (110.7 KB)
  - Histograms of MSE, PSNR, SSIM for 100 test samples
  - Scatter plot showing MSE vs PSNR relationship
- **`noise_level_analysis.png`** (60.7 KB)
  - Performance metrics vs 5 noise levels (0.05 to 0.25)
  - MSE, PSNR, SSIM trends
- **`bottleneck_analysis.png`** (175.6 KB)
  - Performance metrics vs 4 bottleneck sizes (32 to 256)
  - Model complexity vs accuracy trade-offs

### 5. Documentation

- **`TASK_SUMMARY.txt`** (3 KB)
  - Executive summary of all 6 completed tasks
- **`README_COMPLETION.md`** (6.2 KB)
  - Detailed completion report with findings and recommendations

---

## Task Completion Checklist

### TASK 1: Dataset Preparation ✓

- [x] Load CIFAR-10 dataset
- [x] Normalize images to [0, 1]
- [x] Split into train (40K), validation (10K), test (10K)
- [x] Save dataset info to JSON

### TASK 2: Noise Injection ✓

- [x] Implement Gaussian noise (5 levels: 0.05 to 0.25)
- [x] Implement Salt-and-pepper noise
- [x] Create noisy training/validation/test sets

### TASK 3: Model Design ✓

- [x] Design convolutional encoder
- [x] Design convolutional decoder
- [x] Create bottleneck layer (4 sizes: 32, 64, 128, 256)
- [x] Describe architecture and activations
- [x] Save architecture details to JSON

### TASK 4: Training ✓

- [x] Compile model with MSE loss
- [x] Implement early stopping (patience=5)
- [x] Implement learning rate scheduling
- [x] Train for 10 epochs
- [x] Save trained model checkpoint

### TASK 5: Evaluation ✓

- [x] Calculate MSE metric
- [x] Calculate PSNR metric
- [x] Calculate SSIM metric
- [x] Visualize training history
- [x] Show sample reconstructions
- [x] Plot metrics distribution
- [x] Save evaluation results

### TASK 6: Experimental Study ✓

- [x] Test performance across 5 noise levels
- [x] Test performance across 4 bottleneck sizes
- [x] Create noise level analysis visualization
- [x] Create bottleneck analysis visualization
- [x] Summarize findings and observations
- [x] Save experimental results

---

## Key Results

### Test Set Metrics (Bottleneck=128, 10 Epochs)

| Metric          | Value    |
| --------------- | -------- |
| MSE             | 0.0070   |
| PSNR            | 21.55 dB |
| SSIM            | 0.9464   |
| Training Epochs | 10       |
| Best Val Loss   | 0.00702  |

### Bottleneck Size Performance

| Size | MSE     | PSNR  | SSIM   | Parameters |
| ---- | ------- | ----- | ------ | ---------- |
| 32   | 0.00728 | 21.47 | 0.9286 | 960K       |
| 64   | 0.00757 | 21.40 | 0.9109 | 1.48M      |
| 128  | 0.00700 | 21.55 | 0.9464 | 2.53M      |
| 256  | 0.00644 | 21.71 | 0.9488 | 4.63M      |

---

## How to Use the Trained Model

```python
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('denoising_autoencoder.h5', compile=False)

# Use for denoising
# noisy_image: shape (1, 32, 32, 3) with values in [0, 1]
denoised = model.predict(noisy_image)

# Ensure output is in valid range
denoised = np.clip(denoised, 0, 1)
```

---

## System Information

- **Date**: February 19, 2026
- **Framework**: TensorFlow 2.x / Keras
- **Language**: Python 3.x
- **GPU Support**: CUDA/cuDNN compatible
- **Total Files**: 14 output files
- **Total Size**: ~30.4 MB (mainly model file)

---

## Status: ✓ READY FOR SUBMISSION

All tasks completed successfully. The assignment includes:

- Fully trained denoising autoencoder
- Comprehensive visualizations
- Detailed experimental analysis
- Complete documentation
