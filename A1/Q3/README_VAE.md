# Variational Autoencoder (VAE) for Fashion-MNIST

## Overview

This project implements a Variational Autoencoder (VAE) to learn latent representations of Fashion-MNIST images and generate new synthetic samples. The implementation includes a complete experimental study analyzing the impact of different latent dimensions on model performance.

## Project Structure

### Core Components

1. **vae_model.py** - VAE Architecture
   - `build_vae(latent_dim=20)`: Constructs encoder, decoder, and complete VAE
   - `vae_loss()`: Implements combined reconstruction + KL divergence loss
   - Reparameterization trick for efficient sampling

2. **data_loader.py** - Data Preprocessing
   - `load_mnist_data()`: Loads Fashion-MNIST from binary files
   - `create_numpy_datasets()`: Prepares train/val/test splits
   - Normalization to [0, 1] range

3. **train_vae.py** - Training & Evaluation
   - `train_vae()`: Training loop with early stopping and loss tracking
   - `evaluate_vae()`: Evaluation on test set
   - `calculate_reconstruction_mse()`: Quality metric
   - `plot_training_history()`: Visualization of training curves

4. **visualization.py** - Visualization Utilities
   - `visualize_reconstructions()`: Compare original vs reconstructed images
   - `generate_samples()`: Generate new images from random latent vectors
   - `visualize_latent_space()`: t-SNE visualization of latent representations
   - `interpolate_in_latent_space()`: Smooth interpolation between images
   - `visualize_latent_dimensionality_effect()`: Compare quality across dimensions

5. **main.py** - Experimental Pipeline
   - Trains VAE models with different latent dimensions: [2, 5, 10, 20, 50]
   - Generates comprehensive results and visualizations
   - Creates summary report

## VAE Architecture

### Encoder

```
Input (28×28 image) → Flatten (784) → Dense(512, ReLU) → Dense(256, ReLU)
                     → Dense(latent_dim)  [μ - mean]
                     → Dense(latent_dim)  [σ² - log variance]
```

### Reparameterization Trick

```
z = μ + σ * ε
where ε ~ N(0, 1)
```

### Decoder

```
Input (latent_dim) → Dense(256, ReLU) → Dense(512, ReLU) → Dense(784, Sigmoid)
                     → Reshape (28×28 image)
```

## Loss Function

The VAE loss combines two components:

```
L_total = L_reconstruction + β * L_KL

L_reconstruction = BCE(x, x_recon)  [Binary Cross-Entropy]

L_KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)  [KL Divergence from N(0,1)]

β = 1.0  [Standard VAE - equal weighting]
```

## Key Features

### 1. Dataset Preparation ✓

- Loads 60,000 training + 10,000 test Fashion-MNIST images
- Normalizes to [0, 1] range
- Splits training data: 54,000 train / 6,000 validation

### 2. Variational Autoencoder Architecture ✓

- Encoder: 784 → 512 → 256 → latent_dim
- Decoder: latent_dim → 256 → 512 → 784
- Reparameterization trick for efficient gradient flow
- Returns both reconstructed images and latent vectors

### 3. Loss Function Implementation ✓

- Reconstruction loss (Binary Cross-Entropy)
- KL divergence regularization
- Combined weighted loss

### 4. Model Training ✓

- Adam optimizer (lr=1e-3)
- Batch size: 128
- Early stopping with patience=15
- Monitors both reconstruction and KL losses separately

### 5. Image Generation & Visualization ✓

- Reconstruction comparison: original vs reconstructed
- Random generation from learned latent space
- Latent space visualization with t-SNE
- Interpolation between images in latent space
- Dimension comparison visualizations

### 6. Experimental Study ✓

- Tests latent dimensions: [2, 5, 10, 20, 50]
- Tracks: Total Loss, Reconstruction Loss, KL Loss, MSE
- Generates comparison plots and summary report

## Running the Code

### Full Experimental Study

```bash
python main.py
```

This will:

1. Load Fashion-MNIST data
2. Train 5 VAE models with different latent dimensions
3. Generate all visualizations
4. Create experimental_results.csv
5. Save detailed experiment_report.txt

### Quick Demo

```bash
python demo.py
```

This demonstrates the VAE architecture with minimal training (2-3 epochs).

## Results

### Experimental Findings

The study reveals several important patterns:

1. **Reconstruction Quality (MSE)**
   - Improves with larger latent dimensions
   - Saturates at latent_dim ≥ 20

2. **KL Divergence Loss**
   - Higher with larger latent spaces
   - Indicates better latent space utilization
   - Risk of posterior collapse with very large dims

3. **Optimal Dimension**
   - latent_dim=20 provides excellent balance
   - Good reconstruction + meaningful latent space
   - Suitable for generation and interpolation

4. **Latent Dimension Effects**
   - latent_dim=2,5: Excellent visualization, limited diversity
   - latent_dim=10: Good balance at smaller scale
   - latent_dim=20: Recommended for most applications
   - latent_dim=50: Very high quality, larger model

## Generated Outputs

When running the experiment, you get:

### Models

- `vae_latent_*.model` - Complete VAE models
- `encoder_latent_*.model` - Encoders
- `decoder_latent_*.model` - Decoders

### Visualizations

- `reconstructions.png` - Original vs reconstructed (10 samples)
- `generated_samples.png` - 16 newly generated fashion items
- `latent_space.png` - 2D t-SNE vis of entire test set
- `interpolation.png` - Smooth morphing between 2 images
- `latent_dim_comparison.png` - Quality across dimensions
- `comparison_plots.png` - Loss curves and metrics
- `training_history_latent_*.png` - Training history for each model
- `experimental_results.csv` - Results table

### Reports

- `experiment_report.txt` - Detailed analysis and conclusions

## Technical Details

### Framework

- **TensorFlow 2.x** / **Keras** - Deep learning
- **NumPy** - Numerical computations
- **Pandas** - Results analysis
- **Matplotlib** - Visualizations
- **Scikit-learn** - t-SNE dimensionality reduction

### Training Configuration

- Optimizer: Adam (lr=1e-3)
- Batch Size: 128
- Epochs: 30 (with early stopping)
- Loss: BCE + KL divergence
- Gradient clipping: Not applied (stable training)
- Learning rate scheduling: ReduceLROnPlateau

## Key Insights

1. **Mode Coverage**: VAEs balance generation diversity with reconstruction quality
2. **Latent Structure**: Larger latent dimensions allow more expressive representations
3. **Trade-offs**: Small dims = interpretable but limited, Large dims = powerful but less interpretable
4. **Regularization**: KL term prevents posterior collapse and maintains useful latent space

## Applications

This VAE implementation can be used for:

- Image generation and synthesis
- Data augmentation
- Anomaly detection
- Latent representation learning
- Dimensionality reduction
- Interpolation and morphing

## Parameter Tuning Tips

- **Hidden Dims**: [512, 256] works well for 28×28 images
- **Latent Dim**: Start with 20, adjust based on quality/diversity needs
- **Learning Rate**: 1e-3 good default, reduce if unstable
- **Batch Size**: 128 balanced, increase for speed, decrease for stability
- **Beta (KL weight)**: 1.0 is standard, increase for stronger regularization
- **Epochs**: 30-50 typically sufficient, use early stopping

## Dependencies

```
tensorflow >= 2.10
keras >= 2.10
numpy
pandas
matplotlib
scikit-learn
tqdm
```

## Installation

```bash
cd d:\Sem8\GenAI\Assignments\A1\Q3
python -m venv .venv
.venv\Scripts\activate
pip install tensorflow numpy pandas scikit-learn matplotlib tqdm
```

## References

- Kingma & Welling (2014) - "Auto-Encoding Variational Bayes"
- Doersch (2016) - Tutorial on Variational Autoencoders
- Fashion-MNIST Dataset - Zalando Research

## Author Notes

This implementation demonstrates:
✓ Complete VAE architecture from scratch
✓ Proper loss function implementation
✓ Reparameterization trick
✓ Comprehensive evaluation
✓ Extensive visualizations
✓ Systematic experimental study

Perfect for understanding VAEs and their behavior on real image data!
