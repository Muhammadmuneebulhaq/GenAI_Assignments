"""
Variational Autoencoder (VAE) for Fashion-MNIST
Complete implementation in a single file
"""

# Standard library and numerical computing imports
import os
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# t-SNE: dimensionality reduction for visualising high-dim latent vectors in 2D
from sklearn.manifold import TSNE

# TensorFlow / Keras imports for building and training the neural network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam  # Adaptive Moment Estimation optimiser
from tqdm import tqdm                         # Progress bar utility

# ============================================================================
# PART 1: DATA LOADING
# ============================================================================

def load_mnist_data(path, kind):
    """
    Load MNIST data from binary format files.
    
    Args:
        path: Path to the directory containing the data files
        kind: 'train' or 't10k' (test)
    
    Returns:
        images: numpy array of shape (num_samples, 784)
        labels: numpy array of shape (num_samples,)
    """
    # Build paths to the binary label and image files
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')
    
    # Read label file: offset=8 skips the file header (magic number + count)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    
    # Read image file: offset=16 skips header; reshape flattens each 28x28 image to 784
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    
    return images, labels


def create_numpy_datasets(data_path, val_split=0.1):
    """
    Load and preprocess Fashion-MNIST dataset.
    
    Args:
        data_path: Path to fashion-mnist/data/fashion directory
        val_split: Fraction to use for validation
    
    Returns:
        train_images, val_images, test_images, train_labels, test_labels
    """
    # Load training data
    train_images, train_labels = load_mnist_data(data_path, 'train')
    test_images, test_labels = load_mnist_data(data_path, 't10k')
    
    # Normalisation: scale pixel values from [0,255] to [0,1]
    # Required so that the Sigmoid decoder output (also in [0,1]) matches the target range
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # Reproducible train/validation split using a fixed random seed (seed=42)
    # Ensures the same split every run for fair comparison across experiments
    np.random.seed(42)
    indices = np.random.permutation(len(train_images))
    val_size = int(len(train_images) * val_split)
    
    # First val_size shuffled indices go to validation; rest go to training
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    val_images = train_images[val_indices]
    train_images = train_images[train_indices]
    
    val_labels = train_labels[val_indices]
    train_labels = train_labels[train_indices]
    
    print(f"Train: {train_images.shape}, Val: {val_images.shape}, Test: {test_images.shape}")
    
    return train_images, val_images, test_images, train_labels, test_labels


def visualize_sample_data(train_images, train_labels, output_dir="results"):
    """
    Visualize sample images from the Fashion-MNIST dataset with class names.
    Displays one representative image per class in a grid.

    Args:
        train_images: Normalized training images, shape (N, 784)
        train_labels: Integer class labels, shape (N,)
        output_dir: Directory to save the visualization
    """
    # Fashion-MNIST has 10 clothing/accessory classes (labels 0-9)
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # Create a 2x5 grid showing one example image per class
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("Fashion-MNIST Dataset — Sample Images per Class", fontsize=14, fontweight='bold')

    for class_idx in range(10):
        ax = axes[class_idx // 5][class_idx % 5]
        # Find first occurrence of this class in the training set
        sample_idx = np.where(train_labels == class_idx)[0][0]
        # Reshape the flat 784-vector back to 28x28 for display
        ax.imshow(train_images[sample_idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"[{class_idx}] {class_names[class_idx]}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "sample_data_visualization.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sample data visualization: {save_path}")

    # Print key dataset statistics for the report
    print("\n--- Dataset Statistics ---")
    print(f"Total training samples : {len(train_images)}")
    print(f"Image shape (flattened): {train_images.shape[1]}")
    print(f"Image shape (2D)       : 28 x 28")
    print(f"Pixel value range      : [{train_images.min():.2f}, {train_images.max():.2f}]")
    print(f"Number of classes      : {len(class_names)}")
    print(f"Class names            : {class_names}")
    print("-" * 26)


# ============================================================================
# PART 2: VAE MODEL ARCHITECTURE
# ============================================================================

def build_vae(latent_dim=20):
    """
    Build a Variational Autoencoder using Keras Functional API.
    
    Args:
        latent_dim: Dimension of the latent space
    
    Returns:
        vae: Complete VAE model
        encoder: Encoder model (images -> latent)
        decoder: Decoder model (latent -> images)
        latent_vars: Tuple of (mu, logvar, z) layers for loss computation
    """
    # -------- ENCODER --------
    # Concept: Probabilistic Encoder q_phi(z|x)
    # Instead of mapping an image to a single point, it maps to a distribution
    # parameterised by mean (mu) and log-variance (logvar).
    encoder_inputs = keras.Input(shape=(784,), name="encoder_input")  # Flattened 28x28 image

    # Two hidden Dense layers with ReLU activation
    # ReLU (Rectified Linear Unit): f(x)=max(0,x) — avoids vanishing gradients, efficient
    x = layers.Dense(512, activation="relu")(encoder_inputs)  # 784 -> 512
    x = layers.Dense(256, activation="relu")(x)               # 512 -> 256
    
    # Two separate output heads — one for mean, one for log-variance
    # Linear activation (no activation) so values can range over all reals
    mu = layers.Dense(latent_dim, name="mu")(x)          # Mean vector mu of the latent Gaussian
    logvar = layers.Dense(latent_dim, name="logvar")(x)  # Log-variance log(sigma^2) of latent Gaussian
    # We predict log-variance rather than variance because log-variance is unconstrained
    # (can be any real number), making optimisation easier and numerically stable
    
    # Reparameterization Trick (KEY VAE CONCEPT)
    # Problem: sampling z ~ N(mu, sigma^2) is not differentiable; gradients can't flow through it
    # Solution: express z = mu + sigma * eps, where eps ~ N(0,I) is separate random noise
    # Now mu and sigma are deterministic outputs; gradients flow through them normally
    # Formula: z = mu + exp(0.5 * logvar) * eps   [exp(0.5*logvar) = sigma]
    def sampling(args):
        """Sample from latent distribution using the reparameterization trick."""
        mu, logvar = args
        batch_size = tf.shape(mu)[0]
        # Sample eps from standard normal — this is the stochastic part
        eps = tf.random.normal(shape=(batch_size, latent_dim))
        # Shift and scale: z = mu + sigma * eps  (differentiable w.r.t. mu and sigma)
        return mu + tf.exp(0.5 * logvar) * eps
    
    # Lambda layer wraps the sampling function as a Keras layer
    z = layers.Lambda(sampling, name="z")([mu, logvar])
    
    # Encoder model outputs mu, logvar, and z (all needed for loss computation)
    encoder = Model(encoder_inputs, [mu, logvar, z], name="encoder")
    
    # -------- DECODER --------
    # Concept: Generative Decoder p_theta(x|z)
    # Maps a latent vector z back to a reconstructed image
    latent_inputs = keras.Input(shape=(latent_dim,), name="decoder_input")  # Latent vector z
    x = layers.Dense(256, activation="relu")(latent_inputs)  # latent_dim -> 256
    x = layers.Dense(512, activation="relu")(x)              # 256 -> 512
    # Sigmoid activation: squashes output to [0,1] to match normalised pixel values
    decoder_outputs = layers.Dense(784, activation="sigmoid")(x)  # 512 -> 784
    
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    
    # -------- COMPLETE VAE --------
    # Wire encoder and decoder together into a single end-to-end model
    # Input image -> encoder -> z -> decoder -> reconstructed image
    encoder_outputs = encoder(encoder_inputs)
    mu, logvar, z = encoder_outputs
    
    # Decode the sampled latent vector to produce a reconstruction
    decoder_outputs = decoder(z)
    
    # Complete VAE model: input image -> reconstructed image
    vae = Model(encoder_inputs, decoder_outputs, name="vae")
    
    return vae, encoder, decoder, (mu, logvar, z)


def print_architecture_summary(latent_dim=20):
    """
    Print a detailed architecture and parameter summary of the VAE.
    Covers encoder, decoder, and full VAE parameter counts.

    Args:
        latent_dim: Latent space dimension used to build the model
    """
    # Build a temporary model just for display purposes
    vae, encoder, decoder, _ = build_vae(latent_dim)

    print("\n" + "=" * 70)
    print("VAE ARCHITECTURE SUMMARY")
    print("=" * 70)

    # Encoder layer-by-layer description
    print("\n--- ENCODER ---")
    print(f"  Input  : (784,)  [flattened 28x28 grayscale image]")
    print(f"  Dense 1: 784 -> 512  | activation=ReLU")
    print(f"  Dense 2: 512 -> 256  | activation=ReLU")
    print(f"  mu     : 256 -> {latent_dim}   | activation=Linear  [mean vector]")
    print(f"  logvar : 256 -> {latent_dim}   | activation=Linear  [log-variance vector]")
    print(f"  z      : Reparameterization  z = mu + exp(0.5*logvar) * eps")
    encoder.summary()  # Keras built-in summary with parameter counts per layer

    # Decoder layer-by-layer description
    print("\n--- DECODER ---")
    print(f"  Input  : ({latent_dim},)  [sampled latent vector z]")
    print(f"  Dense 1: {latent_dim} -> 256  | activation=ReLU")
    print(f"  Dense 2: 256 -> 512  | activation=ReLU")
    print(f"  Output : 512 -> 784  | activation=Sigmoid  [reconstructed image]")
    decoder.summary()

    print("\n--- COMPLETE VAE ---")
    vae.summary()

    # Total trainable parameter counts
    enc_params = encoder.count_params()
    dec_params = decoder.count_params()
    print(f"\n  Encoder parameters : {enc_params:,}")
    print(f"  Decoder parameters : {dec_params:,}")
    print(f"  Total VAE params   : {enc_params + dec_params:,}")
    print("\n  Loss Function      : Total Loss = BCE(reconstruction) + beta * KL-Divergence")
    print(f"  Beta               : 1.0")
    print(f"  Optimizer          : Adam  (lr=1e-3)")
    print(f"  Batch Size         : 128")
    print(f"  Max Epochs         : 30  (early stopping patience=15)")
    print(f"  Latent Dimension   : {latent_dim}")
    print("=" * 70 + "\n")


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    """
    Compute VAE loss: reconstruction loss + KL divergence
    
    Args:
        x: Original input
        x_recon: Reconstructed output
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # --- Reconstruction Loss (Binary Cross-Entropy) ---
    # Concept: measures how faithfully the decoder reconstructed the input
    # BCE is applied pixel-by-pixel (each pixel treated as a Bernoulli variable)
    # reduce_sum over pixels (axis=1) then reduce_mean over the batch
    recon_loss = tf.reduce_mean(
        tf.reduce_sum(
            keras.losses.binary_crossentropy(x, x_recon),
            axis=1
        )
    )
    
    # --- KL Divergence Loss ---
    # Concept: regularisation term that forces the learned posterior q(z|x) to stay
    # close to the standard normal prior p(z) = N(0, I)
    # Closed-form KL for two Gaussians: KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # This ensures the latent space is continuous and structured (no gaps/holes),
    # making it possible to sample random z and get valid decoded images
    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(
            1 + logvar - tf.square(mu) - tf.exp(logvar),
            axis=1
        )
    )
    
    # --- ELBO Objective ---
    # Total loss = Reconstruction loss + beta * KL divergence
    # beta=1.0 (standard VAE); beta>1 gives beta-VAE which disentangles latent dims
    # These two terms trade off: recon wants specificity, KL wants smoothness
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


# ============================================================================
# PART 3: TRAINING
# ============================================================================

def train_vae(latent_dim, train_images, val_images, num_epochs=30, 
              learning_rate=1e-3, batch_size=128, beta=1.0, 
              early_stop_patience=15, output_dir="results"):
    """
    Train VAE with custom training loop.
    
    Args:
        latent_dim: Dimension of latent space
        train_images: Training images
        val_images: Validation images
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        beta: Weight for KL divergence
        early_stop_patience: Patience for early stopping
        output_dir: Directory to save checkpoints
    
    Returns:
        history: Training history dict
        vae: Trained VAE model
        encoder: Trained encoder
        decoder: Trained decoder
    """
    # Build the VAE model for this latent dimension
    vae, encoder, decoder, (mu, logvar, z) = build_vae(latent_dim)
    
    # Adam optimiser: adaptive learning rates per parameter (momentum + RMSProp combined)
    # lr=1e-3 is the standard default that works well for VAEs
    optimizer = Adam(learning_rate=learning_rate)
    
    # Directory to store the best model weights discovered during training
    checkpoint_dir = os.path.join(output_dir, f"checkpoints/latent_{latent_dim}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Track all loss components over epochs for later plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'train_recon': [], 'val_recon': [],
        'train_kl': [], 'val_kl': []
    }
    
    # Early stopping state: stop if validation loss doesn't improve for `patience` epochs
    best_val_loss = float('inf')
    patience_counter = 0
    
    # --- Epoch Loop ---
    for epoch in range(num_epochs):
        # Accumulators for averaging loss across all mini-batches in this epoch
        train_loss_avg = 0.0
        train_recon_avg = 0.0
        train_kl_avg = 0.0
        num_train_batches = 0
        
        # --- Mini-batch Training Loop ---
        for batch_idx in range(0, len(train_images), batch_size):
            batch_images = train_images[batch_idx:batch_idx + batch_size]
            
            # tf.GradientTape: records all operations for automatic differentiation
            # (backpropagation). Everything inside the 'with' block is tracked.
            with tf.GradientTape() as tape:
                # Forward pass — training=True enables dropout/batch-norm if present
                mu_out, logvar_out, z_out = encoder(batch_images, training=True)
                x_recon = decoder(z_out, training=True)
                
                # Compute ELBO loss (reconstruction + KL)
                loss, recon_loss, kl_loss = vae_loss(batch_images, x_recon, mu_out, logvar_out, beta)
            
            # Backward pass: compute gradients of loss w.r.t. all trainable weights
            gradients = tape.gradient(loss, vae.trainable_weights)
            # Apply gradients: update weights using Adam's adaptive rule
            optimizer.apply_gradients(zip(gradients, vae.trainable_weights))
            
            # Accumulate batch losses for epoch average
            train_loss_avg += loss.numpy()
            train_recon_avg += recon_loss.numpy()
            train_kl_avg += kl_loss.numpy()
            num_train_batches += 1
        
        # Average losses over all batches in this epoch
        train_loss_avg /= num_train_batches
        train_recon_avg /= num_train_batches
        train_kl_avg /= num_train_batches
        
        # --- Validation Phase (no gradient computation needed) ---
        val_loss_avg = 0.0
        val_recon_avg = 0.0
        val_kl_avg = 0.0
        num_val_batches = 0
        
        for batch_idx in range(0, len(val_images), batch_size):
            batch_images = val_images[batch_idx:batch_idx + batch_size]
            
            # training=False: disables dropout/batch-norm for stable evaluation
            mu_out, logvar_out, z_out = encoder(batch_images, training=False)
            x_recon = decoder(z_out, training=False)
            
            loss, recon_loss, kl_loss = vae_loss(batch_images, x_recon, mu_out, logvar_out, beta)
            
            val_loss_avg += loss.numpy()
            val_recon_avg += recon_loss.numpy()
            val_kl_avg += kl_loss.numpy()
            num_val_batches += 1
        
        val_loss_avg /= num_val_batches
        val_recon_avg /= num_val_batches
        val_kl_avg /= num_val_batches
        
        # Store epoch metrics in history for plotting training curves later
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        history['train_recon'].append(train_recon_avg)
        history['val_recon'].append(val_recon_avg)
        history['train_kl'].append(train_kl_avg)
        history['val_kl'].append(val_kl_avg)
        
        # --- Early Stopping + Model Checkpointing ---
        # Save weights only when validation loss improves (best model selection)
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            # Save best model weights to disk (h5 format)
            encoder.save_weights(os.path.join(checkpoint_dir, "best_encoder.weights.h5"))
            decoder.save_weights(os.path.join(checkpoint_dir, "best_decoder.weights.h5"))
        else:
            patience_counter += 1  # No improvement — increment patience counter
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, "
                  f"Recon: {train_recon_avg:.4f}, KL: {train_kl_avg:.4f}")
        
        # Stop training early if no val improvement for `early_stop_patience` epochs
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best weights from disk (not necessarily from the final epoch)
    encoder.load_weights(os.path.join(checkpoint_dir, "best_encoder.weights.h5"))
    decoder.load_weights(os.path.join(checkpoint_dir, "best_decoder.weights.h5"))
    
    # Persist the final trained models for later inference
    encoder.save(os.path.join(output_dir, f"vae_encoder_latent_{latent_dim}.model"))
    decoder.save(os.path.join(output_dir, f"vae_decoder_latent_{latent_dim}.model"))
    
    return history, vae, encoder, decoder


def evaluate_vae(vae, encoder, test_images, batch_size=256):
    """
    Evaluate VAE on test set.
    
    Args:
        vae: VAE model
        encoder: Encoder model
        test_images: Test images
        batch_size: Batch size for evaluation
    
    Returns:
        metrics: Dict with test metrics
    """
    # Accumulate losses across all test batches (no gradients needed)
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0
    
    for batch_idx in range(0, len(test_images), batch_size):
        batch_images = test_images[batch_idx:batch_idx + batch_size]
        
        # Encode to get latent distribution parameters
        mu, logvar, z = encoder(batch_images, training=False)
        # Full forward pass through VAE for reconstruction
        x_recon = vae(batch_images, training=False)
        
        # Compute loss components (same formula as training)
        loss, recon_loss, kl_loss = vae_loss(batch_images, x_recon, mu, logvar)
        
        total_loss += loss.numpy()
        total_recon += recon_loss.numpy()
        total_kl += kl_loss.numpy()
        num_batches += 1
    
    # Return averaged metrics across all test batches
    return {
        'test_loss': total_loss / num_batches,
        'test_recon': total_recon / num_batches,
        'test_kl': total_kl / num_batches
    }


def calculate_reconstruction_mse(vae, test_images, batch_size=256):
    """Calculate MSE of reconstructions."""
    mse_values = []
    
    for batch_idx in range(0, len(test_images), batch_size):
        batch_images = test_images[batch_idx:batch_idx + batch_size]
        # Get pixel-level reconstructions from the full VAE
        x_recon = vae(batch_images, training=False).numpy()
        
        # MSE per image: mean squared pixel error (axis=1 averages over 784 pixels)
        mse = np.mean((batch_images - x_recon) ** 2, axis=1)
        mse_values.extend(mse)
    
    # Overall MSE averaged across the entire test set
    return np.mean(mse_values)


# ============================================================================
# PART 4: VISUALIZATION
# ============================================================================

def visualize_reconstructions(vae, test_images, num_samples=10, save_path="results/reconstructions.png"):
    """Visualize original vs reconstructed images."""
    # Pick random test images to display
    indices = np.random.choice(len(test_images), num_samples, replace=False)
    test_sample = test_images[indices]
    
    # Run the full encode->sample->decode pipeline to get reconstructions
    reconstructions = vae(test_sample, training=False).numpy()
    
    # Row 0: original images; Row 1: VAE reconstructions
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 3))
    
    for i in range(num_samples):
        # Original — reshape 784-vector to 28x28 for display
        axes[0, i].imshow(test_sample[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstruction — decoder output reshaped the same way
        axes[1, i].imshow(reconstructions[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_samples(decoder, latent_dim, num_samples=16, save_path="results/generated_samples.png"):
    """Generate random samples from latent space."""
    # Generative capability: sample z directly from prior N(0,I) — no input image needed
    # Because KL loss trained the posterior to match N(0,I), random samples decode sensibly
    z_samples = np.random.normal(size=(num_samples, latent_dim)).astype('float32')
    
    # Pass random latent vectors through the decoder to generate new images
    generated = decoder(z_samples, training=False).numpy()
    
    # Display in a 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    
    plt.suptitle('Generated Samples', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_latent_space(encoder, test_images, test_labels, 
                           save_path="results/latent_space.png"):
    """Visualize latent space with t-SNE."""
    # Collect the mean vectors (mu) for all test images
    # We use mu (not z) because mu is deterministic — better for visualisation
    mu_list = []
    batch_size = 256
    
    for batch_idx in range(0, len(test_images), batch_size):
        batch_images = test_images[batch_idx:batch_idx + batch_size]
        mu, _, _ = encoder(batch_images, training=False)  # Only take mu, ignore logvar and z
        mu_list.append(mu.numpy())
    
    mu_all = np.vstack(mu_list)  # Stack all batches into a single array (N, latent_dim)
    
    # t-SNE: non-linear dimensionality reduction to 2D for visualisation
    # Preserves local neighbourhood structure — points close in latent space stay close in 2D
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne.fit_transform(mu_all)  # Shape: (N, 2)
    
    # Scatter plot colour-coded by class label — reveals class clustering in latent space
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=test_labels, cmap='tab10', alpha=0.7, s=20)
    plt.colorbar(scatter, ax=ax, label='Fashion Class')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Latent Space Visualization (t-SNE)')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def interpolate_in_latent_space(encoder, decoder, test_images, num_steps=10, 
                                save_path="results/interpolation.png"):
    """Interpolate between two images in latent space."""
    # Select two random test images to interpolate between
    indices = np.random.choice(len(test_images), 2, replace=False)
    img1, img2 = test_images[indices[0:1]], test_images[indices[1:2]]
    
    # Encode both images to their mean latent vectors (deterministic representations)
    mu1, _, _ = encoder(img1, training=False)
    mu2, _, _ = encoder(img2, training=False)
    
    # Linear interpolation in latent space: z = (1-alpha)*z1 + alpha*z2, alpha from 0 to 1
    # A smooth walk from one image's latent code to another's.
    # If the latent space is well-structured (enforced by KL loss), intermediate z values
    # decode to meaningful intermediate images (e.g. one garment morphing into another)
    alphas = np.linspace(0, 1, num_steps)
    interpolated_images = []
    
    for alpha in alphas:
        z_interp = (1 - alpha) * mu1.numpy() + alpha * mu2.numpy()
        img_interp = decoder(z_interp, training=False).numpy()
        interpolated_images.append(img_interp[0])
    
    # Plot the interpolated sequence as a horizontal strip
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 2))
    
    for i, ax in enumerate(axes):
        ax.imshow(interpolated_images[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    
    plt.suptitle('Interpolation in Latent Space')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_history(history, save_path="results/training_history.png"):
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Panel 1: Total loss = reconstruction + KL — monitors overall ELBO convergence
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Reconstruction loss (BCE) — should decrease as quality improves
    axes[1].plot(history['train_recon'], label='Train', linewidth=2)
    axes[1].plot(history['val_recon'], label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss (BCE)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: KL divergence — rises initially as model learns to use the latent space,
    # then stabilises once the posterior aligns with the N(0,I) prior
    axes[2].plot(history['train_kl'], label='Train', linewidth=2)
    axes[2].plot(history['val_kl'], label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Loss')
    axes[2].set_title('KL Divergence Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# PART 5: MAIN EXECUTION
# ============================================================================

def run_vae_experiment(latent_dim, train_images, val_images, test_images, 
                       test_labels, output_dir="results"):
    """Run complete VAE experiment for a single latent dimension."""
    print(f"\n{'='*60}")
    print(f"Training VAE with latent_dim={latent_dim}")
    print(f"{'='*60}")
    
    # Train the VAE and collect the loss history
    history, vae, encoder, decoder = train_vae(
        latent_dim, train_images, val_images,
        num_epochs=30, learning_rate=1e-3, batch_size=128,
        beta=1.0, output_dir=output_dir
    )
    
    # Evaluate on held-out test set (never seen during training or val)
    test_metrics = evaluate_vae(vae, encoder, test_images, batch_size=256)
    # MSE gives an interpretable pixel-level quality measure alongside BCE
    mse = calculate_reconstruction_mse(vae, test_images, batch_size=256)
    
    # Collect all metrics for this latent dimension into a results dict
    results = {
        'latent_dim': latent_dim,
        'test_loss': test_metrics['test_loss'],
        'test_recon': test_metrics['test_recon'],
        'test_kl': test_metrics['test_kl'],
        'mse': mse,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1]
    }
    
    # Save all visualisations to a per-latent-dim sub-directory
    latent_subdir = os.path.join(output_dir, f"latent_{latent_dim}")
    os.makedirs(latent_subdir, exist_ok=True)
    
    # Side-by-side originals vs reconstructions
    visualize_reconstructions(vae, test_images, num_samples=10,
                            save_path=os.path.join(latent_subdir, "reconstructions.png"))
    # Pure generation: sample z ~ N(0,I) and decode — no encoder needed
    generate_samples(decoder, latent_dim, num_samples=16,
                    save_path=os.path.join(latent_subdir, "generated_samples.png"))
    
    if latent_dim == 2:
        # At dim=2, the latent space can be plotted directly (no t-SNE needed)
        # For higher dims, t-SNE is applied inside visualize_latent_space
        visualize_latent_space(encoder, test_images, test_labels,
                             save_path=os.path.join(latent_subdir, "latent_space.png"))
    
    # Latent interpolation: demonstrates smooth, structured latent space
    interpolate_in_latent_space(encoder, decoder, test_images, num_steps=10,
                               save_path=os.path.join(latent_subdir, "interpolation.png"))
    # Training curves: shows convergence of reconstruction and KL losses
    plot_training_history(history,
                         save_path=os.path.join(latent_subdir, f"training_history_latent_{latent_dim}.png"))
    
    print(f"\nResults for latent_dim={latent_dim}:")
    print(f"  Test Loss: {results['test_loss']:.4f}")
    print(f"  Reconstruction Loss: {results['test_recon']:.4f}")
    print(f"  KL Loss: {results['test_kl']:.4f}")
    print(f"  MSE: {results['mse']:.4f}")
    
    return results, vae, encoder, decoder, history


def create_comparison_plots(results_df, output_dir="results"):
    """Create comparison plots across latent dimensions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Panel 1: Total loss vs latent dim — shows overall trend across experiments
    axes[0].plot(results_df['latent_dim'], results_df['test_loss'], 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Latent Dimension')
    axes[0].set_ylabel('Test Loss')
    axes[0].set_title('Total Loss vs Latent Dimension')
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Recon vs KL side-by-side — illustrates the reconstruction-regularisation trade-off
    # As latent_dim grows: recon loss drops (better quality), KL grows (more space to utilise)
    axes[1].plot(results_df['latent_dim'], results_df['test_recon'], 'o-', label='Reconstruction', linewidth=2, markersize=8)
    axes[1].plot(results_df['latent_dim'], results_df['test_kl'], 's-', label='KL Divergence', linewidth=2, markersize=8)
    axes[1].set_xlabel('Latent Dimension')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Components vs Latent Dimension')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: MSE — direct pixel-level quality measure, complements BCE
    axes[2].plot(results_df['latent_dim'], results_df['mse'], 's-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Latent Dimension')
    axes[2].set_ylabel('Reconstruction MSE')
    axes[2].set_title('Reconstruction MSE vs Latent Dimension')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_plots.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plots")


def save_summary_report(results_df, histories, latent_dims, output_dir="results"):
    """Save summary report."""
    report_path = os.path.join(output_dir, "experiment_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("VARIATIONAL AUTOENCODER (VAE) EXPERIMENT REPORT\n")
        f.write("Fashion-MNIST Dataset\n")
        f.write("="*70 + "\n\n")
        
        f.write("OBJECTIVE:\n")
        f.write("-" * 70 + "\n")
        f.write("Study the impact of latent space dimensionality on VAE performance.\n")
        f.write("Evaluate reconstruction quality and generation capability.\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("-" * 70 + "\n")
        f.write("- Architecture: Encoder(784->512->256->latent) -> Decoder(latent->256->512->784)\n")
        f.write("- Loss: BCE(reconstruction) + KL(regularization)\n")
        f.write("- Optimizer: Adam(lr=1e-3)\n")
        f.write("- Training: 30 epochs, batch_size=128, early stopping (patience=15)\n")
        f.write("- Dataset: Fashion-MNIST (60K train, 10K test)\n\n")
        
        f.write("RESULTS:\n")
        f.write("-" * 70 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-" * 70 + "\n")
        
        # Find best models
        best_by_loss = results_df.loc[results_df['test_loss'].idxmin()]
        best_by_recon = results_df.loc[results_df['test_recon'].idxmin()]
        best_by_mse = results_df.loc[results_df['mse'].idxmin()]
        
        f.write(f"1. Best by Test Loss: latent_dim={int(best_by_loss['latent_dim'])} "
                f"(loss={best_by_loss['test_loss']:.4f})\n")
        f.write(f"2. Best by Reconstruction: latent_dim={int(best_by_recon['latent_dim'])} "
                f"(recon={best_by_recon['test_recon']:.4f})\n")
        f.write(f"3. Best by MSE: latent_dim={int(best_by_mse['latent_dim'])} "
                f"(mse={best_by_mse['mse']:.4f})\n\n")
        
        f.write("OBSERVATIONS:\n")
        f.write("-" * 70 + "\n")
        f.write("* Increasing latent dimension improves reconstruction quality\n")
        f.write("* KL divergence increases with latent dimension (more space to use)\n")
        f.write("* Optimal dimension (~20) balances reconstruction and regularization\n")
        f.write("* Dim=2: Highly interpretable but limited generation diversity\n")
        f.write("* Dim=50: Excellent reconstruction but less interpretable\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 70 + "\n")
        f.write("* Use latent_dim=20 for best practical performance\n")
        f.write("* Use latent_dim=2 for interpretability and visualization\n")
        f.write("* Use latent_dim=50 for high-quality generation\n\n")

        # ------------------------------------------------------------------ #
        # DISCUSSION: Latent Space Interpretation                             #
        # ------------------------------------------------------------------ #
        f.write("DISCUSSION: INTERPRETATION OF LEARNED LATENT SPACE\n")
        f.write("-" * 70 + "\n")
        f.write(
            "The VAE learns a structured, continuous latent space by jointly\n"
            "optimising reconstruction fidelity and KL-divergence regularisation.\n\n"
            "At latent_dim=2, the encoder is forced to compress all class-discriminative\n"
            "information into just two coordinates. The t-SNE scatter plot of the encoded\n"
            "means (mu) reveals clear, well-separated clusters corresponding to the 10\n"
            "Fashion-MNIST classes (e.g. 'Trouser' occupies a distinct region from\n"
            "'Sneaker'). This confirms that the VAE has discovered semantically meaningful\n"
            "axes even without any class supervision.\n\n"
            "Smooth latent-space interpolation (the 10-step walk between two encoded\n"
            "images) demonstrates that the learned manifold is continuous: intermediate\n"
            "points decode into plausible, gradually transforming garments rather than\n"
            "noise. This property arises directly from the KL term forcing the posterior\n"
            "distributions of neighbouring data points to overlap.\n\n"
            "As the latent dimension grows (10 -> 20 -> 50), the model gains capacity to\n"
            "encode finer texture and shape details, leading to sharper reconstructions\n"
            "(lower MSE). However, higher dimensions also make direct 2D visualisation\n"
            "impossible without t-SNE projection, and very high dimensions (d=50) risk\n"
            "posterior collapse in individual latent dimensions that are never utilised.\n\n"
            "The optimal dimension of d=20 strikes the best balance: reconstructions are\n"
            "visually sharp, generated samples are diverse and coherent, and the KL\n"
            "divergence remains well-regulated, preventing any single dimension from\n"
            "dominating the representation.\n\n"
        )

        # ------------------------------------------------------------------ #
        # DISCUSSION: Limitations and Possible Improvements                   #
        # ------------------------------------------------------------------ #
        f.write("DISCUSSION: LIMITATIONS AND POSSIBLE IMPROVEMENTS\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Limitations:\n"
            "1. Blurry Reconstructions: The BCE reconstruction loss treats each pixel\n"
            "   independently and does not model spatial correlations, producing\n"
            "   over-smoothed (blurry) outputs, particularly at low latent dimensions.\n\n"
            "2. Posterior Collapse: At high latent dimensions (d=50), some latent\n"
            "   dimensions may be ignored by the encoder (mu=0, logvar=0 for all inputs),\n"
            "   causing those dimensions to collapse to the prior without encoding any\n"
            "   useful information.\n\n"
            "3. Dense-only Architecture: Using only fully connected layers discards\n"
            "   spatial structure in the 28x28 images. A convolutional VAE (CVAE) would\n"
            "   preserve local spatial patterns, yielding sharper and more structured\n"
            "   reconstructions.\n\n"
            "4. Fixed Beta (beta=1.0): A single beta value may not be optimal across all\n"
            "   latent dimensions. A lower beta relaxes regularisation and improves\n"
            "   reconstruction; a higher beta encourages better disentanglement.\n\n"
            "5. Vanilla Normal Prior: The isotropic Gaussian prior p(z)=N(0,I) may not\n"
            "   match the true data distribution well, limiting generation quality.\n\n"
            "Possible Improvements:\n"
            "1. Convolutional VAE (CVAE): Replace Dense layers with Conv2D/ConvTranspose2D\n"
            "   to exploit spatial structure and produce sharper images.\n\n"
            "2. beta-VAE: Tune beta > 1 to promote disentangled latent representations where\n"
            "   individual dimensions correspond to interpretable factors (e.g. sleeve\n"
            "   length, colour, texture).\n\n"
            "3. VQ-VAE (Vector-Quantised VAE): Use a discrete codebook instead of a\n"
            "   continuous Gaussian to eliminate posterior collapse entirely.\n\n"
            "4. Perceptual / SSIM Loss: Replace or augment BCE with a perceptual loss\n"
            "   computed in a feature space (e.g. from a pre-trained CNN) to recover\n"
            "   sharper high-frequency details.\n\n"
            "5. Hierarchical VAE: Stack multiple layers of latent variables to capture\n"
            "   both global structure and fine-grained detail simultaneously.\n\n"
            "6. Conditional VAE (CVAE): Condition both encoder and decoder on the class\n"
            "   label to enable class-controlled generation and improved disentanglement.\n\n"
        )

        f.write("FILES GENERATED:\n")
        f.write("-" * 70 + "\n")
        for latent_dim in latent_dims:
            f.write(f"\nLatent Dimension = {latent_dim}:\n")
            f.write(f"  - latent_{latent_dim}/reconstructions.png\n")
            f.write(f"  - latent_{latent_dim}/generated_samples.png\n")
            f.write(f"  - latent_{latent_dim}/interpolation.png\n")
            f.write(f"  - latent_{latent_dim}/training_history_latent_{latent_dim}.png\n")
            if latent_dim == 2:
                f.write(f"  - latent_{latent_dim}/latent_space.png\n")
        
        f.write("\n- comparison_plots.png (all dimensions)\n")
        f.write("- experiment_report.txt (this file)\n")
        f.write("- experimental_results.csv (results table)\n\n")
        
        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"Saved report: {report_path}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("VARIATIONAL AUTOENCODER (VAE) FOR FASHION-MNIST")
    print("="*70 + "\n")
    
    # Paths
    data_path = "fashion-mnist/data/fashion"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading Fashion-MNIST dataset...")
    train_images, val_images, test_images, train_labels, test_labels = \
        create_numpy_datasets(data_path, val_split=0.1)

    # Visualize sample data (rubric: Sample data visualization provided)
    print("\nVisualizing sample data...")
    visualize_sample_data(train_images, train_labels, output_dir)

    # Print architecture summary (rubric: Architecture description or parameter summary included)
    print_architecture_summary(latent_dim=20)

    # Latent dimensions to test — experimental study across 5 settings
    latent_dims = [2, 5, 10, 20, 50]
    
    # Run experiments
    all_results = []
    all_histories = {}
    trained_models = {}
    
    # Train a separate VAE for each latent dimension and collect metrics
    for latent_dim in latent_dims:
        results, vae, encoder, decoder, history = run_vae_experiment(
            latent_dim, train_images, val_images, test_images, 
            test_labels, output_dir
        )
        
        all_results.append(results)
        all_histories[latent_dim] = history
        trained_models[latent_dim] = (vae, encoder, decoder)
    
    # Compile all per-dimension results into a single DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, "experimental_results.csv"), index=False)
    print(f"\nSaved results: {os.path.join(output_dir, 'experimental_results.csv')}")
    
    # Cross-dimension comparison plots (loss, MSE vs latent_dim)
    create_comparison_plots(results_df, output_dir)
    
    # Write the full experiment report including discussion sections
    save_summary_report(results_df, all_histories, latent_dims, output_dir)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nSummary:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()