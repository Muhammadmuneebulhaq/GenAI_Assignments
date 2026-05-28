"""
Visualization and generation utilities for VAE
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_reconstructions(vae_model, test_images, num_samples=10, save_path='reconstructions.png'):
    """
    Visualize original vs reconstructed images.
    
    Args:
        vae_model: VAE model
        test_images: Test images
        num_samples: Number of samples to visualize
        save_path: Path to save the plot
    """
    images = test_images[:num_samples]
    x_recon = vae_model(images, training=False).numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*2, 4))
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(images[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_title('Original', fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(x_recon[i].reshape(28, 28), cmap='gray')
        axes[1, i].set_title('Reconstructed', fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle('Original vs Reconstructed Images', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Reconstruction visualization saved to {save_path}")
    plt.close()


def generate_samples(decoder_model, latent_dim, num_samples=16, save_path='generated_samples.png'):
    """
    Generate new samples from random latent vectors.
    
    Args:
        decoder_model: Decoder model
        latent_dim: Latent dimension
        num_samples: Number of samples to generate
        save_path: Path to save the plot
    """
    z = tf.random.normal((num_samples, latent_dim))
    samples = decoder_model(z, training=False).numpy()
    
    # Create grid
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if grid_size == 1:
                ax = axes
            elif grid_size > 1:
                ax = axes[i, j]
            
            ax.imshow(samples[idx].reshape(28, 28), cmap='gray')
            ax.axis('off')
    
    plt.suptitle('Generated Samples from Standard Normal Distribution', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Generated samples saved to {save_path}")
    plt.close()


def visualize_latent_space(encoder_model, test_images, test_labels, save_path='latent_space.png'):
    """
    Visualize 2D latent space using t-SNE.
    
    Args:
        encoder_model: Encoder model
        test_images: Test images
        test_labels: Test labels
        save_path: Path to save the plot
    """
    from sklearn.manifold import TSNE
    
    # Collect latent representations
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(32)
    all_z = []
    
    for images in test_dataset:
        mu, _ = encoder_model(images, training=False)
        all_z.append(mu.numpy())
    
    all_z = np.concatenate(all_z, axis=0)
    
    # Apply t-SNE if latent dim > 2
    if all_z.shape[1] > 2:
        print("Applying t-SNE for visualization...")
        tsne = TSNE(n_components=2, random_state=42, verbose=0)
        all_z = tsne.fit_transform(all_z)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(all_z[:, 0], all_z[:, 1], c=test_labels, cmap='tab10', 
                        alpha=0.6, s=20, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('Z[0]')
    ax.set_ylabel('Z[1]')
    ax.set_title('Latent Space Visualization')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Class Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Latent space visualization saved to {save_path}")
    plt.close()


def interpolate_in_latent_space(encoder_model, decoder_model, test_images, num_steps=10, save_path='interpolation.png'):
    """
    Generate interpolation between two random images in latent space.
    
    Args:
        encoder_model: Encoder model
        decoder_model: Decoder model
        test_images: Test images
        num_steps: Number of interpolation steps
        save_path: Path to save the plot
    """
    img1 = test_images[0:1]
    img2 = test_images[1:2]
    
    # Encode them
    mu1, _ = encoder_model(img1, training=False)
    mu2, _ = encoder_model(img2, training=False)
    
    # Interpolate
    interpolated_latents = []
    for t in np.linspace(0, 1, num_steps):
        z_t = (1 - t) * mu1 + t * mu2
        interpolated_latents.append(z_t)
    
    # Decode
    samples = []
    for z in interpolated_latents:
        sample = decoder_model(z, training=False).numpy()
        samples.append(sample)
    
    samples = np.array(samples)
    
    # Plot
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps*2, 2))
    
    for i in range(num_steps):
        if num_steps == 1:
            ax = axes
        else:
            ax = axes[i]
        
        ax.imshow(samples[i, 0].reshape(28, 28), cmap='gray')
        ax.set_title(f't={i/(num_steps-1):.2f}', fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Interpolation in Latent Space', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Interpolation visualization saved to {save_path}")
    plt.close()


def visualize_latent_dimensionality_effect(models_dict, encoders_dict, test_images, save_path='latent_dim_comparison.png'):
    """
    Compare reconstructions from VAEs with different latent dimensions.
    
    Args:
        models_dict: Dictionary of {latent_dim: vae_model}
        encoders_dict: Dictionary of {latent_dim: encoder_model}
        test_images: Test images
        save_path: Path to save the plot
    """
    images = test_images[:5]
    
    latent_dims = sorted(models_dict.keys())
    num_samples = len(images)
    
    fig, axes = plt.subplots(len(latent_dims) + 1, num_samples, figsize=(num_samples*2, (len(latent_dims)+1)*2))
    
    # Original images
    for j in range(num_samples):
        axes[0, j].imshow(images[j].reshape(28, 28), cmap='gray')
        axes[0, j].set_title(f'Sample {j+1}', fontsize=10)
        axes[0, j].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=12, fontweight='bold')
    
    # Reconstructions for each latent dimension
    for i, latent_dim in enumerate(latent_dims):
        model = models_dict[latent_dim]
        x_recon = model(images, training=False).numpy()
        
        for j in range(num_samples):
            axes[i+1, j].imshow(x_recon[j].reshape(28, 28), cmap='gray')
            axes[i+1, j].axis('off')
        
        axes[i+1, 0].set_ylabel(f'Latent Dim={latent_dim}', fontsize=12, fontweight='bold')
    
    plt.suptitle('Impact of Latent Dimensionality on Reconstruction Quality', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Latent dimensionality comparison saved to {save_path}")
    plt.close()

