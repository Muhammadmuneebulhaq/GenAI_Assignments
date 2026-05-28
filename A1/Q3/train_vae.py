"""
Training script for VAE on Fashion-MNIST using TensorFlow
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from vae_model import build_vae, vae_loss


def train_vae(latent_dim, train_dataset, val_dataset, num_epochs=50, learning_rate=1e-3, 
              beta=1.0, checkpoint_dir='checkpoints', verbose=True):
    """
    Train VAE model.
    
    Args:
        latent_dim: Latent dimension
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        beta: Weight for KL divergence
        checkpoint_dir: Directory to save checkpoints
        verbose: Whether to print training progress
    
    Returns:
        history: Dictionary containing training history
        vae_model: Trained VAE model
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Build model
    vae_model, encoder, decoder, latent_vars = build_vae(latent_dim)
    mu, logvar, z = latent_vars
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    history = {
        'train_loss': [],
        'train_recon': [],
        'train_kl': [],
        'val_loss': [],
        'val_recon': [],
        'val_kl': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15
    best_weights = None
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss_epoch = 0
        train_recon_epoch = 0
        train_kl_epoch = 0
        num_batches = 0
        
        for images in train_dataset:
            with tf.GradientTape() as tape:
                x_recon = vae_model(images, training=True)
                mu_val, logvar_val = encoder(images, training=True)
                loss, recon_loss, kl_loss = vae_loss(images, x_recon, mu_val, logvar_val, beta=beta)
            
            grads = tape.gradient(loss, vae_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, vae_model.trainable_variables))
            
            train_loss_epoch += loss.numpy()
            train_recon_epoch += recon_loss.numpy()
            train_kl_epoch += kl_loss.numpy()
            num_batches += 1
        
        train_loss_avg = train_loss_epoch / num_batches
        train_recon_avg = train_recon_epoch / num_batches
        train_kl_avg = train_kl_epoch / num_batches
        
        # Validation phase
        val_loss_epoch = 0
        val_recon_epoch = 0
        val_kl_epoch = 0
        num_val_batches = 0
        
        for images in val_dataset:
            x_recon = vae_model(images, training=False)
            mu_val, logvar_val = encoder(images, training=False)
            loss, recon_loss, kl_loss = vae_loss(images, x_recon, mu_val, logvar_val, beta=beta)
            
            val_loss_epoch += loss.numpy()
            val_recon_epoch += recon_loss.numpy()
            val_kl_epoch += kl_loss.numpy()
            num_val_batches += 1
        
        val_loss_avg = val_loss_epoch / num_val_batches
        val_recon_avg = val_recon_epoch / num_val_batches
        val_kl_avg = val_kl_epoch / num_val_batches
        
        # Update history
        history['train_loss'].append(train_loss_avg)
        history['train_recon'].append(train_recon_avg)
        history['train_kl'].append(train_kl_avg)
        history['val_loss'].append(val_loss_avg)
        history['val_recon'].append(val_recon_avg)
        history['val_kl'].append(val_kl_avg)
        
        # Early stopping
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            best_weights = vae_model.get_weights()
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - Best model saved (val_loss: {val_loss_avg:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss_avg:.4f} (Recon: {train_recon_avg:.4f}, KL: {train_kl_avg:.4f}) | "
                  f"Val Loss: {val_loss_avg:.4f} (Recon: {val_recon_avg:.4f}, KL: {val_kl_avg:.4f})")
    
    # Load best weights
    if best_weights is not None:
        vae_model.set_weights(best_weights)
    
    return history, vae_model, encoder, decoder


def evaluate_vae(vae_model, encoder, test_images, batch_size=32, beta=1.0):
    """
    Evaluate VAE on test set.
    
    Args:
        vae_model: VAE model instance
        encoder: Encoder model
        test_images: Test images
        batch_size: Batch size
        beta: Weight for KL divergence
    
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
    
    test_loss = 0
    test_recon = 0
    test_kl = 0
    num_batches = 0
    
    for images in test_dataset:
        x_recon = vae_model(images, training=False)
        mu_val, logvar_val = encoder(images, training=False)
        loss, recon_loss, kl_loss = vae_loss(images, x_recon, mu_val, logvar_val, beta=beta)
        
        test_loss += loss.numpy()
        test_recon += recon_loss.numpy()
        test_kl += kl_loss.numpy()
        num_batches += 1
    
    metrics = {
        'test_loss': test_loss / num_batches,
        'test_recon': test_recon / num_batches,
        'test_kl': test_kl / num_batches
    }
    
    return metrics


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(history['train_recon'], label='Train', linewidth=2)
    axes[1].plot(history['val_recon'], label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL divergence loss
    axes[2].plot(history['train_kl'], label='Train', linewidth=2)
    axes[2].plot(history['val_kl'], label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('KL Divergence Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def calculate_reconstruction_mse(vae_model, test_images, batch_size=32):
    """
    Calculate mean squared error for reconstructions.
    
    Args:
        vae_model: VAE model
        test_images: Test images
        batch_size: Batch size
    
    Returns:
        mse: Mean squared error
    """
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
    
    total_mse = 0
    num_batches = 0
    
    for images in test_dataset:
        x_recon = vae_model(images, training=False)
        mse = tf.reduce_mean((images - x_recon) ** 2).numpy()
        total_mse += mse
        num_batches += 1
    
    return total_mse / num_batches


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(history['train_recon'], label='Train', linewidth=2)
    axes[1].plot(history['val_recon'], label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL divergence loss
    axes[2].plot(history['train_kl'], label='Train', linewidth=2)
    axes[2].plot(history['val_kl'], label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('KL Divergence Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def calculate_reconstruction_mse(vae_model, test_images, batch_size=32):
    """
    Calculate mean squared error for reconstructions.
    
    Args:
        vae_model: VAE model
        test_images: Test images
        batch_size: Batch size
    
    Returns:
        mse: Mean squared error
    """
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
    
    total_mse = 0
    num_batches = 0
    
    for images in test_dataset:
        x_recon, _, _, _ = vae_model(images, training=False)
        mse = tf.reduce_mean((images - x_recon) ** 2).numpy()
        total_mse += mse
        num_batches += 1
    
    return total_mse / num_batches
