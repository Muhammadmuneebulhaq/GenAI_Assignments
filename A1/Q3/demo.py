"""
Quick Demo Script for VAE
Shows the implementation working with minimal training
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import create_numpy_datasets
from vae_model import build_vae, vae_loss
from train_vae import train_vae, evaluate_vae, calculate_reconstruction_mse, plot_training_history
from visualization import visualize_reconstructions, generate_samples

def custom_plotting():
    """Custom visualization of results"""
    print("\n=== VAE Demonstration Complete ===\n")
    print("✓ VAE Architecture: Encoder (784→512→256→latent) + Decoder (latent→256→512→784)")
    print("✓ Loss Function: Reconstruction (BCE) + KL Divergence") 
    print("✓ Reparameterization: z = μ + σ * ε")
    print("\nKey Features Implemented:")
    print("  1. Dataset Preparation (Fashion-MNIST)")
    print("  2. Complete VAE Architecture")
    print("  3. Loss Function with KL Divergence")
    print("  4. Training Loop with Early Stopping")
    print("  5. Evaluation Metrics (Loss, MSE)")
    print("  6. Image Generation from Latent Space")
    print("  7. Visualization Tools")
    print("\nResults saved to demo_results/")

def main():
    print("="*80)
    print("VAE DEMO FOR FASHION-MNIST")
    print("="*80)
    
    # Configuration
    DATA_PATH = 'fashion-mnist/data/fashion'
    OUTPUT_DIR = 'demo_results'
    LATENT_DIM = 20
    NUM_EPOCHS = 3  # Very short for demo
    BATCH_SIZE = 256  # Larger batches for speed
    
    print(f"\nConfiguration:")
    print(f"  Latent Dimension: {LATENT_DIM}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*80)
    print("Loading Fashion-MNIST data...")
    print("="*80)
    
    train_images, val_images, test_images, _, _, test_labels = create_numpy_datasets(
        DATA_PATH, val_split=0.1
    )
    
    print(f"✓ Train samples: {len(train_images)}")
    print(f"✓ Val samples: {len(val_images)}")
    print(f"✓ Test samples: {len(test_images)}")
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE).shuffle(5000)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_images).batch(BATCH_SIZE)
    
    # Build and train VAE
    print("\n" + "="*80)
    print(f"Building VAE with latent_dim={LATENT_DIM}...")
    print("="*80)
    
    history, vae_model, encoder, decoder = train_vae(
        LATENT_DIM,
        train_dataset,
        val_dataset,
        num_epochs=NUM_EPOCHS,
        learning_rate=1e-3,
        beta=1.0,
        checkpoint_dir=f'{OUTPUT_DIR}/checkpoints',
        verbose=True
    )
    
    print(f"\n✓ VAE Model Summary:")
    print(f"  - Encoder: 784 → 512 → 256 → {LATENT_DIM}")
    print(f"  - Decoder: {LATENT_DIM} → 256 → 512 → 784")
    
    # Evaluate
    print("\n" + "="*80)
    print("Evaluating on test set...")
    print("="*80)
    
    test_metrics = evaluate_vae(vae_model, encoder, test_images[:1000], batch_size=256, beta=1.0)
    mse = calculate_reconstruction_mse(vae_model, test_images[:1000], batch_size=256)
    
    print(f"\nTest Metrics:")
    print(f"  Total Loss: {test_metrics['test_loss']:.4f}")
    print(f"  Reconstruction Loss: {test_metrics['test_recon']:.4f}")
    print(f"  KL Divergence Loss: {test_metrics['test_kl']:.4f}")
    print(f"  Reconstruction MSE: {mse:.6f}")
    
    # Visualizations
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)
    
    print("\n1. Reconstruction comparison")
    visualize_reconstructions(vae_model, test_images[:10], num_samples=10,
                             save_path=f'{OUTPUT_DIR}/reconstructions.png')
    
    print("2. Random generation")
    generate_samples(decoder, LATENT_DIM, num_samples=16,
                    save_path=f'{OUTPUT_DIR}/generated_samples.png')
    
    print("3. Training history")
    plot_training_history(history, f'{OUTPUT_DIR}/training_history.png')
    
    # Create summary visualization
    print("4. Summary analysis")
    create_summary_plot(history, OUTPUT_DIR, LATENT_DIM, test_metrics, mse)
    
    custom_plotting()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in '{OUTPUT_DIR}/':")
    print("  - reconstructions.png")
    print("  - generated_samples.png")
    print("  - training_history.png")
    print("  - summary.png")
    print(f"\nFor full experimental study with 5 latent dimensions and 30 epochs:")
    print("  Run: python main.py")


def create_summary_plot(history, output_dir, latent_dim, test_metrics, mse):
    """Create a comprehensive summary visualization"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Training logs
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(history['train_loss'], 'b-', label='Train Total Loss', linewidth=2)
    ax1.plot(history['val_loss'], 'r-', label='Val Total Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Reconstruction loss
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(history['train_recon'], 'b-', label='Train', linewidth=2)
    ax2.plot(history['val_recon'], 'r-', label='Val', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Reconstruction Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # KL loss
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(history['train_kl'], 'b-', label='Train', linewidth=2)
    ax3.plot(history['val_kl'], 'r-', label='Val', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('KL Divergence Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Metrics summary
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    metrics_text = f"""VAE Summary
    
Latent Dimension: {latent_dim}
Epochs Trained: {len(history['train_loss'])}

Test Metrics:
  Total Loss: {test_metrics['test_loss']:.4f}
  Recon Loss: {test_metrics['test_recon']:.4f}
  KL Loss: {test_metrics['test_kl']:.4f}
  MSE: {mse:.6f}
  
Architecture:
  Encoder: 784→512→256→{latent_dim}
  Decoder: {latent_dim}→256→512→784
  
Loss Function:
  L = L_recon + β*L_KL
  β = 1.0
"""
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Architecture diagram text
    ax5 = fig.add_subplot(gs[0, 2])
    ax5.axis('off')
    arch_text = """VAE Architecture

ENCODER:
  Input (784)
    ↓
  Dense(512, ReLU)
    ↓
  Dense(256, ReLU)
    ↓
  μ: Dense(20)
  σ²: Dense(20)
    ↓
  Reparameterize
  z = μ + σ*ε
    ↓
DECODER:
  Dense(256, ReLU)
    ↓
  Dense(512, ReLU)
    ↓
  Output(784, Sigmoid)
"""
    ax5.text(0.05, 0.95, arch_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Loss function
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    loss_text = """VAE Loss Function

L_total = L_reconstruction + β * L_KL

L_reconstruction = BCE(x, x_recon)  =  -Σ [x*log(x_recon) + (1-x)*log(1-x_recon)]

L_KL = -0.5 * Σ [1 + log(σ²) - μ² - σ²]  (KL divergence from N(0,1))

This loss ensures:
  • Good reconstruction quality (BCE term)
  • Meaningful latent space (KL term prevents posterior collapse)
  • Smooth interpolation in latent space
  • Diversity in generated samples
"""
    ax6.text(0.05, 0.95, loss_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.suptitle('Variational Autoencoder (VAE) Summary - Fashion-MNIST', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(f'{output_dir}/summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Summary saved to {output_dir}/summary.png")
    plt.close()


if __name__ == '__main__':
    main()
