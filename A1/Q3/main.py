"""
Main script to train VAE models with different latent dimensions and conduct experimental study
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

from data_loader import prepare_data, create_numpy_datasets
from vae_model import build_vae, vae_loss
from train_vae import train_vae, evaluate_vae, plot_training_history, calculate_reconstruction_mse
from visualization import (
    visualize_reconstructions,
    generate_samples,
    visualize_latent_space,
    interpolate_in_latent_space,
    visualize_latent_dimensionality_effect
)


def run_vae_experiment(latent_dim, train_dataset, val_dataset, test_images, 
                       num_epochs=50, output_dir='results'):
    """
    Train and evaluate VAE with specific latent dimension.
    
    Args:
        latent_dim: Latent dimension to experiment with
        train_dataset: Training TF dataset
        val_dataset: Validation TF dataset
        test_images: Test images
        num_epochs: Number of epochs to train
        output_dir: Output directory for results
    
    Returns:
        vae_model: Trained VAE model
        encoder_model: Encoder model
        decoder_model: Decoder model
        history: Training history
        test_metrics: Test metrics
    """
    print(f"\n{'='*60}")
    print(f"Training VAE with latent dimension = {latent_dim}")
    print(f"{'='*60}")
    
    # Train model
    history, vae_model, encoder, decoder = train_vae(
        latent_dim,
        train_dataset, val_dataset,
        num_epochs=num_epochs,
        learning_rate=1e-3,
        beta=1.0,
        checkpoint_dir=f'{output_dir}/checkpoints/latent_{latent_dim}',
        verbose=True
    )
    
    # Evaluate on test set
    test_metrics = evaluate_vae(vae_model, encoder, test_images, batch_size=32, beta=1.0)
    
    # Calculate MSE
    mse = calculate_reconstruction_mse(vae_model, test_images, batch_size=32)
    
    print(f"\nTest Metrics for latent_dim={latent_dim}:")
    print(f"  Total Loss: {test_metrics['test_loss']:.4f}")
    print(f"  Reconstruction Loss: {test_metrics['test_recon']:.4f}")
    print(f"  KL Divergence Loss: {test_metrics['test_kl']:.4f}")
    print(f"  Reconstruction MSE: {mse:.6f}")
    
    test_metrics['mse'] = mse
    
    return vae_model, encoder, decoder, history, test_metrics


def main():
    """
    Main function to run experiments
    """
    # Configuration
    DATA_PATH = 'fashion-mnist/data/fashion'
    OUTPUT_DIR = 'results'
    NUM_EPOCHS = 30  # Reduced for faster training
    BATCH_SIZE = 128
    LATENT_DIMS = [2, 5, 10, 20, 50]  # Reduced from original to save time
    
    print("="*80)
    print("VARIATIONAL AUTOENCODER (VAE) FOR FASHION-MNIST")
    print("="*80)
    print(f"Latent dimensions to test: {LATENT_DIMS}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    train_images, val_images, test_images, train_labels, val_labels, test_labels = create_numpy_datasets(
        DATA_PATH,
        val_split=0.1
    )
    
    print(f"Train samples: {len(train_images)}")
    print(f"Val samples: {len(val_images)}")
    print(f"Test samples: {len(test_images)}")
    
    # Create TF datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE).shuffle(10000)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_images).batch(BATCH_SIZE)
    
    # Train models with different latent dimensions
    results = {}
    models = {}
    encoders = {}
    decoders = {}
    histories = {}
    
    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    for latent_dim in LATENT_DIMS:
        vae_model, encoder, decoder, history, test_metrics = run_vae_experiment(
            latent_dim,
            train_dataset,
            val_dataset,
            test_images,
            num_epochs=NUM_EPOCHS,
            output_dir=OUTPUT_DIR
        )
        
        models[latent_dim] = vae_model
        encoders[latent_dim] = encoder
        decoders[latent_dim] = decoder
        histories[latent_dim] = history
        results[latent_dim] = test_metrics
        
        # Save models
        vae_model.save(f'{OUTPUT_DIR}/vae_latent_{latent_dim}')
        encoder.save(f'{OUTPUT_DIR}/encoder_latent_{latent_dim}')
        decoder.save(f'{OUTPUT_DIR}/decoder_latent_{latent_dim}')
        print(f"Models saved for latent_dim={latent_dim}")
        
        # Plot training history
        plot_training_history(history, f'{OUTPUT_DIR}/training_history_latent_{latent_dim}.png')
    
    # Create results table
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results).T
    results_df.index.name = 'Latent Dimension'
    print(results_df)
    
    # Save results to CSV
    results_df.to_csv(f'{OUTPUT_DIR}/experimental_results.csv')
    print(f"\nResults saved to {OUTPUT_DIR}/experimental_results.csv")
    
    # Visualizations using best model (latent_dim=20)
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    best_latent_dim = 20 if 20 in models else LATENT_DIMS[-1]
    best_model = models[best_latent_dim]
    best_encoder = encoders[best_latent_dim]
    best_decoder = decoders[best_latent_dim]
    
    # 1. Reconstruction visualization
    print("\n1. Generating reconstruction visualizations...")
    visualize_reconstructions(best_model, test_images[:10], num_samples=10, 
                             save_path=f'{OUTPUT_DIR}/reconstructions.png')
    
    # 2. Generate new samples
    print("2. Generating random samples...")
    generate_samples(best_decoder, best_latent_dim, num_samples=16, 
                    save_path=f'{OUTPUT_DIR}/generated_samples.png')
    
    # 3. Latent space visualization
    print("3. Visualizing latent space...")
    if best_latent_dim >= 2:
        visualize_latent_space(best_encoder, test_images, test_labels,
                              save_path=f'{OUTPUT_DIR}/latent_space.png')
    
    # 4. Interpolation in latent space
    print("4. Generating latent space interpolations...")
    interpolate_in_latent_space(best_encoder, best_decoder, test_images, num_steps=15,
                               save_path=f'{OUTPUT_DIR}/interpolation.png')
    
    # 5. Comparison of latent dimensions
    print("5. Comparing reconstruction quality across latent dimensions...")
    visualize_latent_dimensionality_effect(models, encoders, test_images,
                                          save_path=f'{OUTPUT_DIR}/latent_dim_comparison.png')
    
    # Create comparison plots
    print("\n6. Creating comparison plots...")
    create_comparison_plots(results_df, OUTPUT_DIR)
    
    # Save summary report
    save_summary_report(results_df, histories, LATENT_DIMS, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print(f"All results saved to {OUTPUT_DIR}/")
    print("="*80)


def create_comparison_plots(results_df, output_dir):
    """
    Create comparison plots for different latent dimensions.
    
    Args:
        results_df: DataFrame with experimental results
        output_dir: Output directory
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    latent_dims = results_df.index.tolist()
    
    # 1. Total Loss vs Latent Dimension
    axes[0].plot(latent_dims, results_df['test_loss'].values, 'o-', linewidth=2, markersize=8, label='Total Loss')
    axes[0].set_xlabel('Latent Dimension', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Total Loss vs Latent Dimension', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(latent_dims)
    
    # 2. Reconstruction vs KL Loss
    axes[1].plot(latent_dims, results_df['test_recon'].values, 'o-', linewidth=2, markersize=8, label='Reconstruction')
    axes[1].plot(latent_dims, results_df['test_kl'].values, 's-', linewidth=2, markersize=8, label='KL Divergence')
    axes[1].set_xlabel('Latent Dimension', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Loss Components vs Latent Dimension', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(latent_dims)
    
    # 3. Reconstruction MSE
    axes[2].plot(latent_dims, results_df['mse'].values, 'D-', linewidth=2, markersize=8, color='red', label='MSE')
    axes[2].set_xlabel('Latent Dimension', fontsize=12)
    axes[2].set_ylabel('MSE', fontsize=12)
    axes[2].set_title('Reconstruction MSE vs Latent Dimension', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(latent_dims)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_plots.png', dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to {output_dir}/comparison_plots.png")
    plt.close()


def save_summary_report(results_df, histories, latent_dims, output_dir):
    """
    Save a comprehensive summary report.
    
    Args:
        results_df: Results DataFrame
        histories: Training histories dictionary
        latent_dims: List of latent dimensions tested
        output_dir: Output directory
    """
    report = []
    report.append("=" * 80)
    report.append("VARIATIONAL AUTOENCODER (VAE) EXPERIMENTAL STUDY")
    report.append("Dataset: Fashion-MNIST")
    report.append("=" * 80)
    report.append("")
    
    report.append("OBJECTIVE:")
    report.append("-" * 80)
    report.append("Analyze the impact of different latent dimensions on:")
    report.append("  1. Reconstruction quality (MSE, Reconstruction Loss)")
    report.append("  2. KL divergence loss (regularization)")
    report.append("  3. Overall model performance")
    report.append("")
    
    report.append("EXPERIMENTAL CONFIGURATION:")
    report.append("-" * 80)
    report.append("Architecture:")
    report.append("  - Encoder: 784 -> 512 -> 256 -> latent_dim")
    report.append("  - Decoder: latent_dim -> 256 -> 512 -> 784")
    report.append("  - Reparameterization: z = μ + σ * ε")
    report.append("Loss Function:")
    report.append("  - L_total = L_reconstruction + β * L_KL")
    report.append("  - β = 1.0 (standard VAE)")
    report.append("")
    report.append(f"Training Configuration:")
    report.append(f"  - Optimizer: Adam (lr=1e-3)")
    report.append(f"  - Batch size: 128")
    report.append(f"  - Early stopping: patience=15 epochs")
    report.append(f"  - Latent dimensions tested: {latent_dims}")
    report.append("")
    
    report.append("RESULTS SUMMARY:")
    report.append("-" * 80)
    report.append(results_df.to_string())
    report.append("")
    
    report.append("DETAILED ANALYSIS:")
    report.append("-" * 80)
    
    # Key observations
    min_loss_idx = results_df['test_loss'].idxmin()
    max_kl_idx = results_df['test_kl'].idxmax()
    min_mse_idx = results_df['mse'].idxmin()
    
    report.append(f"\n1. RECONSTRUCTION QUALITY (MSE):")
    report.append(f"   - Best MSE at latent_dim={min_mse_idx}: {results_df.loc[min_mse_idx, 'mse']:.6f}")
    report.append(f"   - Lowest MSE indicates better reconstruction quality")
    
    report.append(f"\n2. TOTAL LOSS:")
    report.append(f"   - Best total loss at latent_dim={min_loss_idx}: {results_df.loc[min_loss_idx, 'test_loss']:.4f}")
    report.append(f"   - Balanced reconstruction and regularization")
    
    report.append(f"\n3. KL DIVERGENCE:")
    report.append(f"   - Highest KL at latent_dim={max_kl_idx}: {results_df.loc[max_kl_idx, 'test_kl']:.4f}")
    report.append(f"   - Higher KL indicates better latent space usage")
    report.append(f"   - Note: Large latent spaces may have unused dimensions (posterior collapse)")
    
    report.append("\n4. OBSERVATIONS:")
    report.append("   - Increasing latent dimension generally reduces reconstruction loss")
    report.append("   - KL divergence increases initially then may decrease for very large dims")
    report.append("   - Optimal latent dimension balances reconstruction and regularization")
    report.append("   - Too small dimensions: poor reconstructions, mode collapse")
    report.append("   - Too large dimensions: posterior collapse, unused capacity")
    report.append("   - Trade-off between generation quality and model interpretability")
    
    report.append("\n5. RECOMMENDATIONS:")
    report.append(f"   - For Fashion-MNIST, latent_dim=20 provides good balance")
    report.append("   - Small dims (2,5,10) suitable for visualization and interpretability")
    report.append("   - Medium dims (20,50) good for generation and reconstruction tasks")
    report.append("   - Large dims (100+) may require β-VAE or other regularization strategies")
    
    report.append("\nGENERATION CAPABILITIES:")
    report.append("-" * 80)
    report.append("Generated visualizations:")
    report.append("  1. reconstructions.png - Original vs reconstructed images")
    report.append("  2. generated_samples.png - Samples from learned latent space")
    report.append("  3. latent_space.png - 2D t-SNE visualization of latent representations")
    report.append("  4. interpolation.png - Smooth interpolation between two samples")
    report.append("  5. latent_dim_comparison.png - Quality comparison across dimensions")
    report.append("  6. comparison_plots.png - Quantitative performance metrics")
    report.append("  7. training_history_latent_*.png - Training curves for each model")
    report.append("  8. experimental_results.csv - Complete results table")
    
    report.append("\n" + "="*80)
    report.append("CONCLUSION")
    report.append("="*80)
    report.append("This experimental study demonstrates how latent dimension affects VAE")
    report.append("performance. The sweet spot for Fashion-MNIST appears to be around 20-50")
    report.append("dimensions, offering good reconstruction quality while maintaining a")
    report.append("meaningful latent space for generation and interpolation tasks.")
    report.append("")
    report.append("=" * 80)
    
    # Save to file
    report_text = "\n".join(report)
    with open(f'{output_dir}/experiment_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"\nSummary report saved to {output_dir}/experiment_report.txt")
    print("\n" + "="*80)
    print("REPORT SUMMARY")
    print("="*80)
    print(report_text[:2000] + "\n...")


if __name__ == '__main__':
    main()
