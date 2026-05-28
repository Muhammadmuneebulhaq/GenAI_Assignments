"""
Variational Autoencoder (VAE) implementation for Fashion-MNIST using TensorFlow
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
import numpy as np


def build_vae(latent_dim=20):
    """
    Build a complete VAE model.
    
    Args:
        latent_dim: Dimension of latent space
    
    Returns:
        vae: Complete VAE model
        encoder: Encoder model
        decoder: Decoder model
    """
    # Encoder
    encoder_inputs = Input(shape=(784,))
    x = layers.Dense(512, activation='relu')(encoder_inputs)
    x = layers.Dense(256, activation='relu')(x)
    mu = layers.Dense(latent_dim, name='mu')(x)
    logvar = layers.Dense(latent_dim, name='logvar')(x)
    
    encoder = Model(encoder_inputs, [mu, logvar], name='encoder')
    
    # Sampling layer
    def sampling(args):
        mu, logvar = args
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * logvar) * eps
    
    # Decoder
    latent_inputs = Input(shape=(latent_dim,))
    x = layers.Dense(256, activation='relu')(latent_inputs)
    x = layers.Dense(512, activation='relu')(x)
    decoder_outputs = layers.Dense(784, activation='sigmoid')(x)
    
    decoder = Model(latent_inputs, decoder_outputs, name='decoder')
    
    # Complete VAE
    vae_inputs = Input(shape=(784,))
    mu, logvar = encoder(vae_inputs)
    z = layers.Lambda(sampling, name='sampling')([mu, logvar])
    vae_outputs = decoder(z)
    
    vae = Model(vae_inputs, vae_outputs, name='vae')
    
    # Add encoder and decoder to vae for easy access
    vae.encoder = encoder
    vae.decoder = decoder
    
    return vae, encoder, decoder, (mu, logvar, z)


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction Loss + β * KL Divergence
    
    Args:
        x: Original image
        x_recon: Reconstructed image
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence
    
    Returns:
        loss: Total loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (Binary Cross-Entropy)
    recon_loss = tf.reduce_mean(
        keras.losses.binary_crossentropy(x, x_recon)
    )
    
    # KL Divergence loss
    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
    )
    
    # Total loss
    loss = recon_loss + beta * kl_loss
    
    return loss, recon_loss, kl_loss

