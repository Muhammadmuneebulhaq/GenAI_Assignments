"""
Data loading and preprocessing for Fashion-MNIST using TensorFlow
"""

import os
import gzip
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_mnist_data(path, kind='train'):
    """
    Load MNIST/Fashion-MNIST data from binary files.
    
    Args:
        path: Path to data directory
        kind: 'train' or 't10k' for test data
    
    Returns:
        images: numpy array of shape (num_samples, 784)
        labels: numpy array of shape (num_samples,)
    """
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    
    return images, labels


def prepare_data(data_path, val_split=0.1, batch_size=32):
    """
    Load and prepare Fashion-MNIST dataset with train/val/test split.
    
    Args:
        data_path: Path to fashion-mnist/data/fashion
        val_split: Validation split ratio (fraction of training data)
        batch_size: Batch size for data loaders
    
    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        train_images: Training images
        val_images: Validation images
        test_images: Test images
    """
    # Load train and test data
    train_images, train_labels = load_mnist_data(data_path, kind='train')
    test_images, test_labels = load_mnist_data(data_path, kind='t10k')
    
    # Normalize to [0, 1]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    # Split training data into train and validation
    val_size = int(len(train_images) * val_split)
    
    indices = np.arange(len(train_images))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_images_split = train_images[train_indices]
    val_images = train_images[val_indices]
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images_split).batch(batch_size).shuffle(10000)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_images).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
    
    return train_dataset, val_dataset, test_dataset, train_images_split, val_images, test_images


def create_numpy_datasets(data_path, val_split=0.1):
    """
    Load Fashion-MNIST and return as numpy arrays.
    
    Args:
        data_path: Path to fashion-mnist/data/fashion
        val_split: Validation split ratio
    
    Returns:
        Tuple of (train_images, val_images, test_images, train_labels, val_labels, test_labels)
    """
    # Load data
    train_images, train_labels = load_mnist_data(data_path, kind='train')
    test_images, test_labels = load_mnist_data(data_path, kind='t10k')
    
    # Normalize
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    # Split
    val_size = int(len(train_images) * val_split)
    indices = np.arange(len(train_images))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_images_split = train_images[train_indices]
    train_labels_split = train_labels[train_indices]
    
    val_images = train_images[val_indices]
    val_labels = train_labels[val_indices]
    
    return train_images_split, val_images, test_images, train_labels_split, val_labels, test_labels
