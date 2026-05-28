"""
CycleGAN Implementation for Person Face Sketches
=================================================
Based on: "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
           by Zhu et al. (2017)

This script implements CycleGAN to perform bidirectional translation between:
  - Real face photos -> Sketches  (Generator G: Photo → Sketch)
  - Sketches -> Real face photos  (Generator F: Sketch → Photo)

Dataset: Person Face Sketches (Kaggle)
  - data/train/photos & data/train/sketches
  - data/val/photos   & data/val/sketches
  - data/test/photos  & data/test/sketches

Usage:
  Training:   python cyclegan.py --mode train
  Testing:    python cyclegan.py --mode test
  Inference:  python cyclegan.py --mode infer --input <image_path>
"""

import os
import sys
import argparse
import random
import itertools
from collections import deque

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils


# ============================================================================
# Configuration / Hyperparameters
# ============================================================================

class Config:
    """Central configuration for all hyperparameters."""
    # Data
    data_root = "data"
    img_size = 256                  # Resize images to 256×256
    max_samples = None              # Max samples per domain (None = use all). Set to e.g., 100 to limit
    
    # Training
    batch_size = 1                  # CycleGAN typically uses batch_size=1
    num_epochs = 50                 # Total training epochs (reduced from 200)
    lr = 0.0002                     # Initial learning rate (Adam)
    beta1 = 0.5                     # Adam beta1
    beta2 = 0.999                   # Adam beta2
    lambda_cycle = 10.0             # Cycle-consistency loss weight
    lambda_identity = 5.0           # Identity loss weight (0.5 * lambda_cycle)
    decay_epoch = 25                # Epoch from which to start lr decay (adjusted proportionally)
    
    # Model
    n_residual_blocks = 9           # Number of ResNet blocks in generator
    ngf = 64                        # Generator base filter count
    ndf = 64                        # Discriminator base filter count
    
    # Replay Buffer
    buffer_size = 50                # Size of the image replay buffer
    
    # Checkpoints
    checkpoint_dir = "checkpoints"
    save_every = 5                  # Save checkpoint every N epochs
    sample_dir = "samples"          # Directory to save sample outputs
    
    # Workers
    num_workers = 4
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Dataset
# ============================================================================

class FaceSketchDataset(Dataset):
    """
    Dataset for loading paired face photos and sketches.
    
    Even though CycleGAN is designed for unpaired data, the dataset happens 
    to have paired images. We load them as separate domains (photos and sketches)
    and shuffle independently during training to simulate the unpaired setting.
    """
    
    def __init__(self, root_dir, split="train", transform=None, max_samples=None):
        """
        Args:
            root_dir (str): Root data directory (e.g., "data").
            split (str): One of "train", "val", or "test".
            transform: Torchvision transforms to apply.
            max_samples (int or None): Maximum number of samples per domain. If None, use all.
        """
        self.photo_dir = os.path.join(root_dir, split, "photos")
        self.sketch_dir = os.path.join(root_dir, split, "sketches")
        
        self.photo_files = sorted(os.listdir(self.photo_dir))
        self.sketch_files = sorted(os.listdir(self.sketch_dir))
        
        # Limit samples if specified
        if max_samples is not None:
            self.photo_files = self.photo_files[:max_samples]
            self.sketch_files = self.sketch_files[:max_samples]
        
        self.transform = transform
        
        # For unpaired training, we independently sample from each domain
        self.len_photos = len(self.photo_files)
        self.len_sketches = len(self.sketch_files)
    
    def __len__(self):
        return max(self.len_photos, self.len_sketches)
    
    def __getitem__(self, idx):
        # Wrap around if domains have different sizes
        photo_path = os.path.join(self.photo_dir, self.photo_files[idx % self.len_photos])
        sketch_path = os.path.join(self.sketch_dir, self.sketch_files[idx % self.len_sketches])
        
        photo = Image.open(photo_path).convert("RGB")
        sketch = Image.open(sketch_path).convert("RGB")
        
        if self.transform:
            photo = self.transform(photo)
            sketch = self.transform(sketch)
        
        return {"photo": photo, "sketch": sketch}


def get_transforms(img_size, is_train=True):
    """
    Build data augmentation / preprocessing transforms.
    
    For training: resize, random crop, random horizontal flip, normalize.
    For testing:  resize, center crop, normalize.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.12), transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


# ============================================================================
# Generator (ResNet-based)
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Residual block with two 3×3 convolutions, instance normalization, and ReLU.
    Uses reflect padding to reduce boundary artifacts.
    """
    
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
        )
    
    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    ResNet-based Generator following the CycleGAN paper architecture.
    
    Architecture:
      1. Initial convolution block (7×7, reflect-padded)
      2. Downsampling (2 stride-2 convolutions)
      3. Residual blocks (default: 9 blocks for 256×256)
      4. Upsampling (2 transposed convolutions)
      5. Output convolution (7×7, reflect-padded, tanh activation)
    """
    
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_residual_blocks=9):
        super().__init__()
        
        # --- Initial convolution block ---
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        
        # --- Downsampling ---
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]
        
        # --- Residual blocks ---
        mult = 2 ** n_downsampling
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(ngf * mult)]
        
        # --- Upsampling ---
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2,
                                   kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]
        
        # --- Output layer ---
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7),
            nn.Tanh(),
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# Discriminator (PatchGAN / 70×70 PatchGAN)
# ============================================================================

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator (70×70 receptive field).
    
    Classifies overlapping 70×70 patches of the image as real or fake,
    producing a 2D map of predictions rather than a single scalar.
    Uses Instance Normalization and LeakyReLU.
    """
    
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        
        def discriminator_block(in_c, out_c, normalize=True):
            """A single discriminator conv block."""
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, ndf, normalize=False),   # 128×128
            *discriminator_block(ndf, ndf * 2),                         # 64×64
            *discriminator_block(ndf * 2, ndf * 4),                     # 32×32
            nn.ZeroPad2d((1, 0, 1, 0)),                                 # Pad to maintain shape
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, padding=1, bias=False),  # 31×31
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, padding=1),           # 30×30 patch output
        )
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# Replay Buffer (for stabilizing GAN training)
# ============================================================================

class ReplayBuffer:
    """
    Replay buffer to store previously generated images.
    
    With 50% probability, returns a previously stored image instead of the
    current one, which helps stabilize discriminator training.
    From: Shrivastava et al. "Learning from Simulated and Unsupervised Images
    through Adversarial Training" (2017).
    """
    
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, data):
        result = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if random.random() > 0.5:
                    # Return a random element from buffer and replace it
                    idx = random.randint(0, self.max_size - 1)
                    tmp = self.data[idx].clone()
                    self.data[idx] = element
                    result.append(tmp)
                else:
                    result.append(element)
        return torch.cat(result, dim=0)


# ============================================================================
# Weight Initialization
# ============================================================================

def init_weights(m):
    """
    Initialize network weights using Normal distribution (mean=0, std=0.02),
    as specified in the original CycleGAN paper.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1 or classname.find("InstanceNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


# ============================================================================
# Learning Rate Scheduler (Linear Decay)
# ============================================================================

class LambdaLR:
    """
    Linear learning rate decay.
    
    Keeps the initial learning rate for the first `decay_epoch` epochs,
    then linearly decays it to zero over the remaining epochs.
    """
    
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert n_epochs > decay_start_epoch, "Decay must start before training ends."
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
    
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )


# ============================================================================
# Training
# ============================================================================

def train(config):
    """
    Main training loop for CycleGAN.
    
    Trains two generators (G_photo2sketch, G_sketch2photo) and two 
    discriminators (D_photo, D_sketch) using:
      - Adversarial loss (MSE / LSGAN)
      - Cycle-consistency loss (L1)
      - Identity loss (L1)
    """
    print(f"[INFO] Device: {config.device}")
    print(f"[INFO] Training CycleGAN for {config.num_epochs} epochs")
    
    # --- Create directories ---
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)
    
    # --- Data ---
    train_transform = get_transforms(config.img_size, is_train=True)
    val_transform = get_transforms(config.img_size, is_train=False)
    
    train_dataset = FaceSketchDataset(config.data_root, split="train", transform=train_transform, max_samples=config.max_samples)
    val_dataset = FaceSketchDataset(config.data_root, split="val", transform=val_transform, max_samples=config.max_samples)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )
    
    print(f"[INFO] Training samples: {len(train_dataset)}")
    print(f"[INFO] Validation samples: {len(val_dataset)}")
    
    # --- Models ---
    # G_AB: Photo → Sketch, G_BA: Sketch → Photo
    G_AB = Generator(n_residual_blocks=config.n_residual_blocks, ngf=config.ngf).to(config.device)
    G_BA = Generator(n_residual_blocks=config.n_residual_blocks, ngf=config.ngf).to(config.device)
    D_A = Discriminator(ndf=config.ndf).to(config.device)   # Discriminates real photos
    D_B = Discriminator(ndf=config.ndf).to(config.device)   # Discriminates real sketches
    
    # Initialize weights
    G_AB.apply(init_weights)
    G_BA.apply(init_weights)
    D_A.apply(init_weights)
    D_B.apply(init_weights)
    
    # --- Losses ---
    criterion_GAN = nn.MSELoss()        # LSGAN loss
    criterion_cycle = nn.L1Loss()       # Cycle-consistency loss
    criterion_identity = nn.L1Loss()    # Identity loss
    
    # --- Optimizers ---
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=config.lr, betas=(config.beta1, config.beta2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    
    # --- LR Schedulers ---
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(config.num_epochs, 0, config.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(config.num_epochs, 0, config.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(config.num_epochs, 0, config.decay_epoch).step
    )
    
    # --- Replay Buffers ---
    fake_A_buffer = ReplayBuffer(config.buffer_size)
    fake_B_buffer = ReplayBuffer(config.buffer_size)
    
    # --- Resume from checkpoint if available ---
    start_epoch = 0
    latest_ckpt = os.path.join(config.checkpoint_dir, "latest.pth")
    if os.path.exists(latest_ckpt):
        print(f"[INFO] Resuming from checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=config.device)
        G_AB.load_state_dict(ckpt["G_AB"])
        G_BA.load_state_dict(ckpt["G_BA"])
        D_A.load_state_dict(ckpt["D_A"])
        D_B.load_state_dict(ckpt["D_B"])
        optimizer_G.load_state_dict(ckpt["optimizer_G"])
        optimizer_D_A.load_state_dict(ckpt["optimizer_D_A"])
        optimizer_D_B.load_state_dict(ckpt["optimizer_D_B"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[INFO] Resumed at epoch {start_epoch}")
    
    # --- Training Loop ---
    for epoch in range(start_epoch, config.num_epochs):
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()
        
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        
        for i, batch in enumerate(train_loader):
            # Domain A = Photos, Domain B = Sketches
            real_A = batch["photo"].to(config.device)
            real_B = batch["sketch"].to(config.device)
            
            # Targets for LSGAN (real=1, fake=0)
            # Shape matches discriminator output
            valid = torch.ones_like(D_A(real_A), requires_grad=False).to(config.device)
            fake = torch.zeros_like(D_A(real_A), requires_grad=False).to(config.device)
            
            # ================================================================
            # Train Generators (G_AB and G_BA)
            # ================================================================
            optimizer_G.zero_grad()
            
            # --- Identity loss ---
            # G_AB should be identity if B is fed: G_AB(B) ≈ B
            identity_B = G_AB(real_B)
            loss_identity_B = criterion_identity(identity_B, real_B) * config.lambda_identity
            # G_BA should be identity if A is fed: G_BA(A) ≈ A
            identity_A = G_BA(real_A)
            loss_identity_A = criterion_identity(identity_A, real_A) * config.lambda_identity
            
            # --- GAN loss ---
            fake_B = G_AB(real_A)        # Photo → Sketch
            pred_fake_B = D_B(fake_B)
            loss_GAN_AB = criterion_GAN(pred_fake_B, valid)
            
            fake_A = G_BA(real_B)        # Sketch → Photo
            pred_fake_A = D_A(fake_A)
            loss_GAN_BA = criterion_GAN(pred_fake_A, valid)
            
            # --- Cycle-consistency loss ---
            recovered_A = G_BA(fake_B)   # Photo → Sketch → Photo
            loss_cycle_A = criterion_cycle(recovered_A, real_A) * config.lambda_cycle
            
            recovered_B = G_AB(fake_A)   # Sketch → Photo → Sketch
            loss_cycle_B = criterion_cycle(recovered_B, real_B) * config.lambda_cycle
            
            # --- Total generator loss ---
            loss_G = (
                loss_GAN_AB + loss_GAN_BA +
                loss_cycle_A + loss_cycle_B +
                loss_identity_A + loss_identity_B
            )
            loss_G.backward()
            optimizer_G.step()
            
            # ================================================================
            # Train Discriminator A (distinguishes real photos from fake photos)
            # ================================================================
            optimizer_D_A.zero_grad()
            
            pred_real_A = D_A(real_A)
            loss_D_real_A = criterion_GAN(pred_real_A, valid)
            
            fake_A_buffered = fake_A_buffer.push_and_pop(fake_A.detach())
            pred_fake_A = D_A(fake_A_buffered)
            loss_D_fake_A = criterion_GAN(pred_fake_A, fake)
            
            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # ================================================================
            # Train Discriminator B (distinguishes real sketches from fake sketches)
            # ================================================================
            optimizer_D_B.zero_grad()
            
            pred_real_B = D_B(real_B)
            loss_D_real_B = criterion_GAN(pred_real_B, valid)
            
            fake_B_buffered = fake_B_buffer.push_and_pop(fake_B.detach())
            pred_fake_B = D_B(fake_B_buffered)
            loss_D_fake_B = criterion_GAN(pred_fake_B, fake)
            
            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()
            
            # --- Accumulate losses ---
            epoch_loss_G += loss_G.item()
            epoch_loss_D += (loss_D_A.item() + loss_D_B.item())
            
            # --- Log progress ---
            if (i + 1) % 100 == 0:
                print(
                    f"  Epoch [{epoch+1}/{config.num_epochs}] "
                    f"Batch [{i+1}/{len(train_loader)}] "
                    f"G: {loss_G.item():.4f} "
                    f"D_A: {loss_D_A.item():.4f} D_B: {loss_D_B.item():.4f} "
                    f"Cyc: {(loss_cycle_A + loss_cycle_B).item():.4f} "
                    f"Idt: {(loss_identity_A + loss_identity_B).item():.4f}"
                )
        
        # --- Epoch summary ---
        n_batches = len(train_loader)
        print(
            f"Epoch [{epoch+1}/{config.num_epochs}] "
            f"G_loss: {epoch_loss_G/n_batches:.4f} "
            f"D_loss: {epoch_loss_D/n_batches:.4f} "
            f"LR: {optimizer_G.param_groups[0]['lr']:.6f}"
        )
        
        # --- Update learning rates ---
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        
        # --- Save samples ---
        save_samples(G_AB, G_BA, val_loader, epoch, config)
        
        # --- Save checkpoint ---
        if (epoch + 1) % config.save_every == 0 or epoch == config.num_epochs - 1:
            save_checkpoint(
                G_AB, G_BA, D_A, D_B,
                optimizer_G, optimizer_D_A, optimizer_D_B,
                epoch, config
            )
    
    print("[INFO] Training complete!")


# ============================================================================
# Checkpoint Utilities
# ============================================================================

def save_checkpoint(G_AB, G_BA, D_A, D_B, opt_G, opt_D_A, opt_D_B, epoch, config):
    """Save all model weights and optimizer states."""
    ckpt = {
        "epoch": epoch,
        "G_AB": G_AB.state_dict(),
        "G_BA": G_BA.state_dict(),
        "D_A": D_A.state_dict(),
        "D_B": D_B.state_dict(),
        "optimizer_G": opt_G.state_dict(),
        "optimizer_D_A": opt_D_A.state_dict(),
        "optimizer_D_B": opt_D_B.state_dict(),
    }
    # Save epoch-specific checkpoint
    path = os.path.join(config.checkpoint_dir, f"epoch_{epoch+1}.pth")
    torch.save(ckpt, path)
    # Also save as latest
    torch.save(ckpt, os.path.join(config.checkpoint_dir, "latest.pth"))
    print(f"[INFO] Checkpoint saved: {path}")


def load_generators(config, checkpoint_path=None):
    """Load trained generators from checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.checkpoint_dir, "latest.pth")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    G_AB = Generator(n_residual_blocks=config.n_residual_blocks, ngf=config.ngf).to(config.device)
    G_BA = Generator(n_residual_blocks=config.n_residual_blocks, ngf=config.ngf).to(config.device)
    
    ckpt = torch.load(checkpoint_path, map_location=config.device)
    G_AB.load_state_dict(ckpt["G_AB"])
    G_BA.load_state_dict(ckpt["G_BA"])
    
    G_AB.eval()
    G_BA.eval()
    
    epoch = ckpt.get("epoch", "unknown")
    print(f"[INFO] Loaded generators from epoch {epoch}")
    
    return G_AB, G_BA


# ============================================================================
# Sample Generation
# ============================================================================

def save_samples(G_AB, G_BA, val_loader, epoch, config):
    """Generate and save sample translations from the validation set."""
    G_AB.eval()
    G_BA.eval()
    
    with torch.no_grad():
        batch = next(iter(val_loader))
        real_A = batch["photo"].to(config.device)
        real_B = batch["sketch"].to(config.device)
        
        fake_B = G_AB(real_A)       # Photo → Sketch
        fake_A = G_BA(real_B)       # Sketch → Photo
        recovered_A = G_BA(fake_B)  # Photo → Sketch → Photo
        recovered_B = G_AB(fake_A)  # Sketch → Photo → Sketch
        
        # Denormalize: [-1, 1] → [0, 1]
        def denorm(x):
            return (x + 1) / 2.0
        
        # Photo → Sketch → Recovered Photo
        img_grid_A = torch.cat([denorm(real_A), denorm(fake_B), denorm(recovered_A)], dim=3)
        # Sketch → Photo → Recovered Sketch
        img_grid_B = torch.cat([denorm(real_B), denorm(fake_A), denorm(recovered_B)], dim=3)
        
        # Stack both
        img_grid = torch.cat([img_grid_A, img_grid_B], dim=0)
        
        save_path = os.path.join(config.sample_dir, f"epoch_{epoch+1}.png")
        vutils.save_image(img_grid, save_path, nrow=1, normalize=False)
    
    G_AB.train()
    G_BA.train()


# ============================================================================
# Testing / Evaluation
# ============================================================================

def test(config, checkpoint_path=None):
    """
    Run evaluation on the test set.
    
    Generates translated images for all test samples and saves them
    to the output directory.
    """
    print("[INFO] Running test evaluation...")
    
    G_AB, G_BA = load_generators(config, checkpoint_path)
    
    test_transform = get_transforms(config.img_size, is_train=False)
    test_dataset = FaceSketchDataset(config.data_root, split="test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)
    
    output_dir = os.path.join("test_results")
    os.makedirs(os.path.join(output_dir, "photo_to_sketch"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "sketch_to_photo"), exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            real_A = batch["photo"].to(config.device)
            real_B = batch["sketch"].to(config.device)
            
            fake_B = G_AB(real_A)   # Photo → Sketch
            fake_A = G_BA(real_B)   # Sketch → Photo
            
            # Save results
            vutils.save_image(
                (fake_B + 1) / 2.0,
                os.path.join(output_dir, "photo_to_sketch", f"{i:04d}.png")
            )
            vutils.save_image(
                (fake_A + 1) / 2.0,
                os.path.join(output_dir, "sketch_to_photo", f"{i:04d}.png")
            )
            
            # Save comparison images (real | fake)
            if i < 20:  # Save first 20 comparisons
                comparison_AB = torch.cat([(real_A + 1) / 2.0, (fake_B + 1) / 2.0], dim=3)
                comparison_BA = torch.cat([(real_B + 1) / 2.0, (fake_A + 1) / 2.0], dim=3)
                vutils.save_image(
                    comparison_AB,
                    os.path.join(output_dir, f"compare_photo2sketch_{i:04d}.png")
                )
                vutils.save_image(
                    comparison_BA,
                    os.path.join(output_dir, f"compare_sketch2photo_{i:04d}.png")
                )
    
    print(f"[INFO] Test results saved to: {output_dir}")
    print(f"[INFO] Generated {len(test_dataset)} image translations.")


# ============================================================================
# Single Image Inference
# ============================================================================

def infer(config, image_path, checkpoint_path=None):
    """
    Perform inference on a single image.
    
    Automatically detects whether the input is a photo or sketch based on
    color statistics, then applies the appropriate generator.
    """
    print(f"[INFO] Running inference on: {image_path}")
    
    G_AB, G_BA = load_generators(config, checkpoint_path)
    
    transform = get_transforms(config.img_size, is_train=False)
    
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(config.device)
    
    # Detect input type based on color saturation
    # Sketches tend to have low saturation (mostly grayscale)
    img_np = np.array(img)
    saturation = np.std(img_np[:, :, 0].astype(float) - img_np[:, :, 1].astype(float))
    is_sketch = saturation < 15  # Low color variation → likely a sketch
    
    with torch.no_grad():
        if is_sketch:
            print("[INFO] Detected input as SKETCH → Generating Photo")
            output = G_BA(img_tensor)
            mode = "sketch_to_photo"
        else:
            print("[INFO] Detected input as PHOTO → Generating Sketch")
            output = G_AB(img_tensor)
            mode = "photo_to_sketch"
    
    # Save output
    output_path = os.path.splitext(image_path)[0] + f"_{mode}.png"
    vutils.save_image((output + 1) / 2.0, output_path)
    print(f"[INFO] Output saved to: {output_path}")
    
    return output_path


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CycleGAN for Face Photo ↔ Sketch Translation")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "infer"],
                        help="Mode: train, test, or infer")
    parser.add_argument("--input", type=str, default=None,
                        help="Input image path (for infer mode)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file (optional)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--img_size", type=int, default=None,
                        help="Override image size")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data root directory")
    
    args = parser.parse_args()
    
    # Build configuration
    config = Config()
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.img_size:
        config.img_size = args.img_size
    if args.data_root:
        config.data_root = args.data_root
    
    print("=" * 60)
    print("  CycleGAN - Face Photo ↔ Sketch Translation")
    print("=" * 60)
    print(f"  Mode:       {args.mode}")
    print(f"  Device:     {config.device}")
    print(f"  Image Size: {config.img_size}×{config.img_size}")
    print(f"  Data Root:  {config.data_root}")
    print("=" * 60)
    
    if args.mode == "train":
        train(config)
    elif args.mode == "test":
        test(config, args.checkpoint)
    elif args.mode == "infer":
        if args.input is None:
            print("[ERROR] Please provide --input <image_path> for inference mode.")
            sys.exit(1)
        infer(config, args.input, args.checkpoint)


if __name__ == "__main__":
    main()
