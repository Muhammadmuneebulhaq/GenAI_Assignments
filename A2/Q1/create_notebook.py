"""Script to generate cyclegan.ipynb notebook."""
import json

cells = []

def md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().split("\n")
    })

def code(source):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": source.strip().split("\n"),
        "outputs": [],
        "execution_count": None
    })

# ===========================================================================
# Cell 1: Title
# ===========================================================================
md("""# CycleGAN Implementation for Person Face Sketches
---
**Based on:** "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" by Zhu et al. (2017)

**Objective:** Perform bidirectional image-to-image translation:
- Real face photos → Sketches (Generator G_AB)
- Sketches → Real face photos (Generator G_BA)

**Dataset:** [Person Face Sketches (Kaggle)](https://www.kaggle.com/datasets/almightyj/person-face-sketches)""")

# ===========================================================================
# Cell 2: Imports
# ===========================================================================
md("## 1. Imports and Setup")

code("""import os
import sys
import random
import itertools
from collections import deque

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils

# For inline plots in notebook
%matplotlib inline

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")""")

# ===========================================================================
# Cell 3: Config
# ===========================================================================
md("## 2. Configuration / Hyperparameters")

code('''class Config:
    """Central configuration for all hyperparameters."""
    # Data
    data_root = "data"
    img_size = 256                  # Resize images to 256x256

    # Training
    batch_size = 1                  # CycleGAN typically uses batch_size=1
    num_epochs = 200                # Total training epochs
    lr = 0.0002                     # Initial learning rate (Adam)
    beta1 = 0.5                     # Adam beta1
    beta2 = 0.999                   # Adam beta2
    lambda_cycle = 10.0             # Cycle-consistency loss weight
    lambda_identity = 5.0           # Identity loss weight (0.5 * lambda_cycle)
    decay_epoch = 100               # Epoch from which to start lr decay

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
    num_workers = 2                 # Reduced for Colab compatibility

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
print(f"Using device: {config.device}")
os.makedirs(config.checkpoint_dir, exist_ok=True)
os.makedirs(config.sample_dir, exist_ok=True)''')

# ===========================================================================
# Cell 4: Dataset
# ===========================================================================
md("""## 3. Dataset

The dataset has paired face photos and sketches in `data/{train,val,test}/{photos,sketches}/`.
Even though CycleGAN works with unpaired data, we shuffle each domain independently to simulate unpaired training.""")

code('''class FaceSketchDataset(Dataset):
    """
    Dataset for loading face photos and sketches.
    Loads them as separate domains and shuffles independently
    to simulate unpaired training.
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.photo_dir = os.path.join(root_dir, split, "photos")
        self.sketch_dir = os.path.join(root_dir, split, "sketches")

        self.photo_files = sorted(os.listdir(self.photo_dir))
        self.sketch_files = sorted(os.listdir(self.sketch_dir))

        self.transform = transform
        self.len_photos = len(self.photo_files)
        self.len_sketches = len(self.sketch_files)

    def __len__(self):
        return max(self.len_photos, self.len_sketches)

    def __getitem__(self, idx):
        photo_path = os.path.join(self.photo_dir, self.photo_files[idx % self.len_photos])
        sketch_path = os.path.join(self.sketch_dir, self.sketch_files[idx % self.len_sketches])

        photo = Image.open(photo_path).convert("RGB")
        sketch = Image.open(sketch_path).convert("RGB")

        if self.transform:
            photo = self.transform(photo)
            sketch = self.transform(sketch)

        return {"photo": photo, "sketch": sketch}


def get_transforms(img_size, is_train=True):
    """Build data augmentation / preprocessing transforms."""
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
        ])''')

# ===========================================================================
# Cell 5: Visualize dataset
# ===========================================================================
md("### 3.1 Visualize Dataset Samples")

code('''def show_dataset_samples(dataset, n=4):
    """Display n random samples from the dataset."""
    fig, axes = plt.subplots(n, 2, figsize=(8, 4*n))
    for i in range(n):
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        photo = (sample["photo"] * 0.5 + 0.5).permute(1, 2, 0).numpy()
        sketch = (sample["sketch"] * 0.5 + 0.5).permute(1, 2, 0).numpy()
        axes[i, 0].imshow(photo)
        axes[i, 0].set_title("Photo")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(sketch)
        axes[i, 1].set_title("Sketch")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()

# Load a small preview
preview_transform = get_transforms(config.img_size, is_train=False)
preview_dataset = FaceSketchDataset(config.data_root, split="train", transform=preview_transform)
print(f"Train photos: {preview_dataset.len_photos}, Train sketches: {preview_dataset.len_sketches}")
show_dataset_samples(preview_dataset, n=4)''')

# ===========================================================================
# Cell 6: Generator
# ===========================================================================
md("""## 4. Generator (ResNet-based)

The generator follows the architecture from the CycleGAN paper:
1. **Initial convolution** (7x7, reflect-padded)
2. **Downsampling** (2 stride-2 convolutions)
3. **Residual blocks** (9 blocks for 256x256 images)
4. **Upsampling** (2 transposed convolutions)
5. **Output convolution** (7x7, reflect-padded, tanh activation)

Uses **Instance Normalization** instead of Batch Normalization.""")

code('''class ResidualBlock(nn.Module):
    """Residual block with reflect padding, InstanceNorm, and ReLU."""

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
    """ResNet-based Generator for CycleGAN."""

    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_residual_blocks=9):
        super().__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]

        # Residual blocks
        mult = 2 ** n_downsampling
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(ngf * mult)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2,
                                   kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Verify generator architecture
G_test = Generator(n_residual_blocks=config.n_residual_blocks, ngf=config.ngf)
test_input = torch.randn(1, 3, config.img_size, config.img_size)
test_output = G_test(test_input)
print(f"Generator input:  {test_input.shape}")
print(f"Generator output: {test_output.shape}")
total_params = sum(p.numel() for p in G_test.parameters())
print(f"Generator parameters: {total_params:,}")
del G_test, test_input, test_output''')

# ===========================================================================
# Cell 7: Discriminator
# ===========================================================================
md("""## 5. Discriminator (PatchGAN)

The **70x70 PatchGAN** discriminator classifies overlapping patches of the image as real or fake, producing a 2D map of predictions rather than a single scalar.""")

code('''class Discriminator(nn.Module):
    """PatchGAN Discriminator (70x70 receptive field)."""

    def __init__(self, in_channels=3, ndf=64):
        super().__init__()

        def discriminator_block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, ndf, normalize=False),
            *discriminator_block(ndf, ndf * 2),
            *discriminator_block(ndf * 2, ndf * 4),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, padding=1),
        )

    def forward(self, x):
        return self.model(x)

# Verify discriminator architecture
D_test = Discriminator(ndf=config.ndf)
test_input = torch.randn(1, 3, config.img_size, config.img_size)
test_output = D_test(test_input)
print(f"Discriminator input:  {test_input.shape}")
print(f"Discriminator output: {test_output.shape} (patch map)")
total_params = sum(p.numel() for p in D_test.parameters())
print(f"Discriminator parameters: {total_params:,}")
del D_test, test_input, test_output''')

# ===========================================================================
# Cell 8: Utilities
# ===========================================================================
md("""## 6. Utility Classes

### Replay Buffer
Stores previously generated images. With 50% probability, returns a stored image instead of the current one to stabilize discriminator training.

### Weight Initialization
Normal distribution (mean=0, std=0.02) as specified in the CycleGAN paper.

### Learning Rate Scheduler
Linear decay: keeps initial LR for first `decay_epoch` epochs, then linearly decays to zero.""")

code('''class ReplayBuffer:
    """Image replay buffer for stabilizing GAN training."""

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
                    idx = random.randint(0, self.max_size - 1)
                    tmp = self.data[idx].clone()
                    self.data[idx] = element
                    result.append(tmp)
                else:
                    result.append(element)
        return torch.cat(result, dim=0)


def init_weights(m):
    """Initialize weights: Normal(0, 0.02) for Conv, Normal(1, 0.02) for Norm layers."""
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


class LambdaLR:
    """Linear learning rate decay starting at decay_start_epoch."""

    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert n_epochs > decay_start_epoch
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )

print("Utilities defined successfully.")''')

# ===========================================================================
# Cell 9: Build models
# ===========================================================================
md("## 7. Initialize Models, Optimizers, and Schedulers")

code('''# Create models
# G_AB: Photo -> Sketch,  G_BA: Sketch -> Photo
G_AB = Generator(n_residual_blocks=config.n_residual_blocks, ngf=config.ngf).to(config.device)
G_BA = Generator(n_residual_blocks=config.n_residual_blocks, ngf=config.ngf).to(config.device)
D_A  = Discriminator(ndf=config.ndf).to(config.device)   # Discriminates real photos
D_B  = Discriminator(ndf=config.ndf).to(config.device)   # Discriminates real sketches

# Initialize weights
G_AB.apply(init_weights)
G_BA.apply(init_weights)
D_A.apply(init_weights)
D_B.apply(init_weights)

# Losses
criterion_GAN      = nn.MSELoss()    # LSGAN loss
criterion_cycle    = nn.L1Loss()     # Cycle-consistency loss
criterion_identity = nn.L1Loss()     # Identity loss

# Optimizers
optimizer_G   = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()),
    lr=config.lr, betas=(config.beta1, config.beta2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

# LR Schedulers
lr_scheduler_G   = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(config.num_epochs, 0, config.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(config.num_epochs, 0, config.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(config.num_epochs, 0, config.decay_epoch).step
)

# Replay Buffers
fake_A_buffer = ReplayBuffer(config.buffer_size)
fake_B_buffer = ReplayBuffer(config.buffer_size)

total = (sum(p.numel() for p in G_AB.parameters()) +
         sum(p.numel() for p in G_BA.parameters()) +
         sum(p.numel() for p in D_A.parameters()) +
         sum(p.numel() for p in D_B.parameters()))
print("All models, optimizers, and schedulers initialized.")
print(f"Total trainable parameters: {total:,}")''')

# ===========================================================================
# Cell 10: Data loaders
# ===========================================================================
md("## 8. Create Data Loaders")

code('''train_transform = get_transforms(config.img_size, is_train=True)
val_transform   = get_transforms(config.img_size, is_train=False)

train_dataset = FaceSketchDataset(config.data_root, split="train", transform=train_transform)
val_dataset   = FaceSketchDataset(config.data_root, split="val",   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                          num_workers=config.num_workers, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False,
                          num_workers=config.num_workers, pin_memory=True)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Training batches per epoch: {len(train_loader)}")''')

# ===========================================================================
# Cell 11: Resume checkpoint
# ===========================================================================
md("## 9. Resume from Checkpoint (if available)")

code('''start_epoch = 0
latest_ckpt = os.path.join(config.checkpoint_dir, "latest.pth")

if os.path.exists(latest_ckpt):
    print(f"Resuming from checkpoint: {latest_ckpt}")
    ckpt = torch.load(latest_ckpt, map_location=config.device)
    G_AB.load_state_dict(ckpt["G_AB"])
    G_BA.load_state_dict(ckpt["G_BA"])
    D_A.load_state_dict(ckpt["D_A"])
    D_B.load_state_dict(ckpt["D_B"])
    optimizer_G.load_state_dict(ckpt["optimizer_G"])
    optimizer_D_A.load_state_dict(ckpt["optimizer_D_A"])
    optimizer_D_B.load_state_dict(ckpt["optimizer_D_B"])
    start_epoch = ckpt["epoch"] + 1
    print(f"Resumed at epoch {start_epoch}")
else:
    print("No checkpoint found. Training from scratch.")''')

# ===========================================================================
# Cell 12: Helper functions
# ===========================================================================
md("## 10. Helper Functions for Saving Samples")

code('''def save_and_show_samples(G_AB, G_BA, val_loader, epoch, config, show=True):
    """Generate and optionally display sample translations."""
    G_AB.eval()
    G_BA.eval()

    with torch.no_grad():
        batch = next(iter(val_loader))
        real_A = batch["photo"].to(config.device)
        real_B = batch["sketch"].to(config.device)

        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        recovered_A = G_BA(fake_B)
        recovered_B = G_AB(fake_A)

        def denorm(x):
            return (x.cpu() + 1) / 2.0

        if show:
            n_show = min(4, real_A.size(0))
            fig, axes = plt.subplots(n_show, 6, figsize=(18, 3 * n_show))
            titles = ["Photo", "Photo->Sketch", "Recovered Photo",
                      "Sketch", "Sketch->Photo", "Recovered Sketch"]

            for i in range(n_show):
                imgs = [denorm(real_A[i]), denorm(fake_B[i]), denorm(recovered_A[i]),
                        denorm(real_B[i]), denorm(fake_A[i]), denorm(recovered_B[i])]
                for j, img in enumerate(imgs):
                    axes[i, j].imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
                    axes[i, j].axis("off")
                    if i == 0:
                        axes[i, j].set_title(titles[j], fontsize=10)
            plt.suptitle(f"Epoch {epoch+1}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.show()

        # Save grid to file
        img_grid_A = torch.cat([denorm(real_A), denorm(fake_B), denorm(recovered_A)], dim=3)
        img_grid_B = torch.cat([denorm(real_B), denorm(fake_A), denorm(recovered_B)], dim=3)
        img_grid = torch.cat([img_grid_A, img_grid_B], dim=0)
        save_path = os.path.join(config.sample_dir, f"epoch_{epoch+1}.png")
        vutils.save_image(img_grid, save_path, nrow=1, normalize=False)

    G_AB.train()
    G_BA.train()


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
    path = os.path.join(config.checkpoint_dir, f"epoch_{epoch+1}.pth")
    torch.save(ckpt, path)
    torch.save(ckpt, os.path.join(config.checkpoint_dir, "latest.pth"))
    print(f"Checkpoint saved: {path}")

print("Helper functions defined.")''')

# ===========================================================================
# Cell 13: Training loop
# ===========================================================================
md("""## 11. Training Loop

The training loop alternates between:
1. **Generator training**: adversarial loss + cycle-consistency loss + identity loss
2. **Discriminator A training**: real photos vs. fake photos (from sketch->photo generator)
3. **Discriminator B training**: real sketches vs. fake sketches (from photo->sketch generator)

Uses LSGAN (MSE) loss for adversarial training and L1 loss for cycle/identity.""")

code('''# Training history
history = {"G_loss": [], "D_loss": [], "cycle_loss": [], "identity_loss": []}

for epoch in range(start_epoch, config.num_epochs):
    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()

    epoch_G, epoch_D, epoch_cyc, epoch_idt = 0.0, 0.0, 0.0, 0.0

    for i, batch in enumerate(train_loader):
        real_A = batch["photo"].to(config.device)
        real_B = batch["sketch"].to(config.device)

        # Target tensors for LSGAN
        valid = torch.ones_like(D_A(real_A), requires_grad=False).to(config.device)
        fake  = torch.zeros_like(D_A(real_A), requires_grad=False).to(config.device)

        # ---- Train Generators ----
        optimizer_G.zero_grad()

        # Identity loss
        identity_B = G_AB(real_B)
        loss_idt_B = criterion_identity(identity_B, real_B) * config.lambda_identity
        identity_A = G_BA(real_A)
        loss_idt_A = criterion_identity(identity_A, real_A) * config.lambda_identity

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        # Cycle loss
        recovered_A = G_BA(fake_B)
        loss_cyc_A = criterion_cycle(recovered_A, real_A) * config.lambda_cycle
        recovered_B = G_AB(fake_A)
        loss_cyc_B = criterion_cycle(recovered_B, real_B) * config.lambda_cycle

        # Total generator loss
        loss_G = loss_GAN_AB + loss_GAN_BA + loss_cyc_A + loss_cyc_B + loss_idt_A + loss_idt_B
        loss_G.backward()
        optimizer_G.step()

        # ---- Train Discriminator A ----
        optimizer_D_A.zero_grad()
        loss_D_real_A = criterion_GAN(D_A(real_A), valid)
        fake_A_buf = fake_A_buffer.push_and_pop(fake_A.detach())
        loss_D_fake_A = criterion_GAN(D_A(fake_A_buf), fake)
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()

        # ---- Train Discriminator B ----
        optimizer_D_B.zero_grad()
        loss_D_real_B = criterion_GAN(D_B(real_B), valid)
        fake_B_buf = fake_B_buffer.push_and_pop(fake_B.detach())
        loss_D_fake_B = criterion_GAN(D_B(fake_B_buf), fake)
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()

        # Accumulate
        epoch_G   += loss_G.item()
        epoch_D   += (loss_D_A.item() + loss_D_B.item())
        epoch_cyc += (loss_cyc_A + loss_cyc_B).item()
        epoch_idt += (loss_idt_A + loss_idt_B).item()

        if (i + 1) % 200 == 0:
            print(f"  [{epoch+1}/{config.num_epochs}] Batch {i+1}/{len(train_loader)} | "
                  f"G: {loss_G.item():.4f} D: {(loss_D_A+loss_D_B).item():.4f}")

    # Epoch summary
    n = len(train_loader)
    history["G_loss"].append(epoch_G / n)
    history["D_loss"].append(epoch_D / n)
    history["cycle_loss"].append(epoch_cyc / n)
    history["identity_loss"].append(epoch_idt / n)

    lr_val = optimizer_G.param_groups[0]["lr"]
    print(f"Epoch [{epoch+1}/{config.num_epochs}] G: {epoch_G/n:.4f} D: {epoch_D/n:.4f} "
          f"Cyc: {epoch_cyc/n:.4f} Idt: {epoch_idt/n:.4f} LR: {lr_val:.6f}")

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save samples (show every 10 epochs)
    save_and_show_samples(G_AB, G_BA, val_loader, epoch, config, show=((epoch+1) % 10 == 0))

    # Save checkpoint
    if (epoch + 1) % config.save_every == 0 or epoch == config.num_epochs - 1:
        save_checkpoint(G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, epoch, config)

print("Training complete!")''')

# ===========================================================================
# Cell 14: Plot losses
# ===========================================================================
md("## 12. Plot Training Curves")

code('''fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(history["G_loss"], label="Generator", color="#4facfe")
axes[0,0].set_title("Generator Loss"); axes[0,0].set_xlabel("Epoch"); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

axes[0,1].plot(history["D_loss"], label="Discriminator", color="#f5576c")
axes[0,1].set_title("Discriminator Loss"); axes[0,1].set_xlabel("Epoch"); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

axes[1,0].plot(history["cycle_loss"], label="Cycle Consistency", color="#43e97b")
axes[1,0].set_title("Cycle-Consistency Loss"); axes[1,0].set_xlabel("Epoch"); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

axes[1,1].plot(history["identity_loss"], label="Identity", color="#f093fb")
axes[1,1].set_title("Identity Loss"); axes[1,1].set_xlabel("Epoch"); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

plt.suptitle("CycleGAN Training Curves", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()''')

# ===========================================================================
# Cell 15: Test evaluation
# ===========================================================================
md("""## 13. Test Evaluation

Run the trained model on the test set and visualize results.""")

code('''def evaluate_test_set(G_AB, G_BA, config, num_display=8):
    """Evaluate on test set and display sample results."""
    test_transform = get_transforms(config.img_size, is_train=False)
    test_dataset = FaceSketchDataset(config.data_root, split="test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    os.makedirs("test_results/photo_to_sketch", exist_ok=True)
    os.makedirs("test_results/sketch_to_photo", exist_ok=True)

    G_AB.eval()
    G_BA.eval()

    results_AB, results_BA = [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            real_A = batch["photo"].to(config.device)
            real_B = batch["sketch"].to(config.device)

            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)

            vutils.save_image((fake_B + 1) / 2.0, f"test_results/photo_to_sketch/{i:04d}.png")
            vutils.save_image((fake_A + 1) / 2.0, f"test_results/sketch_to_photo/{i:04d}.png")

            if i < num_display:
                results_AB.append((real_A.cpu(), fake_B.cpu()))
                results_BA.append((real_B.cpu(), fake_A.cpu()))

    # Display Photo -> Sketch results
    fig, axes = plt.subplots(num_display, 2, figsize=(8, 4 * num_display))
    fig.suptitle("Photo -> Sketch (Test Set)", fontsize=16, fontweight="bold")
    for i, (real, fake_img) in enumerate(results_AB):
        axes[i, 0].imshow(((real.squeeze() + 1) / 2).permute(1,2,0).clamp(0,1).numpy())
        axes[i, 0].set_title("Real Photo"); axes[i, 0].axis("off")
        axes[i, 1].imshow(((fake_img.squeeze() + 1) / 2).permute(1,2,0).clamp(0,1).numpy())
        axes[i, 1].set_title("Generated Sketch"); axes[i, 1].axis("off")
    plt.tight_layout(); plt.show()

    # Display Sketch -> Photo results
    fig, axes = plt.subplots(num_display, 2, figsize=(8, 4 * num_display))
    fig.suptitle("Sketch -> Photo (Test Set)", fontsize=16, fontweight="bold")
    for i, (real, fake_img) in enumerate(results_BA):
        axes[i, 0].imshow(((real.squeeze() + 1) / 2).permute(1,2,0).clamp(0,1).numpy())
        axes[i, 0].set_title("Real Sketch"); axes[i, 0].axis("off")
        axes[i, 1].imshow(((fake_img.squeeze() + 1) / 2).permute(1,2,0).clamp(0,1).numpy())
        axes[i, 1].set_title("Generated Photo"); axes[i, 1].axis("off")
    plt.tight_layout(); plt.show()

    print(f"Test results saved to test_results/ ({len(test_dataset)} images)")

evaluate_test_set(G_AB, G_BA, config)''')

# ===========================================================================
# Cell 16: Single inference
# ===========================================================================
md("""## 14. Single Image Inference

Upload or provide a path to a single image. The model auto-detects whether it is a photo or sketch and translates accordingly.""")

code('''def infer_single_image(image_path, G_AB, G_BA, config):
    """Run inference on a single image with auto-detection."""
    transform = get_transforms(config.img_size, is_train=False)
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(config.device)

    # Detect type based on color saturation
    img_np = np.array(img).astype(np.float32)
    saturation = np.std(img_np[:,:,0] - img_np[:,:,1])
    is_sketch = saturation < 15

    G_AB.eval()
    G_BA.eval()

    with torch.no_grad():
        if is_sketch:
            output = G_BA(img_tensor)
            direction = "Sketch -> Photo"
        else:
            output = G_AB(img_tensor)
            direction = "Photo -> Sketch"

    # Display
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("Input"); axes[0].axis("off")

    out_img = ((output.squeeze().cpu() + 1) / 2).permute(1,2,0).clamp(0,1).numpy()
    axes[1].imshow(out_img)
    axes[1].set_title(f"Output ({direction})"); axes[1].axis("off")

    plt.suptitle(f"Auto-detected: {direction}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Save
    output_path = os.path.splitext(image_path)[0] + "_translated.png"
    vutils.save_image((output + 1) / 2.0, output_path)
    print(f"Saved to: {output_path}")

# Example usage (uncomment and provide a path):
# infer_single_image("data/test/photos/0.jpg", G_AB, G_BA, config)
# infer_single_image("data/test/sketches/0.jpg", G_AB, G_BA, config)''')

# ===========================================================================
# Build notebook JSON
# ===========================================================================

# Fix source lines: each line needs a newline except the last
for cell in cells:
    lines = cell["source"]
    new_lines = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            new_lines.append(line + "\n")
        else:
            new_lines.append(line)
    cell["source"] = new_lines

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

with open("cyclegan.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook created: cyclegan.ipynb ({len(cells)} cells)")
