import tensorflow as tf
import numpy as np
import pandas as pd
import json
import pickle
import time
from datetime import datetime
import matplotlib.pyplot as plt

print("=" * 80)
print("TASK 6: MODEL TRAINING AND EXPERIMENT TRACKING")
print("=" * 80)

# ==================== HYPERPARAMETERS ====================
print("\nLoading configuration...")

# Model parameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
VOCAB_SIZES = {'english': 11421, 'urdu': 7534}
PAD_IDX = 0

# Training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10
GRADIENT_CLIP = 5.0
EARLY_STOPPING_PATIENCE = 3

print(f"Model Config: Embedding={EMBEDDING_DIM}, Hidden={HIDDEN_DIM}, Layers={NUM_LAYERS}")
print(f"Training Config: LR={LEARNING_RATE}, Epochs={NUM_EPOCHS}, Grad Clip={GRADIENT_CLIP}")

# ==================== LOAD DATA ====================
print("\n" + "=" * 80)
print("LOADING ENCODED DATASETS")
print("=" * 80)

train_encoder = np.load('train_encoder_inputs.npy')
train_decoder_in = np.load('train_decoder_inputs.npy')
train_decoder_target = np.load('train_decoder_targets.npy')

val_encoder = np.load('validation_encoder_inputs.npy')
val_decoder_in = np.load('validation_decoder_inputs.npy')
val_decoder_target = np.load('validation_decoder_targets.npy')

print(f"Train: {train_encoder.shape[0]} samples")
print(f"Validation: {val_encoder.shape[0]} samples")

# ==================== BUILD MODEL ====================
print("\n" + "=" * 80)
print("BUILDING AND COMPILING MODEL")
print("=" * 80)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.0):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.rnn_layers = [
            tf.keras.layers.SimpleRNN(hidden_dim, return_sequences=(i < num_layers - 1),
                                    return_state=True, dropout=dropout if i > 0 else 0.0)
            for i in range(num_layers)
        ]
    
    def call(self, x, training=False):
        x = self.embedding(x)
        states = []
        for i, rnn_layer in enumerate(self.rnn_layers):
            if i < len(self.rnn_layers) - 1:
                x, state = rnn_layer(x, training=training)
            else:
                _, state = rnn_layer(x, training=training)
            states.append(state)
        return state, states

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.0):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.rnn_layers = [
            tf.keras.layers.SimpleRNN(hidden_dim, return_sequences=True, return_state=True,
                                    dropout=dropout if i > 0 else 0.0)
            for i in range(num_layers)
        ]
        self.output_dense = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x, states_input, training=False):
        x = self.embedding(x)
        final_states = []
        for i, rnn_layer in enumerate(self.rnn_layers):
            x, state = rnn_layer(x, initial_state=states_input[i], training=training)
            final_states.append(state)
        logits = self.output_dense(x)
        return logits, final_states

class NMTModel(tf.keras.Model):
    def __init__(self, encoder, decoder, pad_idx=0):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
    
    def call(self, encoder_inputs, decoder_inputs, training=False):
        context_vector, encoder_states = self.encoder(encoder_inputs, training=training)
        decoder_logits, _ = self.decoder(decoder_inputs, encoder_states, training=training)
        return decoder_logits

# Create model
encoder = Encoder(VOCAB_SIZES['english'], EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(VOCAB_SIZES['urdu'], EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model = NMTModel(encoder, decoder, PAD_IDX)

print("✅ Model built successfully")

# ==================== LOSS AND OPTIMIZER ====================
print("\nConfiguring loss and optimizer...")

def loss_fn(y_true, y_pred):
    """Masked sparse categorical crossentropy"""
    mask = tf.cast(tf.not_equal(y_true, PAD_IDX), tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss *= mask
    return tf.reduce_mean(loss)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
print(f"✅ Loss: Masked Sparse Categorical Crossentropy")
print(f"✅ Optimizer: Adam (lr={LEARNING_RATE})")

# ==================== TRAINING LOOP ====================
print("\n" + "=" * 80)
print("TRAINING LOOP WITH GRADIENT CLIPPING")
print("=" * 80)

@tf.function
def train_step(encoder_input, decoder_input, target):
    with tf.GradientTape() as tape:
        logits = model(encoder_input, decoder_input, training=True)
        loss = loss_fn(target, logits)
    
    trainable_vars = model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    
    # Gradient clipping
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, GRADIENT_CLIP)
    
    optimizer.apply_gradients(zip(clipped_gradients, trainable_vars))
    return loss

@tf.function
def val_step(encoder_input, decoder_input, target):
    logits = model(encoder_input, decoder_input, training=False)
    loss = loss_fn(target, logits)
    return loss

# Tracking
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0
start_time = time.time()

# Training loop
for epoch in range(NUM_EPOCHS):
    epoch_train_loss = 0
    num_batches = 0
    
    # Train
    indices = np.random.permutation(len(train_encoder))
    for i in range(0, len(train_encoder), BATCH_SIZE):
        batch_indices = indices[i:min(i+BATCH_SIZE, len(train_encoder))]
        
        enc_batch = tf.constant(train_encoder[batch_indices])
        dec_batch = tf.constant(train_decoder_in[batch_indices])
        tgt_batch = tf.constant(train_decoder_target[batch_indices])
        
        batch_loss = train_step(enc_batch, dec_batch, tgt_batch)
        epoch_train_loss += batch_loss.numpy()
        num_batches += 1
    
    epoch_train_loss /= num_batches
    
    # Validation
    epoch_val_loss = 0
    num_val_batches = 0
    for i in range(0, len(val_encoder), BATCH_SIZE):
        enc_batch = tf.constant(val_encoder[i:min(i+BATCH_SIZE, len(val_encoder))])
        dec_batch = tf.constant(val_decoder_in[i:min(i+BATCH_SIZE, len(val_encoder))])
        tgt_batch = tf.constant(val_decoder_target[i:min(i+BATCH_SIZE, len(val_encoder))])
        
        batch_loss = val_step(enc_batch, dec_batch, tgt_batch)
        epoch_val_loss += batch_loss.numpy()
        num_val_batches += 1
    
    epoch_val_loss /= num_val_batches
    
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    
    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
    
    # Early stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        # Save checkpoint
        model.save_weights(f'best_model_epoch_{epoch+1}.h5')
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"⚠ Early stopping at epoch {epoch+1}")
            break

training_time = time.time() - start_time
print(f"\n✅ Training completed in {training_time/60:.2f} minutes")

# ==================== SAVE TRAINING HISTORY ====================
print("\n" + "=" * 80)
print("SAVING TRAINING HISTORY")
print("=" * 80)

history = {
    'train_loss': [float(l) for l in train_losses],
    'val_loss': [float(l) for l in val_losses],
    'best_val_loss': float(best_val_loss),
    'best_epoch': np.argmin(val_losses) + 1,
    'training_time_seconds': training_time
}

with open('training_history.json', 'w') as f:
    json.dump(history, f, indent=4)

print("✅ Training history saved")

# Plot and save training curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
print("✅ Training curves saved: training_curves.png")
plt.close()

# ==================== TRAINING REPORT ====================
report = f"""
MODEL TRAINING AND EXPERIMENT TRACKING REPORT
==============================================

1. TRAINING CONFIGURATION
   Learning Rate: {LEARNING_RATE}
   Batch Size: {BATCH_SIZE}
   Number of Epochs: {NUM_EPOCHS}
   Gradient Clipping: {GRADIENT_CLIP}
   Early Stopping Patience: {EARLY_STOPPING_PATIENCE}

2. TRAINING PROGRESS
   Total Training Time: {training_time/60:.2f} minutes
   Best Validation Loss: {best_val_loss:.4f}
   Best Epoch: {np.argmin(val_losses) + 1}
   Final Training Loss: {train_losses[-1]:.4f}
   Final Validation Loss: {val_losses[-1]:.4f}

3. LOSS PROGRESSION
   Training Set:
"""

for i, loss in enumerate(train_losses, 1):
    report += f"   - Epoch {i:2d}: {loss:.6f}\n"

report += "\n   Validation Set:\n"
for i, loss in enumerate(val_losses, 1):
    report += f"   - Epoch {i:2d}: {loss:.6f}\n"

report += f"""
4. CONVERGENCE ANALYSIS
   Loss Reduction: {(train_losses[0] - train_losses[-1])/train_losses[0]*100:.2f}%
   Validation Improvement: {(val_losses[0] - best_val_loss)/val_losses[0]*100:.2f}%

5. GRADIENT CLIPPING
   Max Gradient Norm: {GRADIENT_CLIP}
   Purpose: Prevent exploding gradients in RNN training

6. CHECKPOINTS
   Best Model: best_model_epoch_{np.argmin(val_losses) + 1}.h5
   
7. FILES GENERATED
   - training_history.json: Complete loss values
   - best_model_epoch_X.h5: Best model weights
   - training_curves.png: Loss curve visualization

8. NEXT STEPS
   Task 7: Hyperparameter Tuning
   Task 8: Inference and Evaluation
"""

with open('training_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("✅ Report saved: training_report.txt")

print("\n" + "=" * 80)
print("✅ TASK 6 COMPLETED")
print("=" * 80)
