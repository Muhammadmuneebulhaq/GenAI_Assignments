import pandas as pd
import numpy as np
import json
import pickle

print("=" * 80)
print("TASK 4: SEQUENCE ENCODING, PADDING, AND BATCHING")
print("=" * 80)

# ==================== LOAD VOCABULARIES ====================
print("\nLoading vocabularies...")

with open('english_vocab.json', 'r', encoding='utf-8') as f:
    english_vocab = json.load(f)

with open('urdu_vocab.json', 'r', encoding='utf-8') as f:
    urdu_vocab = json.load(f)

with open('english_reverse_vocab.json', 'r', encoding='utf-8') as f:
    english_reverse_vocab = json.load(f)

with open('urdu_reverse_vocab.json', 'r', encoding='utf-8') as f:
    urdu_reverse_vocab = json.load(f)

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

print(f"English Vocab Size: {len(english_vocab)}")
print(f"Urdu Vocab Size: {len(urdu_vocab)}")

# ==================== TOKENIZATION FUNCTIONS ====================
def tokenize_english(text):
    """Tokenize English text"""
    return text.lower().split()

def tokenize_urdu(text):
    """Tokenize Urdu text"""
    return text.split()

# ==================== ENCODING FUNCTIONS ====================
def encode_sequence(tokens, vocab, add_special=True):
    """
    Convert tokens to integer sequence using vocabulary
    
    Args:
        tokens: List of tokens
        vocab: Token to index mapping
        add_special: If True, add BOS at start and EOS at end
    
    Returns:
        List of integer indices
    """
    sequence = []
    
    # Add BOS token
    if add_special:
        sequence.append(BOS_IDX)
    
    # Encode each token
    for token in tokens:
        if token in vocab:
            sequence.append(vocab[token])
        else:
            sequence.append(UNK_IDX)  # Unknown token
    
    # Add EOS token
    if add_special:
        sequence.append(EOS_IDX)
    
    return sequence

# ==================== PADDING FUNCTIONS ====================
def pad_sequence(sequence, max_len, pad_idx=PAD_IDX):
    """
    Pad or truncate sequence to max_len
    
    Args:
        sequence: List of integers
        max_len: Target length
        pad_idx: Padding index
    
    Returns:
        Padded sequence and mask
    """
    seq_len = len(sequence)
    
    if seq_len >= max_len:
        # Truncate
        padded_seq = sequence[:max_len]
        mask = [1] * max_len
    else:
        # Pad
        padded_seq = sequence + [pad_idx] * (max_len - seq_len)
        mask = [1] * seq_len + [0] * (max_len - seq_len)
    
    return padded_seq, mask

# ==================== LOAD DATASETS ====================
print("\n" + "=" * 80)
print("LOADING DATASETS")
print("=" * 80)

train_df = pd.read_csv('train_set.csv', encoding='utf-8')
val_df = pd.read_csv('validation_set.csv', encoding='utf-8')
test_df = pd.read_csv('test_set.csv', encoding='utf-8')

print(f"\nTrain: {len(train_df)} samples")
print(f"Validation: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")

# ==================== ENCODE AND PAD DATASETS ====================
print("\n" + "=" * 80)
print("ENCODING AND PADDING SEQUENCES")
print("=" * 80)

def process_dataset(df, name):
    """Process a dataset: tokenize, encode, and pad"""
    print(f"\nProcessing {name}...")
    
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []
    encoder_masks = []
    decoder_masks = []
    
    # Calculate max lengths from dataset
    max_eng_len = 0
    max_urdu_len = 0
    
    for idx in range(len(df)):
        eng_tokens = tokenize_english(df.iloc[idx]['eng'])
        urdu_tokens = tokenize_urdu(df.iloc[idx]['urdu'])
        
        max_eng_len = max(max_eng_len, len(eng_tokens) + 2)  # +2 for BOS, EOS
        max_urdu_len = max(max_urdu_len, len(urdu_tokens) + 2)
    
    # Add padding buffer (round up to nearest 10)
    max_eng_len = ((max_eng_len + 9) // 10) * 10
    max_urdu_len = ((max_urdu_len + 9) // 10) * 10
    
    print(f"  Max English sequence length: {max_eng_len}")
    print(f"  Max Urdu sequence length: {max_urdu_len}")
    
    # Process each sample
    for idx in range(len(df)):
        eng_tokens = tokenize_english(df.iloc[idx]['eng'])
        urdu_tokens = tokenize_urdu(df.iloc[idx]['urdu'])
        
        # Encode sequences
        eng_encoded = encode_sequence(eng_tokens, english_vocab, add_special=True)
        urdu_encoded = encode_sequence(urdu_tokens, urdu_vocab, add_special=True)
        
        # Pad sequences
        eng_padded, eng_mask = pad_sequence(eng_encoded, max_eng_len, PAD_IDX)
        urdu_padded, urdu_mask = pad_sequence(urdu_encoded, max_urdu_len, PAD_IDX)
        
        # Decoder input: decoder_targets shifted by 1 (for teacher forcing)
        # decoder_input: same as urdu_padded (with BOS)
        # decoder_target: shifted version (without leading BOS, with trailing EOS)
        decoder_target = urdu_padded[1:]  # Remove BOS, keep EOS
        if len(decoder_target) < max_urdu_len:
            decoder_target.append(EOS_IDX)
        
        encoder_inputs.append(eng_padded)
        decoder_inputs.append(urdu_padded)
        decoder_targets.append(decoder_target)
        encoder_masks.append(eng_mask)
        decoder_masks.append(urdu_mask)
    
    return {
        'encoder_inputs': np.array(encoder_inputs, dtype=np.int32),
        'decoder_inputs': np.array(decoder_inputs, dtype=np.int32),
        'decoder_targets': np.array(decoder_targets, dtype=np.int32),
        'encoder_masks': np.array(encoder_masks, dtype=np.float32),
        'decoder_masks': np.array(decoder_masks, dtype=np.float32),
        'max_eng_len': max_eng_len,
        'max_urdu_len': max_urdu_len
    }

# Process all datasets
train_encoded = process_dataset(train_df, "TRAIN SET")
val_encoded = process_dataset(val_df, "VALIDATION SET")
test_encoded = process_dataset(test_df, "TEST SET")

# ==================== SAVE ENCODED DATASETS ====================
print("\n" + "=" * 80)
print("SAVING ENCODED DATASETS")
print("=" * 80)

def save_encoded_data(data, prefix):
    """Save encoded data to files"""
    np.save(f'{prefix}_encoder_inputs.npy', data['encoder_inputs'])
    np.save(f'{prefix}_decoder_inputs.npy', data['decoder_inputs'])
    np.save(f'{prefix}_decoder_targets.npy', data['decoder_targets'])
    np.save(f'{prefix}_encoder_masks.npy', data['encoder_masks'])
    np.save(f'{prefix}_decoder_masks.npy', data['decoder_masks'])
    
    config = {
        'max_eng_len': data['max_eng_len'],
        'max_urdu_len': data['max_urdu_len'],
        'num_samples': data['encoder_inputs'].shape[0]
    }
    
    with open(f'{prefix}_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✅ Saved {prefix} data")

save_encoded_data(train_encoded, 'train')
save_encoded_data(val_encoded, 'validation')
save_encoded_data(test_encoded, 'test')

# ==================== BATCHING PIPELINE ====================
print("\n" + "=" * 80)
print("IMPLEMENTING BATCHING PIPELINE")
print("=" * 80)

class NMTDataLoader:
    """Simple data loader for batching"""
    
    def __init__(self, encoder_inputs, decoder_inputs, decoder_targets,
                 encoder_masks, decoder_masks, batch_size=32, shuffle=False):
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets
        self.encoder_masks = encoder_masks
        self.decoder_masks = decoder_masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = encoder_inputs.shape[0]
        self.indices = np.arange(self.num_samples)
    
    def __len__(self):
        """Number of batches"""
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches"""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            yield {
                'encoder_inputs': self.encoder_inputs[batch_indices],
                'decoder_inputs': self.decoder_inputs[batch_indices],
                'decoder_targets': self.decoder_targets[batch_indices],
                'encoder_masks': self.encoder_masks[batch_indices],
                'decoder_masks': self.decoder_masks[batch_indices],
                'batch_size': len(batch_indices)
            }

# Create data loaders
train_loader = NMTDataLoader(
    train_encoded['encoder_inputs'],
    train_encoded['decoder_inputs'],
    train_encoded['decoder_targets'],
    train_encoded['encoder_masks'],
    train_encoded['decoder_masks'],
    batch_size=32,
    shuffle=True
)

val_loader = NMTDataLoader(
    val_encoded['encoder_inputs'],
    val_encoded['decoder_inputs'],
    val_encoded['decoder_targets'],
    val_encoded['encoder_masks'],
    val_encoded['decoder_masks'],
    batch_size=32,
    shuffle=False
)

test_loader = NMTDataLoader(
    test_encoded['encoder_inputs'],
    test_encoded['decoder_inputs'],
    test_encoded['decoder_targets'],
    test_encoded['encoder_masks'],
    test_encoded['decoder_masks'],
    batch_size=32,
    shuffle=False
)

print(f"\n✅ Train loader: {len(train_loader)} batches")
print(f"✅ Validation loader: {len(val_loader)} batches")
print(f"✅ Test loader: {len(test_loader)} batches")

# Save loaders
with open('train_loader.pkl', 'wb') as f:
    pickle.dump(train_loader, f)

with open('validation_loader.pkl', 'wb') as f:
    pickle.dump(val_loader, f)

with open('test_loader.pkl', 'wb') as f:
    pickle.dump(test_loader, f)

# ==================== BATCH EXAMPLES ====================
print("\n" + "=" * 80)
print("BATCH EXAMPLES")
print("=" * 80)

# Get first batch
batch = next(iter(train_loader))
print(f"\nBatch Size: {batch['batch_size']}")
print(f"Encoder Input Shape: {batch['encoder_inputs'].shape}")
print(f"Decoder Input Shape: {batch['decoder_inputs'].shape}")
print(f"Decoder Target Shape: {batch['decoder_targets'].shape}")
print(f"Encoder Mask Shape: {batch['encoder_masks'].shape}")
print(f"Decoder Mask Shape: {batch['decoder_masks'].shape}")

# Show first sample in batch
print(f"\nFirst Sample in Batch:")
print(f"Encoder Input (indices): {batch['encoder_inputs'][0]}")
print(f"Decoder Input (indices): {batch['decoder_inputs'][0]}")
print(f"Encoder Mask: {batch['encoder_masks'][0]}")

# Decode first sample
print(f"\nDecoded First Sample (Encoder Input):")
eng_sequence = batch['encoder_inputs'][0]
decoded = [english_reverse_vocab[str(int(idx))] if str(int(idx)) in english_reverse_vocab else f"<{idx}>" for idx in eng_sequence]
print(" ".join(decoded))

print(f"\nDecoded First Sample (Decoder Input):")
urdu_sequence = batch['decoder_inputs'][0]
decoded = [urdu_reverse_vocab[str(int(idx))] if str(int(idx)) in urdu_reverse_vocab else f"<{idx}>" for idx in urdu_sequence]
print(" ".join(decoded))

# ==================== SAVE REPORT ====================
report = f"""
SEQUENCE ENCODING, PADDING, AND BATCHING REPORT
===============================================

1. ENCODING STRATEGY
   - Special Tokens Added:
     * BOS (Begin of Sequence, Index {BOS_IDX}): Added at start
     * EOS (End of Sequence, Index {EOS_IDX}): Added at end
     * UNK (Unknown, Index {UNK_IDX}): For out-of-vocabulary words
     * PAD (Padding, Index {PAD_IDX}): For sequence alignment

2. MAXIMUM SEQUENCE LENGTHS
   Train Set:
   - English: {train_encoded['max_eng_len']} tokens
   - Urdu: {train_encoded['max_urdu_len']} tokens
   
   Validation Set:
   - English: {val_encoded['max_eng_len']} tokens
   - Urdu: {val_encoded['max_urdu_len']} tokens
   
   Test Set:
   - English: {test_encoded['max_eng_len']} tokens
   - Urdu: {test_encoded['max_urdu_len']} tokens

3. PADDING STRATEGY
   - Method: Right padding (pad after sequence)
   - Mask Generation: Binary masks (1 for real tokens, 0 for padding)
   - Purpose: Attention masks enable model to ignore padding

4. DATASET SHAPES (After Encoding & Padding)
   Train Set:
   - Encoder inputs: {train_encoded['encoder_inputs'].shape}
   - Decoder inputs: {train_encoded['decoder_inputs'].shape}
   - Decoder targets: {train_encoded['decoder_targets'].shape}
   - Encoder masks: {train_encoded['encoder_masks'].shape}
   - Decoder masks: {train_encoded['decoder_masks'].shape}
   
   Validation Set:
   - Encoder inputs: {val_encoded['encoder_inputs'].shape}
   - Decoder inputs: {val_encoded['decoder_inputs'].shape}
   
   Test Set:
   - Encoder inputs: {test_encoded['encoder_inputs'].shape}
   - Decoder inputs: {test_encoded['decoder_inputs'].shape}

5. BATCHING PIPELINE
   - Batch Size: 32 samples
   - Shuffle Training: Yes (for randomization)
   - Shuffle Validation/Test: No (for reproducibility)
   
   Number of Batches:
   - Train: {len(train_loader)} batches
   - Validation: {len(val_loader)} batches
   - Test: {len(test_loader)} batches

6. OUTPUT FILES
   Encoded Data (NumPy):
   - train_encoder_inputs.npy
   - train_decoder_inputs.npy
   - train_decoder_targets.npy
   - train_encoder_masks.npy
   - train_decoder_masks.npy
   - validation_encoder_inputs.npy
   - validation_decoder_inputs.npy
   - validation_decoder_targets.npy
   - validation_encoder_masks.npy
   - validation_decoder_masks.npy
   - test_encoder_inputs.npy
   - test_decoder_inputs.npy
   - test_decoder_targets.npy
   - test_encoder_masks.npy
   - test_decoder_masks.npy
   
   Configuration Files (JSON):
   - train_config.json
   - validation_config.json
   - test_config.json
   
   Data Loaders (Python Pickle):
   - train_loader.pkl
   - validation_loader.pkl
   - test_loader.pkl

7. NEXT STEP
   Proceed to Task 5: Vanilla RNN Encoder-Decoder Model Implementation
"""

with open('encoding_padding_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n✅ Report saved: encoding_padding_report.txt")

print("\n" + "=" * 80)
print("✅ TASK 4 COMPLETED SUCCESSFULLY")
print("=" * 80)
