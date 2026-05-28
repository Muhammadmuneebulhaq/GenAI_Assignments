"""Generate the English-to-Urdu Transformer translation notebook."""
import json

cells = []

def md(source):
    cells.append({"cell_type":"markdown","metadata":{},"source":source.strip().split("\n")})

def code(source):
    cells.append({"cell_type":"code","metadata":{},"source":source.strip().split("\n"),"outputs":[],"execution_count":None})

# ===== CELL 1: Title =====
md("""# Machine Translation using Transformers: English to Urdu
---
**Based on:** "Attention Is All You Need" by Vaswani et al. (2017)

**Objective:**
- Implement a Transformer-based model for English-to-Urdu translation
- Train the model on a parallel corpus (24,525 sentence pairs)
- Evaluate using BLEU score
- Fine-tune a pre-trained mBART model for comparison

**Dataset:** Parallel Corpus for English-Urdu Language""")

# ===== CELL 2: Imports =====
md("## 1. Imports and Setup")

code("""import os
import re
import math
import random
import time
import unicodedata
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

%matplotlib inline

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")""")

# ===== CELL 3: Config =====
md("## 2. Configuration")

code("""class Config:
    # Data
    data_dir = "Dataset"
    en_file = "english-corpus.txt"
    ur_file = "urdu-corpus.txt"

    # Tokenizer
    max_vocab_size = 16000        # BPE vocab size per language
    max_seq_len = 50              # Max sequence length (tokens)

    # Model (Transformer)
    d_model = 256                 # Embedding dimension
    n_heads = 8                   # Number of attention heads
    n_encoder_layers = 3          # Encoder layers
    n_decoder_layers = 3          # Decoder layers
    d_ff = 512                    # Feed-forward hidden dim
    dropout = 0.1

    # Training
    batch_size = 64
    num_epochs = 30
    lr = 0.0001
    warmup_steps = 4000
    label_smoothing = 0.1
    clip_grad = 1.0

    # Split ratios
    train_ratio = 0.9
    val_ratio = 0.05
    test_ratio = 0.05

    # Checkpoints
    checkpoint_dir = "checkpoints"
    save_every = 5

    device = DEVICE

config = Config()
os.makedirs(config.checkpoint_dir, exist_ok=True)
print("Config ready.")""")

# ===== CELL 4: Load & preprocess data =====
md("""## 3. Data Loading and Preprocessing

Load the parallel corpus, clean the text, and split into train/val/test sets.""")

code("""def load_data(config):
    en_path = os.path.join(config.data_dir, config.en_file)
    ur_path = os.path.join(config.data_dir, config.ur_file)

    with open(en_path, "r", encoding="utf-8") as f:
        en_lines = [line.strip() for line in f.readlines()]
    with open(ur_path, "r", encoding="utf-8") as f:
        ur_lines = [line.strip() for line in f.readlines()]

    assert len(en_lines) == len(ur_lines), "Mismatch in number of lines"
    print(f"Loaded {len(en_lines)} sentence pairs")
    return en_lines, ur_lines


def clean_english(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\\s.,!?'\\-]", "", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def clean_urdu(text):
    text = text.strip()
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def preprocess_data(en_lines, ur_lines):
    pairs = []
    for en, ur in zip(en_lines, ur_lines):
        en_clean = clean_english(en)
        ur_clean = clean_urdu(ur)
        if len(en_clean) > 0 and len(ur_clean) > 0:
            pairs.append((en_clean, ur_clean))
    print(f"After cleaning: {len(pairs)} pairs")
    return pairs


en_lines, ur_lines = load_data(config)
pairs = preprocess_data(en_lines, ur_lines)
print(f"\\nSample pairs:")
for i in range(5):
    print(f"  EN: {pairs[i][0]}")
    print(f"  UR: {pairs[i][1]}")
    print()""")

# ===== CELL 5: Tokenizer =====
md("""## 4. Tokenizer (Word-level with BPE-like vocabulary)

We build a simple word-level tokenizer with special tokens. For a production system, you'd use SentencePiece or HuggingFace tokenizers with BPE/WordPiece.""")

code("""class SimpleTokenizer:
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, max_vocab_size=16000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word_freq = Counter()

    def build_vocab(self, sentences):
        for sent in sentences:
            tokens = sent.split()
            self.word_freq.update(tokens)
        most_common = self.word_freq.most_common(self.max_vocab_size - 4)
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        print(f"Vocabulary size: {len(self.word2idx)}")

    def encode(self, sentence, max_len=50):
        tokens = sentence.split()
        ids = [self.SOS]
        for t in tokens[:max_len - 2]:
            ids.append(self.word2idx.get(t, self.UNK))
        ids.append(self.EOS)
        return ids

    def decode(self, ids):
        words = []
        for idx in ids:
            if idx == self.EOS:
                break
            if idx in (self.PAD, self.SOS):
                continue
            words.append(self.idx2word.get(idx, "<UNK>"))
        return " ".join(words)

    @property
    def vocab_size(self):
        return len(self.word2idx)


# Build tokenizers
en_sentences = [p[0] for p in pairs]
ur_sentences = [p[1] for p in pairs]

en_tokenizer = SimpleTokenizer(config.max_vocab_size)
en_tokenizer.build_vocab(en_sentences)

ur_tokenizer = SimpleTokenizer(config.max_vocab_size)
ur_tokenizer.build_vocab(ur_sentences)

# Test
sample = pairs[0]
en_enc = en_tokenizer.encode(sample[0])
print(f"EN: '{sample[0]}' -> {en_enc} -> '{en_tokenizer.decode(en_enc)}'")""")

# ===== CELL 6: Dataset =====
md("## 5. Translation Dataset and DataLoader")

code("""class TranslationDataset(Dataset):
    def __init__(self, pairs, en_tokenizer, ur_tokenizer, max_len=50):
        self.pairs = pairs
        self.en_tok = en_tokenizer
        self.ur_tok = ur_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, ur = self.pairs[idx]
        en_ids = self.en_tok.encode(en, self.max_len)
        ur_ids = self.ur_tok.encode(ur, self.max_len)
        return en_ids, ur_ids


def collate_fn(batch):
    en_batch, ur_batch = zip(*batch)
    en_max = max(len(s) for s in en_batch)
    ur_max = max(len(s) for s in ur_batch)

    en_padded = [s + [SimpleTokenizer.PAD] * (en_max - len(s)) for s in en_batch]
    ur_padded = [s + [SimpleTokenizer.PAD] * (ur_max - len(s)) for s in ur_batch]

    return torch.tensor(en_padded, dtype=torch.long), torch.tensor(ur_padded, dtype=torch.long)


# Split data
random.seed(42)
random.shuffle(pairs)
n = len(pairs)
n_train = int(n * config.train_ratio)
n_val = int(n * config.val_ratio)

train_pairs = pairs[:n_train]
val_pairs = pairs[n_train:n_train + n_val]
test_pairs = pairs[n_train + n_val:]

print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

train_dataset = TranslationDataset(train_pairs, en_tokenizer, ur_tokenizer, config.max_seq_len)
val_dataset = TranslationDataset(val_pairs, en_tokenizer, ur_tokenizer, config.max_seq_len)
test_dataset = TranslationDataset(test_pairs, en_tokenizer, ur_tokenizer, config.max_seq_len)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

print(f"Train batches: {len(train_loader)}")""")

# ===== CELL 7: Positional encoding =====
md("""## 6. Transformer Model

### 6.1 Positional Encoding
Sinusoidal positional encoding as described in "Attention Is All You Need".""")

code("""class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)""")

# ===== CELL 8: Multi-head attention =====
md("### 6.2 Multi-Head Attention")

code("""class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q = self.W_q(query).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.W_o(context), attn""")

# ===== CELL 9: Feed-forward & encoder/decoder layers =====
md("### 6.3 Feed-Forward Network, Encoder and Decoder Layers")

code("""class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        attn_out, attn_weights = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x, attn_weights""")

# ===== CELL 10: Full Transformer =====
md("### 6.4 Full Transformer Model")

code("""class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, n_heads=8,
                 n_enc_layers=3, n_dec_layers=3, d_ff=512, dropout=0.1, max_len=200):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2)  # (B,1,1,S)

    def make_tgt_mask(self, tgt):
        B, T = tgt.shape
        pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
        causal = torch.tril(torch.ones(T, T, device=tgt.device)).bool().unsqueeze(0).unsqueeze(0)
        return pad_mask & causal

    def encode(self, src, src_mask):
        x = self.dropout(self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model)))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.dropout(self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model)))
        for layer in self.decoder_layers:
            x, _ = layer(x, enc_out, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encode(src, src_mask)
        output = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return output


# Build model
model = Transformer(
    src_vocab=en_tokenizer.vocab_size,
    tgt_vocab=ur_tokenizer.vocab_size,
    d_model=config.d_model,
    n_heads=config.n_heads,
    n_enc_layers=config.n_encoder_layers,
    n_dec_layers=config.n_decoder_layers,
    d_ff=config.d_ff,
    dropout=config.dropout,
).to(config.device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}")
print(model)""")

# ===== CELL 11: LR scheduler & training utils =====
md("## 7. LR Scheduler and Training Utilities")

code("""class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
        for p in self.optimizer.param_groups:
            p["lr"] = lr
        return lr


optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
lr_scheduler = TransformerLRScheduler(optimizer, config.d_model, config.warmup_steps)
criterion = nn.CrossEntropyLoss(ignore_index=SimpleTokenizer.PAD, label_smoothing=config.label_smoothing)

print("Optimizer & scheduler ready.")""")

# ===== CELL 12: BLEU score =====
md("## 8. BLEU Score Evaluation")

code("""def compute_bleu(reference, hypothesis, max_n=4):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    if len(hyp_tokens) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter()
        for i in range(len(ref_tokens) - n + 1):
            ngram = tuple(ref_tokens[i:i+n])
            ref_ngrams[ngram] += 1

        hyp_ngrams = Counter()
        for i in range(len(hyp_tokens) - n + 1):
            ngram = tuple(hyp_tokens[i:i+n])
            hyp_ngrams[ngram] += 1

        clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = max(sum(hyp_ngrams.values()), 1)
        precisions.append(clipped / total)

    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n

    # Brevity penalty
    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))

    return bp * math.exp(log_avg)


def corpus_bleu(references, hypotheses):
    scores = [compute_bleu(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    return sum(scores) / max(len(scores), 1)""")

# ===== CELL 13: Greedy decode =====
md("## 9. Greedy Decoding")

code("""@torch.no_grad()
def greedy_decode(model, src, max_len=50):
    model.eval()
    src = src.to(config.device)
    src_mask = model.make_src_mask(src)
    enc_out = model.encode(src, src_mask)

    tgt = torch.tensor([[SimpleTokenizer.SOS]], device=config.device)

    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(tgt)
        output = model.decode(tgt, enc_out, src_mask, tgt_mask)
        next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)
        if next_token.item() == SimpleTokenizer.EOS:
            break

    return tgt.squeeze(0).tolist()


def translate_sentence(sentence, model, en_tokenizer, ur_tokenizer):
    en_clean = clean_english(sentence)
    en_ids = en_tokenizer.encode(en_clean)
    src = torch.tensor([en_ids], dtype=torch.long)
    output_ids = greedy_decode(model, src)
    return ur_tokenizer.decode(output_ids)""")

# ===== CELL 14: Training loop =====
md("""## 10. Training Loop

Teacher forcing: feed ground truth target tokens as decoder input.""")

code("""history = {"train_loss": [], "val_loss": [], "val_bleu": []}

for epoch in range(config.num_epochs):
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (src, tgt) in enumerate(train_loader):
        src = src.to(config.device)
        tgt = tgt.to(config.device)

        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        output = model(src, tgt_input)
        output = output.reshape(-1, output.size(-1))
        tgt_label = tgt_label.reshape(-1)

        loss = criterion(output, tgt_label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        current_lr = lr_scheduler.step()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    history["train_loss"].append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(config.device), tgt.to(config.device)
            output = model(src, tgt[:, :-1])
            output = output.reshape(-1, output.size(-1))
            tgt_label = tgt[:, 1:].reshape(-1)
            val_loss += criterion(output, tgt_label).item()

    avg_val_loss = val_loss / len(val_loader)
    history["val_loss"].append(avg_val_loss)

    # BLEU on sample
    refs, hyps = [], []
    for en_s, ur_s in val_pairs[:100]:
        pred = translate_sentence(en_s, model, en_tokenizer, ur_tokenizer)
        refs.append(ur_s)
        hyps.append(pred)
    bleu = corpus_bleu(refs, hyps)
    history["val_bleu"].append(bleu)

    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1}/{config.num_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
          f"BLEU: {bleu:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")

    if (epoch + 1) % config.save_every == 0:
        path = os.path.join(config.checkpoint_dir, f"epoch_{epoch+1}.pth")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        print(f"  Saved: {path}")

# Save final
torch.save({"epoch": config.num_epochs-1, "model": model.state_dict()},
           os.path.join(config.checkpoint_dir, "best_model.pth"))
print("Training complete!")""")

# ===== CELL 15: Plot =====
md("## 11. Training Curves")

code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history["train_loss"], label="Train", color="#4facfe")
axes[0].plot(history["val_loss"], label="Val", color="#f5576c")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history["val_bleu"], label="Val BLEU", color="#43e97b")
axes[1].set_title("BLEU Score"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(history["train_loss"], label="Train", color="#4facfe")
axes[2].set_title("Train Loss (log)"); axes[2].set_xlabel("Epoch"); axes[2].set_yscale("log"); axes[2].grid(alpha=0.3)

plt.suptitle("Transformer Training Curves - English to Urdu", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()""")

# ===== CELL 16: Test evaluation =====
md("## 12. Test Set Evaluation")

code("""def evaluate_test(model, test_pairs, en_tokenizer, ur_tokenizer, n_display=15):
    model.eval()
    references, hypotheses = [], []

    for en_s, ur_s in test_pairs:
        pred = translate_sentence(en_s, model, en_tokenizer, ur_tokenizer)
        references.append(ur_s)
        hypotheses.append(pred)

    bleu = corpus_bleu(references, hypotheses)
    print(f"Test BLEU Score: {bleu:.4f}")
    print(f"Test samples: {len(test_pairs)}")
    print("\\n" + "="*80)
    print(f"{'English':<35} | {'Predicted Urdu':<35}")
    print("="*80)
    for i in range(min(n_display, len(test_pairs))):
        en_s = test_pairs[i][0]
        ref = test_pairs[i][1]
        hyp = hypotheses[i]
        print(f"EN:   {en_s}")
        print(f"REF:  {ref}")
        print(f"PRED: {hyp}")
        print("-"*80)
    return bleu

test_bleu = evaluate_test(model, test_pairs, en_tokenizer, ur_tokenizer)""")

# ===== CELL 17: Interactive translate =====
md("## 13. Interactive Translation")

code("""# Translate custom sentences
test_sentences = [
    "how are you",
    "what is your name",
    "i love my country",
    "the weather is beautiful today",
    "where is the school",
    "he is a good person",
    "please help me",
    "i am going home",
]

print("Custom Translations:")
print("=" * 70)
for sent in test_sentences:
    translation = translate_sentence(sent, model, en_tokenizer, ur_tokenizer)
    print(f"EN: {sent}")
    print(f"UR: {translation}")
    print("-" * 70)""")

# ===== CELL 18: mBART fine-tuning =====
md("""## 14. Fine-tuning Pre-trained mBART (Bonus)

Fine-tune Facebook's mBART-large-50 for comparison. This provides a strong baseline leveraging multilingual pre-training.

> **Note:** This section requires `transformers` and `sentencepiece` packages.
> Install with: `pip install transformers sentencepiece`""")

code("""# Uncomment to run mBART fine-tuning
# Requires: pip install transformers sentencepiece

'''
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

# Load pre-trained mBART
model_name = "facebook/mbart-large-50"
mbart_tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
mbart_model = MBartForConditionalGeneration.from_pretrained(model_name)

mbart_tokenizer.src_lang = "en_XX"
mbart_tokenizer.tgt_lang = "ur_PK"


class MBartTranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=64):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, ur = self.pairs[idx]
        source = self.tokenizer(en, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            target = self.tokenizer(ur, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": target["input_ids"].squeeze(),
        }


mbart_train = MBartTranslationDataset(train_pairs, mbart_tokenizer)
mbart_val = MBartTranslationDataset(val_pairs, mbart_tokenizer)

training_args = Seq2SeqTrainingArguments(
    output_dir="mbart_checkpoints",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_steps=100,
)

data_collator = DataCollatorForSeq2Seq(mbart_tokenizer, model=mbart_model)

trainer = Seq2SeqTrainer(
    model=mbart_model,
    args=training_args,
    train_dataset=mbart_train,
    eval_dataset=mbart_val,
    data_collator=data_collator,
    tokenizer=mbart_tokenizer,
)

trainer.train()
trainer.save_model("mbart_finetuned")

# Test mBART
def translate_mbart(text, model, tokenizer):
    tokenizer.src_lang = "en_XX"
    inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True).to(model.device)
    generated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["ur_PK"], max_length=64)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

print("\\nmBART Translations:")
for sent in test_sentences:
    print(f"EN: {sent}")
    print(f"UR: {translate_mbart(sent, mbart_model, mbart_tokenizer)}")
    print("-" * 50)
'''
print("mBART section ready (uncomment to run).")""")

# ===== CELL 19: Attention visualization =====
md("## 15. Attention Visualization")

code("""@torch.no_grad()
def visualize_attention(sentence, model, en_tokenizer, ur_tokenizer):
    model.eval()
    en_clean = clean_english(sentence)
    en_ids = en_tokenizer.encode(en_clean)
    src = torch.tensor([en_ids], dtype=torch.long).to(config.device)
    src_mask = model.make_src_mask(src)
    enc_out = model.encode(src, src_mask)

    tgt = torch.tensor([[SimpleTokenizer.SOS]], device=config.device)
    attentions = []
    output_tokens = []

    for _ in range(config.max_seq_len):
        tgt_mask = model.make_tgt_mask(tgt)
        x = model.dropout(model.pos_enc(model.tgt_embed(tgt) * math.sqrt(model.d_model)))
        for layer in model.decoder_layers:
            x, attn_w = layer(x, enc_out, src_mask, tgt_mask)
        logits = model.fc_out(x)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)
        attentions.append(attn_w[0, 0, -1, :].cpu().numpy())
        output_tokens.append(next_token.item())
        if next_token.item() == SimpleTokenizer.EOS:
            break

    # Build labels
    src_tokens = [en_tokenizer.idx2word.get(i, "?") for i in en_ids]
    tgt_labels = [ur_tokenizer.idx2word.get(i, "?") for i in output_tokens]

    attn_matrix = np.array(attentions)[:, :len(src_tokens)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn_matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(tgt_labels)))
    ax.set_yticklabels(tgt_labels)
    ax.set_xlabel("Source (English)")
    ax.set_ylabel("Target (Urdu)")
    ax.set_title(f"Cross-Attention: '{sentence}'")
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

# Example
visualize_attention("how are you", model, en_tokenizer, ur_tokenizer)""")

# ===== Build notebook =====
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
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

with open("english_to_urdu_transformer.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook created: english_to_urdu_transformer.ipynb ({len(cells)} cells)")
