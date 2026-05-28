import pandas as pd
import json
from collections import Counter
import pickle

print("=" * 80)
print("TASK 3: TOKENIZATION AND VOCABULARY CONSTRUCTION")
print("=" * 80)

# Special tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

print("\nSpecial Tokens Reserved:")
print(f"  {PAD_IDX}: {PAD_TOKEN} (Padding)")
print(f"  {UNK_IDX}: {UNK_TOKEN} (Unknown)")
print(f"  {BOS_IDX}: {BOS_TOKEN} (Beginning of Sequence)")
print(f"  {EOS_IDX}: {EOS_TOKEN} (End of Sequence)")

# ==================== LOAD TRAIN SET ====================
print("\n" + "=" * 80)
print("LOADING TRAINING DATA")
print("=" * 80)

train_df = pd.read_csv('train_set.csv', encoding='utf-8')
print(f"\nTrain set samples: {len(train_df)}")

# ==================== ENGLISH TOKENIZATION ====================
print("\n" + "=" * 80)
print("ENGLISH TOKENIZATION")
print("=" * 80)

def tokenize_english(text):
    """Simple word-level tokenization for English"""
    return text.lower().split()

# Tokenize all English sentences
print("\nTokenizing {0} English sentences...".format(len(train_df)))
english_tokens = [tokenize_english(sent) for sent in train_df['eng']]

# Count word frequencies
english_counter = Counter()
for tokens in english_tokens:
    english_counter.update(tokens)

print(f"Total unique English words: {len(english_counter)}")
print(f"Total English tokens: {sum(english_counter.values())}")

# Create English vocabulary
english_vocab = {word: idx + 4 for idx, word in enumerate(sorted(english_counter.keys()))}
english_vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX, BOS_TOKEN: BOS_IDX, EOS_TOKEN: EOS_IDX, **english_vocab}

print(f"\nEnglish Vocabulary Size: {len(english_vocab)}")
print(f"  - Special tokens: 4")
print(f"  - Regular words: {len(english_vocab) - 4}")

# Show most common words
print("\nMost Common English Words (top 20):")
for i, (word, count) in enumerate(english_counter.most_common(20), 1):
    print(f"  {i:2d}. {word:20s} - {count:5d} occurrences")

# ==================== URDU TOKENIZATION ====================
print("\n" + "=" * 80)
print("URDU TOKENIZATION")
print("=" * 80)

def tokenize_urdu(text):
    """Simple word-level tokenization for Urdu"""
    # Urdu is already in Unicode, just split by whitespace
    return text.split()

# Tokenize all Urdu sentences
print("\nTokenizing {0} Urdu sentences...".format(len(train_df)))
urdu_tokens = [tokenize_urdu(sent) for sent in train_df['urdu']]

# Count word frequencies
urdu_counter = Counter()
for tokens in urdu_tokens:
    urdu_counter.update(tokens)

print(f"Total unique Urdu words: {len(urdu_counter)}")
print(f"Total Urdu tokens: {sum(urdu_counter.values())}")

# Create Urdu vocabulary
urdu_vocab = {word: idx + 4 for idx, word in enumerate(sorted(urdu_counter.keys()))}
urdu_vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX, BOS_TOKEN: BOS_IDX, EOS_TOKEN: EOS_IDX, **urdu_vocab}

print(f"\nUrdu Vocabulary Size: {len(urdu_vocab)}")
print(f"  - Special tokens: 4")
print(f"  - Regular words: {len(urdu_vocab) - 4}")

# Show most common Urdu words
print("\nMost Common Urdu Words (top 20):")
for i, (word, count) in enumerate(urdu_counter.most_common(20), 1):
    print(f"  {i:2d}. {word:20s} - {count:5d} occurrences")

# ==================== CREATE REVERSE VOCABULARIES ====================
print("\n" + "=" * 80)
print("CREATING REVERSE VOCABULARIES (ID → WORD)")
print("=" * 80)

english_reverse_vocab = {v: k for k, v in english_vocab.items()}
urdu_reverse_vocab = {v: k for k, v in urdu_vocab.items()}

print("\n✅ English reverse vocabulary created")
print("✅ Urdu reverse vocabulary created")

# ==================== SAVE VOCABULARIES ====================
print("\n" + "=" * 80)
print("SAVING VOCABULARIES")
print("=" * 80)

# Save as JSON
with open('english_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(english_vocab, f, indent=2, ensure_ascii=False)

with open('urdu_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(urdu_vocab, f, indent=2, ensure_ascii=False)

with open('english_reverse_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(english_reverse_vocab, f, indent=2, ensure_ascii=False)

with open('urdu_reverse_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(urdu_reverse_vocab, f, indent=2, ensure_ascii=False)

print("✅ Saved: english_vocab.json")
print("✅ Saved: urdu_vocab.json")
print("✅ Saved: english_reverse_vocab.json")
print("✅ Saved: urdu_reverse_vocab.json")

# Save as Python pickle for faster loading
with open('english_vocab.pkl', 'wb') as f:
    pickle.dump(english_vocab, f)

with open('urdu_vocab.pkl', 'wb') as f:
    pickle.dump(urdu_vocab, f)

with open('english_reverse_vocab.pkl', 'wb') as f:
    pickle.dump(english_reverse_vocab, f)

with open('urdu_reverse_vocab.pkl', 'wb') as f:
    pickle.dump(urdu_reverse_vocab, f)

print("✅ Saved: english_vocab.pkl")
print("✅ Saved: urdu_vocab.pkl")
print("✅ Saved: english_reverse_vocab.pkl")
print("✅ Saved: urdu_reverse_vocab.pkl")

# ==================== TOKENIZATION EXAMPLES ====================
print("\n" + "=" * 80)
print("TOKENIZATION EXAMPLES")
print("=" * 80)

print("\nExample 1:")
sample_eng = train_df.iloc[0]['eng']
sample_urdu = train_df.iloc[0]['urdu']

tokens_eng = tokenize_english(sample_eng)
tokens_urdu = tokenize_urdu(sample_urdu)

print(f"Original English: {sample_eng}")
print(f"Tokens: {tokens_eng}")
print(f"Token count: {len(tokens_eng)}")

print(f"\nOriginal Urdu: {sample_urdu}")
print(f"Tokens: {tokens_urdu}")
print(f"Token count: {len(tokens_urdu)}")

# ==================== VOCABULARY USAGE STATISTICS ====================
print("\n" + "=" * 80)
print("VOCABULARY COVERAGE ANALYSIS")
print("=" * 80)

# Calculate coverage on training set
english_coverage = sum(1 for tokens in english_tokens for token in tokens if token in english_vocab) / sum(len(tokens) for tokens in english_tokens) * 100
urdu_coverage = sum(1 for tokens in urdu_tokens for token in tokens if token in urdu_vocab) / sum(len(tokens) for tokens in urdu_tokens) * 100

print(f"\nEnglish Coverage: {english_coverage:.2f}% of training tokens")
print(f"Urdu Coverage: {urdu_coverage:.2f}% of training tokens")

# ==================== SAVE METADATA ====================
print("\n" + "=" * 80)
print("SAVING VOCABULARY METADATA")
print("=" * 80)

vocab_metadata = {
    "english": {
        "vocab_size": len(english_vocab),
        "unique_words": len(english_vocab) - 4,
        "coverage": english_coverage
    },
    "urdu": {
        "vocab_size": len(urdu_vocab),
        "unique_words": len(urdu_vocab) - 4,
        "coverage": urdu_coverage
    },
    "special_tokens": {
        "pad": PAD_TOKEN,
        "unk": UNK_TOKEN,
        "bos": BOS_TOKEN,
        "eos": EOS_TOKEN
    },
    "special_token_indices": {
        "pad_idx": PAD_IDX,
        "unk_idx": UNK_IDX,
        "bos_idx": BOS_IDX,
        "eos_idx": EOS_IDX
    }
}

with open('vocab_metadata.json', 'w') as f:
    json.dump(vocab_metadata, f, indent=4)

print("✅ Metadata saved: vocab_metadata.json")

# ==================== SAVE REPORT ====================
report = f"""
TOKENIZATION AND VOCABULARY CONSTRUCTION REPORT
===============================================

1. TOKENIZATION STRATEGY
   - Method: Word-level tokenization (whitespace split)
   - English: Convert to lowercase before splitting
   - Urdu: Split on whitespace (preserve Unicode)

2. SPECIAL TOKENS
   - <pad> (Index {PAD_IDX}): Padding token for sequence alignment
   - <unk> (Index {UNK_IDX}): Unknown token for OOV words
   - <bos> (Index {BOS_IDX}): Beginning of sequence marker
   - <eos> (Index {EOS_IDX}): End of sequence marker

3. ENGLISH VOCABULARY
   - Total Vocabulary Size: {len(english_vocab)}
   - Special Tokens: 4
   - Unique Words: {len(english_vocab) - 4}
   - Total Training Tokens: {sum(english_counter.values())}
   - Training Set Coverage: {english_coverage:.2f}%

4. URDU VOCABULARY
   - Total Vocabulary Size: {len(urdu_vocab)}
   - Special Tokens: 4
   - Unique Words: {len(urdu_vocab) - 4}
   - Total Training Tokens: {sum(urdu_counter.values())}
   - Training Set Coverage: {urdu_coverage:.2f}%

5. TOP 10 ENGLISH WORDS
"""

for i, (word, count) in enumerate(english_counter.most_common(10), 1):
    report += f"   {i:2d}. {word:20s} {count:6d} occurrences\n"

report += "\n6. TOP 10 URDU WORDS\n"
for i, (word, count) in enumerate(urdu_counter.most_common(10), 1):
    report += f"   {i:2d}. {word:20s} {count:6d} occurrences\n"

report += f"""
7. OUTPUT FILES
   Vocabularies (JSON):
   - english_vocab.json (word → index mapping)
   - urdu_vocab.json (word → index mapping)
   - english_reverse_vocab.json (index → word mapping)
   - urdu_reverse_vocab.json (index → word mapping)
   
   Vocabularies (Python Pickle):
   - english_vocab.pkl
   - urdu_vocab.pkl
   - english_reverse_vocab.pkl
   - urdu_reverse_vocab.pkl
   
   Metadata:
   - vocab_metadata.json

8. NEXT STEP
   Proceed to Task 4: Sequence Encoding, Padding, and Batching
"""

with open('tokenization_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("✅ Report saved: tokenization_report.txt")

print("\n" + "=" * 80)
print("✅ TASK 3 COMPLETED SUCCESSFULLY")
print("=" * 80)
