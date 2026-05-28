import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# ==================== STEP 1: LOAD AND INSPECT DATASET ====================
print("=" * 80)
print("STEP 1: LOADING AND INSPECTING DATASET")
print("=" * 80)

dataset_path = "english_to_urdu_dataset.xlsx"
df = pd.read_excel(dataset_path)

print(f"\nDataset Shape: {df.shape}")
print(f"Total Samples: {len(df)}")
print(f"\nColumn Names: {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 samples:")
print(df.head())

# ==================== STEP 2: EXPLORATORY DATA ANALYSIS ====================
print("\n" + "=" * 80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Statistics:")
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"\n{col}:")
        print(f"  - Non-null count: {df[col].notna().sum()}")
        print(f"  - Sample lengths (chars): min={df[col].str.len().min()}, max={df[col].str.len().max()}, avg={df[col].str.len().mean():.2f}")
        print(f"  - First 3 samples:")
        for i, sample in enumerate(df[col].head(3)):
            print(f"    {i+1}. {repr(sample)[:100]}")

# ==================== STEP 3: PREPROCESSING FUNCTIONS ====================
print("\n" + "=" * 80)
print("STEP 3: IMPLEMENTING PREPROCESSING FUNCTIONS")
print("=" * 80)

def clean_english(text):
    """
    Clean English text:
    - Remove extra whitespace
    - Normalize punctuation
    - Handle special characters
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize punctuation
    text = text.replace('–', '-')  # normalize dashes
    text = text.replace(''', "'")   # normalize quotes
    text = text.replace(''', "'")
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove non-ASCII characters except common punctuation
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text.strip()

def clean_urdu(text):
    """
    Clean Urdu text:
    - Remove extra whitespace
    - Normalize punctuation
    - Keep Urdu characters
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize Unicode (NFD to NFC)
    text = unicodedata.normalize('NFC', text)
    
    # Normalize punctuation
    text = text.replace('–', '-')
    text = text.replace(''', "'")
    text = text.replace(''', "'")
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    return text.strip()

def is_valid_pair(eng_text, urdu_text, min_length=1, max_length=200):
    """
    Check if a sentence pair is valid:
    - Both texts are non-empty
    - Reasonable length (within min_length and max_length)
    - Have similar word count (with some tolerance)
    """
    if not eng_text or not urdu_text:
        return False
    
    eng_len = len(eng_text.split())
    urdu_len = len(urdu_text.split())
    
    # Check length constraints
    if eng_len < min_length or eng_len > max_length:
        return False
    if urdu_len < min_length or urdu_len > max_length:
        return False
    
    # Check if lengths are reasonably similar (allow up to 50% difference)
    max_allowed = max(eng_len, urdu_len)
    min_allowed = min(eng_len, urdu_len)
    if max_allowed > 0 and min_allowed / max_allowed < 0.3:
        return False
    
    # Check if text contains mostly valid characters (not corrupted)
    # Should have reasonable character diversity
    if len(set(eng_text)) < 3 or len(set(urdu_text)) < 3:
        return False
    
    return True

# ==================== STEP 4: APPLY PREPROCESSING ====================
print("\nApplying preprocessing pipelines...")

# Identify column names
col_names = df.columns.tolist()
if len(col_names) >= 2:
    eng_col = col_names[0]
    urdu_col = col_names[1]
else:
    eng_col, urdu_col = col_names[0], col_names[0]

print(f"English column: {eng_col}")
print(f"Urdu column: {urdu_col}")

# Create copies for processing
df_clean = df.copy()

# Clean English sentences
print(f"\nCleaning {eng_col}...")
df_clean[eng_col] = df_clean[eng_col].apply(clean_english)

# Clean Urdu sentences
print(f"Cleaning {urdu_col}...")
df_clean[urdu_col] = df_clean[urdu_col].apply(clean_urdu)

# ==================== STEP 5: REMOVE CORRUPTED SAMPLES ====================
print("\n" + "=" * 80)
print("STEP 5: REMOVING CORRUPTED SAMPLES")
print("=" * 80)

initial_count = len(df_clean)
print(f"\nInitial samples: {initial_count}")

# Remove rows with missing values
df_clean = df_clean.dropna(subset=[eng_col, urdu_col])
print(f"After removing missing values: {len(df_clean)} (removed {initial_count - len(df_clean)})")

# Remove empty strings
initial_count = len(df_clean)
df_clean = df_clean[(df_clean[eng_col].str.len() > 0) & (df_clean[urdu_col].str.len() > 0)]
print(f"After removing empty strings: {len(df_clean)} (removed {initial_count - len(df_clean)})")

# Remove corrupted pairs
initial_count = len(df_clean)
valid_mask = df_clean.apply(
    lambda row: is_valid_pair(str(row[eng_col]), str(row[urdu_col])), 
    axis=1
)
df_clean = df_clean[valid_mask]
print(f"After removing corrupted pairs: {len(df_clean)} (removed {initial_count - len(df_clean)})")

# Remove duplicates
initial_count = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=[eng_col, urdu_col], keep='first')
print(f"After removing duplicates: {len(df_clean)} (removed {initial_count - len(df_clean)})")

# ==================== STEP 6: STATISTICS OF CLEANED DATA ====================
print("\n" + "=" * 80)
print("STEP 6: CLEANED DATASET STATISTICS")
print("=" * 80)

print(f"\nFinal Dataset Shape: {df_clean.shape}")
print(f"Final Total Samples: {len(df_clean)}")

print(f"\nEnglish Text Statistics:")
eng_lengths = df_clean[eng_col].str.split().str.len()
print(f"  - Word count: min={eng_lengths.min()}, max={eng_lengths.max()}, avg={eng_lengths.mean():.2f}, median={eng_lengths.median():.0f}")
print(f"  - Character count: min={df_clean[eng_col].str.len().min()}, max={df_clean[eng_col].str.len().max()}, avg={df_clean[eng_col].str.len().mean():.2f}")

print(f"\nUrdu Text Statistics:")
urdu_lengths = df_clean[urdu_col].str.split().str.len()
print(f"  - Word count: min={urdu_lengths.min()}, max={urdu_lengths.max()}, avg={urdu_lengths.mean():.2f}, median={urdu_lengths.median():.0f}")
print(f"  - Character count: min={df_clean[urdu_col].str.len().min()}, max={df_clean[urdu_col].str.len().max()}, avg={df_clean[urdu_col].str.len().mean():.2f}")

print(f"\nSample 5 cleaned pairs:")
for idx in range(min(5, len(df_clean))):
    print(f"\n{idx+1}. English: {df_clean.iloc[idx][eng_col][:80]}")
    print(f"   Urdu:    {df_clean.iloc[idx][urdu_col][:80]}")

# ==================== STEP 7: SAVE CLEANED DATA ====================
print("\n" + "=" * 80)
print("STEP 7: SAVING CLEANED DATASET")
print("=" * 80)

# Save as Excel
output_excel = "english_to_urdu_dataset_cleaned.xlsx"
df_clean.to_excel(output_excel, index=False)
print(f"\nSaved cleaned dataset to: {output_excel}")

# Save as CSV for easier access
output_csv = "english_to_urdu_dataset_cleaned.csv"
df_clean.to_csv(output_csv, index=False, encoding='utf-8')
print(f"Saved cleaned dataset to: {output_csv}")

# Save statistics report
with open("preprocessing_report.txt", "w", encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("DATA PREPROCESSING REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("DATASET INFORMATION\n")
    f.write("-" * 80 + "\n")
    f.write(f"Original samples: {initial_count}\n")
    f.write(f"Cleaned samples: {len(df_clean)}\n")
    f.write(f"Removed samples: {initial_count - len(df_clean)}\n")
    f.write(f"Retention rate: {(len(df_clean)/initial_count)*100:.2f}%\n\n")
    
    f.write("ENGLISH TEXT STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Word count - Min: {eng_lengths.min()}, Max: {eng_lengths.max()}\n")
    f.write(f"Word count - Average: {eng_lengths.mean():.2f}, Median: {eng_lengths.median():.0f}\n")
    f.write(f"Character count - Min: {df_clean[eng_col].str.len().min()}, Max: {df_clean[eng_col].str.len().max()}\n")
    f.write(f"Character count - Average: {df_clean[eng_col].str.len().mean():.2f}\n\n")
    
    f.write("URDU TEXT STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Word count - Min: {urdu_lengths.min()}, Max: {urdu_lengths.max()}\n")
    f.write(f"Word count - Average: {urdu_lengths.mean():.2f}, Median: {urdu_lengths.median():.0f}\n")
    f.write(f"Character count - Min: {df_clean[urdu_col].str.len().min()}, Max: {df_clean[urdu_col].str.len().max()}\n")
    f.write(f"Character count - Average: {df_clean[urdu_col].str.len().mean():.2f}\n\n")
    
    f.write("SAMPLE PAIRS (First 10)\n")
    f.write("-" * 80 + "\n")
    for idx in range(min(10, len(df_clean))):
        f.write(f"\n{idx+1}. English: {df_clean.iloc[idx][eng_col]}\n")
        f.write(f"   Urdu:    {df_clean.iloc[idx][urdu_col]}\n")

print(f"Saved preprocessing report to: preprocessing_report.txt")

print("\n" + "=" * 80)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 80)
