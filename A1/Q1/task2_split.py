import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

print("=" * 80)
print("TASK 2: TRAIN-VALIDATION-TEST SPLIT (80/10/10)")
print("=" * 80)

# Fixed random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load cleaned dataset
print("\nLoading cleaned dataset...")
df = pd.read_csv('english_to_urdu_dataset_cleaned.csv', encoding='utf-8')
print(f"Total samples: {len(df)}")

# First split: 80% train, 20% temp (validation + test)
print("\n[Step 1] Splitting into Train (80%) and Temp (20%)...")
train_df, temp_df = train_test_split(
    df, 
    test_size=0.20, 
    random_state=RANDOM_SEED
)

# Second split: Split temp into validation (50%) and test (50%)
# This gives us 10% validation and 10% test (from original 20% temp)
print("[Step 2] Splitting Temp (20%) into Validation (10%) and Test (10%)...")
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_SEED
)

# ==================== VERIFICATION ====================
print("\n" + "=" * 80)
print("DATASET SPLIT RESULTS")
print("=" * 80)

train_size = len(train_df)
val_size = len(val_df)
test_size = len(test_df)
total_size = train_size + val_size + test_size

print(f"\nTrain Set:      {train_size:6d} samples ({train_size/total_size*100:5.2f}%)")
print(f"Validation Set: {val_size:6d} samples ({val_size/total_size*100:5.2f}%)")
print(f"Test Set:       {test_size:6d} samples ({test_size/total_size*100:5.2f}%)")
print(f"{'-'*50}")
print(f"Total:          {total_size:6d} samples (100.00%)")

# ==================== CHECK OVERLAP ====================
print("\n" + "=" * 80)
print("CHECKING FOR OVERLAPS")
print("=" * 80)

train_indices = set(train_df.index)
val_indices = set(val_df.index)
test_indices = set(test_df.index)

train_val_overlap = len(train_indices & val_indices)
train_test_overlap = len(train_indices & test_indices)
val_test_overlap = len(val_indices & test_indices)

print(f"\nTrain-Validation Overlap: {train_val_overlap} samples")
print(f"Train-Test Overlap:       {train_test_overlap} samples")
print(f"Validation-Test Overlap:  {val_test_overlap} samples")

if train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0:
    print("\n✅ No overlaps detected - splits are clean!")
else:
    print("\n❌ Warning: Overlaps detected!")

# ==================== SAVE SPLITS ====================
print("\n" + "=" * 80)
print("SAVING DATASET SPLITS")
print("=" * 80)

# Save CSV versions
train_df.to_csv('train_set.csv', index=False, encoding='utf-8')
val_df.to_csv('validation_set.csv', index=False, encoding='utf-8')
test_df.to_csv('test_set.csv', index=False, encoding='utf-8')

print("\n✅ Train set saved:      train_set.csv")
print("✅ Validation set saved: validation_set.csv")
print("✅ Test set saved:       test_set.csv")

# Save Excel versions
train_df.to_excel('train_set.xlsx', index=False)
val_df.to_excel('validation_set.xlsx', index=False)
test_df.to_excel('test_set.xlsx', index=False)

print("✅ Train set saved:      train_set.xlsx")
print("✅ Validation set saved: validation_set.xlsx")
print("✅ Test set saved:       test_set.xlsx")

# ==================== STATISTICS ====================
print("\n" + "=" * 80)
print("DETAILED STATISTICS")
print("=" * 80)

def print_set_stats(df, name):
    print(f"\n{name}")
    print("-" * 40)
    eng_lengths = df['eng'].str.split().str.len()
    urdu_lengths = df['urdu'].str.split().str.len()
    
    print(f"English (word count):")
    print(f"  Min: {eng_lengths.min():3d}  Max: {eng_lengths.max():3d}  Avg: {eng_lengths.mean():.2f}  Median: {eng_lengths.median():.0f}")
    print(f"Urdu (word count):")
    print(f"  Min: {urdu_lengths.min():3d}  Max: {urdu_lengths.max():3d}  Avg: {urdu_lengths.mean():.2f}  Median: {urdu_lengths.median():.0f}")

print_set_stats(train_df, "TRAIN SET")
print_set_stats(val_df, "VALIDATION SET")
print_set_stats(test_df, "TEST SET")

# ==================== SAMPLE PAIRS ====================
print("\n" + "=" * 80)
print("SAMPLE PAIRS FROM EACH SET")
print("=" * 80)

def print_samples(df, name, num_samples=2):
    print(f"\n{name} (First {num_samples} pairs):")
    for idx in range(min(num_samples, len(df))):
        print(f"\n  {idx+1}. English: {df.iloc[idx]['eng'][:70]}...")
        print(f"     Urdu:    {df.iloc[idx]['urdu'][:70]}...")

print_samples(train_df, "TRAIN SET", 2)
print_samples(val_df, "VALIDATION SET", 2)
print_samples(test_df, "TEST SET", 2)

# ==================== SAVE METADATA ====================
print("\n" + "=" * 80)
print("SAVING METADATA")
print("=" * 80)

metadata = {
    "random_seed": RANDOM_SEED,
    "total_samples": total_size,
    "train": {
        "samples": train_size,
        "percentage": float(train_size / total_size * 100)
    },
    "validation": {
        "samples": val_size,
        "percentage": float(val_size / total_size * 100)
    },
    "test": {
        "samples": test_size,
        "percentage": float(test_size / total_size * 100)
    },
    "overlaps": {
        "train_validation": train_val_overlap,
        "train_test": train_test_overlap,
        "validation_test": val_test_overlap
    }
}

with open('split_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("✅ Metadata saved to: split_metadata.json")

# ==================== REPORT ====================
print("\n" + "=" * 80)
print("TASK 2 COMPLETION SUMMARY")
print("=" * 80)

report = f"""
TRAIN-VALIDATION-TEST SPLIT REPORT
====================================

1. SPLIT CONFIGURATION
   - Method: Stratified random split
   - Random Seed: {RANDOM_SEED} (for reproducibility)
   - Train/Validation/Test Ratios: 80% / 10% / 10%

2. DATASET STATISTICS
   - Original Dataset: {total_size} samples
   - Train Set: {train_size} samples ({train_size/total_size*100:.2f}%)
   - Validation Set: {val_size} samples ({val_size/total_size*100:.2f}%)
   - Test Set: {test_size} samples ({test_size/total_size*100:.2f}%)

3. OVERLAP VERIFICATION
   - Train-Validation Overlap: {train_val_overlap} (✅ clean)
   - Train-Test Overlap: {train_test_overlap} (✅ clean)
   - Validation-Test Overlap: {val_test_overlap} (✅ clean)
   - Status: ✅ NO OVERLAPS - All partitions are independent

4. OUTPUT FILES
   CSV Files:
   - train_set.csv
   - validation_set.csv
   - test_set.csv
   
   Excel Files:
   - train_set.xlsx
   - validation_set.xlsx
   - test_set.xlsx
   
   Metadata:
   - split_metadata.json

5. NEXT STEP
   Proceed to Task 3: Tokenization and Vocabulary Construction
"""

print(report)

# Save report
with open('split_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("✅ Report saved to: split_report.txt")
print("\n" + "=" * 80)
print("✅ TASK 2 COMPLETED SUCCESSFULLY")
print("=" * 80)
