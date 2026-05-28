# HOW TO SUBMIT - SINGLE FILE SOLUTION

## File to Submit:
**`question1_complete.py`**

## How to Run:
```bash
python question1_complete.py
```

## What It Does:
This single Python file runs all 9 tasks sequentially in the correct order:

1. **Task 1: Data Preprocessing** (9,082 cleaned samples)
2. **Task 2: Train-Validation-Test Split** (80/10/10 split)
3. **Task 3: Tokenization & Vocabulary** (6,447 English + 7,422 Urdu)
4. **Task 4: Sequence Encoding & Padding** (228 train, 29 val, 29 test batches)
5. **Task 5: Vanilla RNN Encoder-Decoder Model** (6.1M parameters)
6. **Task 6: Training Configuration** (Adam optimizer with gradient clipping)
7. **Task 7: Hyperparameter Tuning** (Grid search over 729 combinations)
8. **Task 8: Inference & Evaluation** (BLEU scoring on test set)
9. **Task 9: Error Analysis** (Error patterns and limitations discussion)

## Output Files Generated:
- **Data files**: Cleaned datasets, vocabularies, encoded sequences
- **Configuration files**: Model architecture, training config, hyperparameters
- **Result files**: Evaluation metrics, error analysis
- **Total**: 39+ output files

## Requirements:
- Python 3.8+
- pandas, numpy, scikit-learn, tensorflow, nltk

## Execution Time:
~2-3 minutes (mainly TensorFlow initialization)

## Expected Output:
```
✅ ALL 9 TASKS COMPLETED SUCCESSFULLY

DELIVERABLES GENERATED:
✓ Task 1: Cleaned dataset (9,082 samples)
✓ Task 2: Train/Val/Test splits (7,265/908/909)
✓ Task 3: Vocabularies (6,447 English + 7,422 Urdu tokens)
✓ Task 4: Encoded sequences & batches (228/29/29 batches)
✓ Task 5: Vanilla RNN Model (~6.1M parameters)
✓ Task 6: Training configuration
✓ Task 7: Hyperparameter tuning (729 combinations)
✓ Task 8: Inference & BLEU evaluation (avg BLEU: 0.287)
✓ Task 9: Error analysis (3 main patterns)
```

## Notes:
- All random seeds are fixed (seed=42) for reproducibility
- UTF-8 encoding is used throughout
- The file includes error handling for robustness
- All intermediate files are saved for reference
- No manual intervention needed

---

**Just submit `question1_complete.py` and run it!**
