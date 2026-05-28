# QUESTION 1: DATA PREPROCESSING - STEP-BY-STEP INSTRUCTIONS

## Objective
Design, implement, and apply preprocessing pipelines for English and Urdu sentences from a parallel corpus dataset for English-to-Urdu Neural Machine Translation (NMT) system.

---

## STEP 1: DATASET LOADING AND INSPECTION

### 1.1 Download and Load the Dataset
- **Source**: English-to-Urdu parallel corpus dataset (XLSX format)
- **Location**: `english_to_urdu_dataset.xlsx`
- **Tool**: Use pandas library to load the dataset

```python
import pandas as pd

# Load the dataset
df = pd.read_excel('english_to_urdu_dataset.xlsx')
```

### 1.2 Inspect Dataset Structure
Examine the fundamental properties of the dataset:

**Results:**
- **Dataset Shape**: (9103, 2) - 9103 sentence pairs with 2 columns
- **Columns**: 
  - `eng`: English sentences
  - `urdu`: Urdu sentences
- **Data Types**: Both columns contain string (object) data
- **Missing Values**: 
  - English: 0 missing values (100% complete)
  - Urdu: 1 missing value (99.99% complete)

**Sample Pairs:**
| # | English | Urdu |
|---|---------|------|
| 1 | "the book of the generation of jesus christ the son of david the son of abraham" | "یسوع مسیح ابن داود ابن ابرہام کا نسب نامہ" |
| 2 | "abraham begat isaac and isaac begat jacob..." | "ابراہام سے اضحاق پیدا ہوا اور اضحاق سے یعقوب..." |

---

## STEP 2: DESIGN PREPROCESSING PIPELINES

### 2.1 English Text Preprocessing Pipeline

#### A. Whitespace Normalization
- **Goal**: Remove extra spaces and normalize whitespace
- **Operations**:
  - Strip leading/trailing whitespace
  - Replace multiple consecutive spaces with single space
  - Use regex: `\s+` → ` `

**Example:**
```
Input:  "  the  book   of   life  "
Output: "the book of life"
```

#### B. Punctuation Normalization
- **Goal**: Standardize different representations of the same punctuation
- **Operations**:
  - Normalize dashes: `–` (en-dash) → `-` (hyphen)
  - Normalize quotes: `'` `'` → `'`
  - Normalize double quotes: `"` `"` → `"`

**Example:**
```
Input:  "john's book – a bestseller"
Output: "john's book - a bestseller"
```

#### C. Special Character Handling
- **Goal**: Remove non-essential or corrupted characters
- **Operations**:
  - Remove URLs (http://, https://, www.)
  - Remove HTML tags (<tag>, </tag>)
  - Remove non-ASCII characters (keep only ASCII characters)

**Example:**
```
Input:  "Check <link>http://example.com</link> for more"
Output: "Check  for more"
```

#### D. Final Cleaning
- Strip any remaining whitespace
- Final validation of text integrity

### 2.2 Urdu Text Preprocessing Pipeline

#### A. Whitespace Normalization
- Same as English: normalize spaces
- Use regex: `\s+` → ` `

#### B. Unicode Normalization
- **Goal**: Ensure consistent Unicode representation
- **Operation**: Convert to NFC (Canonical Decomposition, followed by Canonical Composition)
- **Importance**: Critical for non-Latin scripts like Urdu

```python
import unicodedata
urdu_text = unicodedata.normalize('NFC', urdu_text)
```

#### C. Punctuation Normalization
- Same as English text preprocessing
- Normalize dashes, quotes, and other punctuation marks

#### D. Special Character Handling
- Remove URLs
- Remove HTML tags
- Keep all Urdu Unicode characters (unlike English, don't remove non-ASCII)

#### E. Final Cleaning
- Strip whitespace and validate integrity

---

## STEP 3: IMPLEMENTATION DETAILS

### 3.1 English Cleaning Function

```python
def clean_english(text):
    """
    Clean English text with normalization pipeline
    """
    if not isinstance(text, str):
        return ""
    
    # Step 1: Whitespace normalization
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    # Step 2: Punctuation normalization
    text = text.replace('–', '-')      # en-dash to hyphen
    text = text.replace(''', "'")      # curly single quote
    text = text.replace(''', "'")      # another curly quote variant
    text = text.replace('"', '"')      # curly left double quote
    text = text.replace('"', '"')      # curly right double quote
    
    # Step 3: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Step 4: Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Step 5: Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Step 6: Final cleanup
    return text.strip()
```

### 3.2 Urdu Cleaning Function

```python
def clean_urdu(text):
    """
    Clean Urdu text with normalization pipeline
    """
    if not isinstance(text, str):
        return ""
    
    # Step 1: Whitespace normalization
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    # Step 2: Unicode normalization (CRITICAL FOR URDU)
    text = unicodedata.normalize('NFC', text)
    
    # Step 3: Punctuation normalization
    text = text.replace('–', '-')
    text = text.replace(''', "'")
    text = text.replace(''', "'")
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    
    # Step 4: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Step 5: Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Step 6: Final cleanup (preserve Urdu characters)
    return text.strip()
```

---

## STEP 4: REMOVAL OF CORRUPTED SAMPLES

### 4.1 Define Sample Validity Criteria

A sentence pair is considered **valid** if it meets ALL of the following criteria:

#### Criterion 1: Non-Empty Text
```python
if not eng_text or not urdu_text:
    return False  # Invalid if either text is empty
```

#### Criterion 2: Length Constraints
- **Minimum length**: 1 word per sentence
- **Maximum length**: 200 words per sentence
- **Rationale**: Extremely long sequences can cause training issues; very short sequences may lack context

```python
eng_len = len(eng_text.split())
urdu_len = len(urdu_text.split())

if eng_len < 1 or eng_len > 200:
    return False
if urdu_len < 1 or urdu_len > 200:
    return False
```

#### Criterion 3: Length Similarity
- **Tolerance**: Word counts should not differ by more than 70%
- **Rationale**: Drastically different lengths suggest poor alignment or translation quality

```python
max_allowed = max(eng_len, urdu_len)
min_allowed = min(eng_len, urdu_len)

if max_allowed > 0 and min_allowed / max_allowed < 0.3:
    return False  # More than 70% length difference
```

#### Criterion 4: Character Diversity
- **Minimum character diversity**: At least 3 unique characters
- **Rationale**: Prevents corrupted text with repeated characters or gibberish

```python
if len(set(eng_text)) < 3 or len(set(urdu_text)) < 3:
    return False
```

### 4.2 Additional Cleaning Steps

#### Remove Rows with Missing Values
```python
df_clean = df_clean.dropna(subset=['eng', 'urdu'])
```

#### Remove Empty Strings After Cleaning
```python
df_clean = df_clean[(df_clean['eng'].str.len() > 0) & 
                    (df_clean['urdu'].str.len() > 0)]
```

#### Remove Corrupted Pairs
```python
valid_mask = df_clean.apply(
    lambda row: is_valid_pair(str(row['eng']), str(row['urdu'])), 
    axis=1
)
df_clean = df_clean[valid_mask]
```

#### Remove Duplicate Pairs
```python
df_clean = df_clean.drop_duplicates(subset=['eng', 'urdu'], keep='first')
```

---

## STEP 5: QUALITY STATISTICS OF CLEANED DATA

### 5.1 Dataset Size Summary

| Metric | Value |
|--------|-------|
| Original Samples | 9,103 |
| Final Samples | 9,082 |
| Samples Removed | 21 |
| Retention Rate | 99.77% |
| Reason for Removal | 1 missing value, 10 corrupted pairs, 10 duplicates |

### 5.2 English Text Statistics

| Metric | Min | Max | Average | Median |
|--------|-----|-----|---------|--------|
| **Word Count** | 1 | 68 | 20.63 | 20 |
| **Character Count** | 3 | 351 | 104.04 | - |

**Interpretation:**
- English sentences average ~20 words per sentence
- Most sentences are between 10-30 words (typical for translation pairs)
- Maximum 68 words is reasonable for translation data

### 5.3 Urdu Text Statistics

| Metric | Min | Max | Average | Median |
|--------|-----|-----|---------|--------|
| **Word Count** | 1 | 84 | 23.14 | 23 |
| **Character Count** | 4 | 333 | 95.39 | - |

**Interpretation:**
- Urdu sentences average ~23 words per sentence
- Slightly longer on average than English (ratio: 23.14/20.63 ≈ 1.12)
- This is normal due to grammatical differences between English and Urdu

### 5.4 Sample Quality Examples

**Excellent Quality Pair:**
```
English: "and jesse begat david the king and david the king begat solomon 
          of her that had been the wife of urias"
Urdu:    "اور یسّی سے داود بادشاہ پیدا ہوا ۔ اور داود سے سلیمان اس عورت 
          سے پیدا ہوا جو پہلے اوریاہ کی بیوی تھی ۔"
```
- Well-structured, grammatically correct
- Good semantic alignment
- Proper punctuation and Unicode representation

---

## STEP 6: SAVING CLEANED DATASETS

### 6.1 Output File Formats

The cleaned dataset is saved in multiple formats for different use cases:

#### Format 1: Excel (.xlsx)
- **File**: `english_to_urdu_dataset_cleaned.xlsx`
- **Use Case**: Manual inspection, visualization in spreadsheet applications
- **Advantages**: Intuitive interface, easy to view/edit

#### Format 2: CSV (.csv)
- **File**: `english_to_urdu_dataset_cleaned.csv`
- **Use Case**: Data analysis, Python/R scripts, compatibility with standard tools
- **Advantages**: Plain text, widely supported, smaller file size

#### Format 3: Report (.txt)
- **File**: `preprocessing_report.txt`
- **Use Case**: Documentation, review of preprocessing results
- **Contains**: Statistics, sample pairs, removal summary

### 6.2 File Specifications

All cleaned datasets contain:
- **Columns**: `eng` (English), `urdu` (Urdu)
- **Rows**: 9,082 processed sentence pairs
- **Encoding**: UTF-8 (preserves Urdu Unicode characters)
- **Delimiter**: Tab (for CSV) / None (for XLSX)

---

## STEP 7: VALIDATION AND QUALITY ASSURANCE

### 7.1 Verification Checklist

✅ **Dataset Loading**: Successfully loaded 9,103 original pairs
✅ **Structure Inspection**: Confirmed 2 columns (English, Urdu)
✅ **Missing Value Handling**: Removed 1 row with missing Urdu text
✅ **English Preprocessing**: Applied whitespace, punctuation, and ASCII normalization
✅ **Urdu Preprocessing**: Applied whitespace, Unicode normalization, preserved Urdu characters
✅ **Corruption Removal**: Removed 10 corrupted pairs based on validity criteria
✅ **Duplicate Removal**: Removed 10 duplicate entries
✅ **Final Statistics**: Computed and documented language-specific metrics
✅ **File Saving**: Exported to Excel, CSV, and text report formats

### 7.2 Quality Metrics

**Preprocessing Quality Indicators:**
- High retention rate (99.77%) indicates minimal data loss
- Word count distribution is reasonable for translation tasks
- Character count distribution shows good data variety
- Unicode normalization ensures consistent Urdu representation

---

## SUMMARY OF QUESTION 1 COMPLETION

**Task**: Data Preprocessing for English-to-Urdu NMT System

**Deliverables Completed**:

1. ✅ **Dataset Download & Loading**: Loaded `english_to_urdu_dataset.xlsx` (9,103 pairs)

2. ✅ **Structural Inspection**: 
   - Confirmed 2-column structure (English, Urdu)
   - Identified data types and missing values

3. ✅ **English Preprocessing Pipeline**:
   - Whitespace normalization
   - Punctuation standardization
   - URL and HTML tag removal
   - ASCII character filtering

4. ✅ **Urdu Preprocessing Pipeline**:
   - Whitespace normalization
   - Unicode NFC normalization
   - Punctuation standardization
   - URL and HTML tag removal
   - Urdu character preservation

5. ✅ **Corrupted Sample Removal**:
   - Removed 1 missing value row
   - Removed 10 corrupted pairs (length mismatch, low diversity)
   - Removed 10 duplicates
   - Final dataset: 9,082 clean, valid pairs

6. ✅ **Dataset Preparation**:
   - English: avg 20.63 words/sentence, 104.04 chars/sentence
   - Urdu: avg 23.14 words/sentence, 95.39 chars/sentence
   - Exported to Excel, CSV, and report formats

7. ✅ **Documentation**:
   - Step-by-step preprocessing instructions
   - Validity criteria definitions
   - Statistical analysis and validation

**Python Implementation**: `preprocess.py`
- Modular functions for English/Urdu cleaning
- Comprehensive error handling
- Detailed logging and statistics

**Output Files**:
- `english_to_urdu_dataset_cleaned.xlsx` - Cleaned data in Excel format
- `english_to_urdu_dataset_cleaned.csv` - Cleaned data in CSV format
- `preprocessing_report.txt` - Detailed preprocessing report
- `preprocess.py` - Complete Python implementation

---

## NEXT STEPS
Once Question 1 (Data Preprocessing) is complete, proceed with:
- **Question 2**: Train-Validation-Test Split (80/10/10)
- **Question 3**: Tokenization and Vocabulary Construction
- **Question 4**: Sequence Encoding, Padding, and Batching
- ... and subsequent tasks for the NMT system implementation

