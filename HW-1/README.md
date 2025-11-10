# Finding Similar Items: Textually Similar Documents

This project implements text similarity technique using shingling, MinHashing, and Locality Sensitive Hashing (LSH).

## Requirements

- Python 3.11 or higher
- No external dependencies required (uses only the Python standard library)

## Setup

No installation needed! The script uses only Python's built-in libraries:
- `re`, `os`, `glob`, `random`, `hashlib`, `typing`, `collections`

## How to Run

1. Navigate to the `HW-1` directory:
   ```bash
   cd HW-1
   ```

2. Run the main script:
   ```bash
   python main.py
   ```
   
   Or run the notebook version:
   ```
   text_similarity.ipynb
   ```

## What It Does

The script processes text files from the `data/` directory and performs:

1. **Text Preprocessing**: Removes spaces, symbols, and normalizes text
2. **Shingling**: Creates k-shingles (k-grams) from preprocessed text
3. **Jaccard Similarity**: Computes exact similarity between document pairs
4. **MinHashing**: Creates signatures to estimate similarity
5. **LSH**: Uses Locality Sensitive Hashing to efficiently find similar document pairs

## Output

When run, the script prints:
- Preprocessing statistics and file details (e.g., document length before and after cleaning)
- Number of unique shingles per file
- Exact Jaccard similarity values between all document pairs
- MinHash-based similarity estimates along with error analysis compared to exact values
- List of document pairs detected as similar by LSH and the similarity threshold used

## Data Files

The script expects text files in the `data/` directory matching the pattern `text*.txt`. The included data directory already contains `text1.txt` through `text13.txt`.

