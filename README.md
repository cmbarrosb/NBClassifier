# Homework 2: Sentiment Classification with Naïve Bayes

## Project Overview
This project implements a simple binary sentiment classifier using a bag-of-words Naïve Bayes model, trained and tested on the IMDB movie-review dataset. The pipeline consists of:

1. **Pre-processing** raw review text into feature vectors.
2. **Training & Testing** the Naïve Bayes classifier.
3. **Error Analysis** to identify model weaknesses.
4. **Model-weight Analysis** to inspect top positive/negative tokens.

## Repository Structure
```
├── pre-process.py       # Script to convert raw reviews to BOW vectors
├── NB.py                # Script to train and test the Naïve Bayes model
├── analyze_weights.py   # (Optional) Ranks tokens by log‑odds (may need path updates)
├── inspect_errors.py    # (Optional) Samples and inspects misclassified examples (may need path updates)
├── snippet_generator.py  # (Optional) Extracts text snippets around misclassifications (may need relocation)
├── moviereview/         # Original IMDB dataset (train/pos, train/neg, test/pos, test/neg)
├── imdb.vocab           # Vocabulary file (one token per line)
├── train.vectors        # Generated training vectors
├── test.vectors         # Generated test vectors
├── model.pkl            # Serialized Naïve Bayes model
├── predictions.txt      # Predicted labels + accuracy output
└── README.md            # This file
```

## Requirements
- **Python 3.x** (no external dependencies)
- The IMDB dataset directory structure:
  ```
  moviereview/aclImdb/
  ├── train/pos
  ├── train/neg
  ├── test/pos
  └── test/neg
  ```

## Pre-processing: `pre-process.py`
Converts raw `.txt` reviews into sparse BOW vectors.

### Usage
```bash
python3 pre-process.py \
  --input-dir moviereview/aclImdb/train \
  --vocab-file moviereview/aclImdb/imdb.vocab \
  --output-file train.vectors \
  [--negation] [--min-freq <N>]
```
Repeat for the test set:
```bash
python3 pre-process.py \
  --input-dir moviereview/aclImdb/test \
  --vocab-file moviereview/aclImdb/imdb.vocab \
  --output-file test.vectors \
  [--negation] [--min-freq <N>]
```

To see all available options, run:
```bash
python3 pre-process.py -h
```

### Flags
- `--negation`, `-n`  
  Enable negation removal: tokens in a negated scope (after “not”, “never”, etc., until punctuation) are dropped to prevent misleading positive cues.

- `--min-freq`, `-f`  
  Minimum global token frequency to include tokens in the vocabulary and output vectors (default: 1).

## Training & Testing: `NB.py`
Implements a multinomial Naïve Bayes classifier with add‑one smoothing on likelihoods.

### Usage
```bash
python3 NB.py \
  --train train.vectors \
  --test  test.vectors \
  --model model.pkl \
  --pred  predictions.txt \
  [--binary]
```

To see all available options, run:
```bash
python3 NB.py -h
```

### Flags
- `--binary`, `-b`  
  Use binary bag-of-words (presence/absence) instead of raw term frequencies.

### Output
- **`model.pkl`**: Pickled model parameters (class priors & word likelihoods)
- **`predictions.txt`**: One predicted label per line, ending with:

## Error Analysis (Optional)
**Note:** The scripts `inspect_errors.py` and `snippet_generator.py` may need to be moved out of the dataset folder or have file paths updated within the code to locate raw review files correctly.

Use `inspect_errors.py` to sample misclassified reviews, view their non-zero tokens, and inspect the original text for manual categorization of error types (negation, rare tokens, mixed sentiment, etc.).

## Model-weight Analysis
**Note:** The script `analyze_weights.py` may need its internal paths adjusted if it resides outside the project root.

Run `analyze_weights.py` to list the top-N tokens by log‑odds
This surfaces the strongest positive and negative indicators learned by the model.

## Further Improvements
- Include **negation marking** or **binary features** (already supported).
- Apply a **frequency cutoff** on vocabulary to remove rare tokens.
- Integrate a **sentiment lexicon** as additional features.