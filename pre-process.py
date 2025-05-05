#!/usr/bin/env python3
"""
pre-process.py

Read a directory of raw movie-review files (train or test),
separate punctuation, lowercase, and output BOW vectors
using a fixed vocabulary.
"""

import argparse
import os
import re
from collections import Counter

NEG_TOKENS = {"not", "never", "n't", "no"}
PUNCTUATION = {".", ",", "!", "?", ";", ":"}

PUNCT_RE = re.compile(r'([^\w\s])') #matches any non-word character or whitespace

def load_vocab(path):
    """Read a vocab file (one word per line) and return (vocab_list, word2idx_dict)."""
    vocab = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            w = line.strip() 
            if w:
                vocab.append(w)
    word2idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word2idx

def review_paths(input_dir):
    """Yield (filepath, label) tuples for all '.txt' files under 'pos' and 'neg'."""
    for label in ('pos', 'neg'):
        dir_path = os.path.join(input_dir, label)
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Expected directory: {dir_path}")
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                yield os.path.join(dir_path, filename), label


def process_reviews(input_dir, output_file, vocab, word2idx, negation=False):
    """
    Process each review file and write BOW vectors to output_file.
    """
    with open(output_file, 'w', encoding='utf8') as out_f:
        for filepath, label in review_paths(input_dir):
            # Read raw text
            with open(filepath, 'r', encoding='utf8') as f:
                text = f.read()
            # Normalize: lowercase and separate punctuation
            text = text.lower()
            text = PUNCT_RE.sub(r' \1 ', text)
            # Tokenize and count
            tokens = text.split()
            # Apply negation marking if enabled
            if negation:
                filtered = []
                neg_scope = False
                for tok in tokens:
                    if tok in NEG_TOKENS:
                        neg_scope = True
                        continue
                    if tok in PUNCTUATION:
                        neg_scope = False
                        continue
                    if neg_scope:
                        continue
                    filtered.append(tok)
                tokens = filtered
            counts = Counter(tokens)
            # Build BOW vector
            vec = [counts.get(word, 0) for word in vocab]
            # Build sparse representation: only nonzero features as "index:count"
            sparse_feats = [f"{i}:{cnt}" for i, cnt in enumerate(vec, start=1) if cnt > 0]
            # Write label + sparse feature list
            out_f.write(label + ' ' + ' '.join(sparse_feats) + '\n')

def parse_args():
    """Parse command-line arguments for input-dir, vocab-file, and output-file."""
    parser = argparse.ArgumentParser(
        description='Convert raw reviews into BOW feature vectors.' \
        '\test output = test.vectors' \
        'Defaults to train.vectors a moviereview/aclImdb/train and moviereview/aclImdb/imdb.vocab'
    )
    parser.add_argument(
        '--input-dir', '-i',
        default='moviereview/aclImdb/train',
        help='Root folder containing "pos" and "neg" subdirs'
    )
    parser.add_argument(
        '--vocab-file', '-v',
        default='moviereview/aclImdb/imdb.vocab',
        help='Path to vocabulary file (one word per line)'
    )
    parser.add_argument(
        '--output-file', '-o',
        default='train.vectors',
        help='File path to write the vectorized output'
    )
    parser.add_argument(
        '--negation', '-n',
        action='store_true',
        help='Enable negation marking: prepend NOT_ to tokens until punctuation'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    vocab, word2idx = load_vocab(args.vocab_file)
    process_reviews(args.input_dir, args.output_file, vocab, word2idx, args.negation)

    print(f"Loaded {len(vocab)} words from vocabulary.") 
    print(f"INPUT DIR   = {args.input_dir}")
    print(f"VOCAB FILE  = {args.vocab_file}")
    print(f"OUTPUT FILE = {args.output_file}")

if __name__ == '__main__':
    main()