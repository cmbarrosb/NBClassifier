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

PUNCT_RE = re.compile(r'([^\w\s])') # Matches any non-word character or whitespace

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


def process_reviews(input_dir, output_file, vocab, word2idx):
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
            counts = Counter(tokens)
            # Build BOW vector
            vec = [counts.get(word, 0) for word in vocab]
            # Write label + feature counts
            out_f.write(label + ' ' + ' '.join(str(v) for v in vec) + '\n')

def parse_args():
    """Parse command-line arguments for input-dir, vocab-file, and output-file."""
    parser = argparse.ArgumentParser(
        description='Convert raw reviews into BOW feature vectors.'
    )
    parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='Root folder containing "pos" and "neg" subdirs'
    )
    parser.add_argument(
        '--vocab-file', '-v',
        required=True,
        help='Path to vocabulary file (one word per line)'
    )
    parser.add_argument(
        '--output-file', '-o',
        required=True,
        help='File path to write the vectorized output'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    vocab, word2idx = load_vocab(args.vocab_file)
    process_reviews(args.input_dir, args.output_file, vocab, word2idx)

    print(f"Loaded {len(vocab)} words from vocabulary.") 
    print(f"INPUT DIR   = {args.input_dir}")
    print(f"VOCAB FILE  = {args.vocab_file}")
    print(f"OUTPUT FILE = {args.output_file}")

if __name__ == '__main__':
    main()