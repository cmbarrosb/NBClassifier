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

def load_vocab(path): #Read a vocab file (one word per line) and return (vocab, word2idx)
    vocab = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            w = line.strip() 
            if w:
                vocab.append(w) #append the word to vocab
    word2idx = {w: i for i, w in enumerate(vocab)} #create a dictionary with word as key and index as value
    return vocab, word2idx

def parse_args():
    # Create a parser object
    # and add arguments for input directory, vocab file, and output file
    # The input directory should contain "pos" and "neg" subdirectories
    # The vocab file should contain one word per line
    # The output file is where the vectorized output will be written
    # The parser will also display a description of the script
    # when the user runs the script with the -h or --help option
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
    print(f"Loaded {len(vocab)} words from vocabulary.")
    print(f"INPUT DIR   = {args.input_dir}")
    print(f"VOCAB FILE  = {args.vocab_file}")
    print(f"OUTPUT FILE = {args.output_file}")

if __name__ == '__main__':
    main()