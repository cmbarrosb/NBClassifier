

#!/usr/bin/env python3
"""
NB.py

Train and evaluate a Naïve Bayes classifier using pre-processed BOW vectors.
"""

import argparse
import math
import pickle
import sys

def parse_args():
    """Parse command-line arguments for training and testing."""
    parser = argparse.ArgumentParser(
        description='Train and test a Naïve Bayes classifier.'
    )
    parser.add_argument(
        '--train-file', '-t', required=True,
        help='Path to the training vector file'
    )
    parser.add_argument(
        '--test-file', '-e', required=True,
        help='Path to the test vector file'
    )
    parser.add_argument(
        '--model-out', '-m', required=True,
        help='Output path to save trained model parameters'
    )
    parser.add_argument(
        '--pred-out', '-p', required=True,
        help='Output file for test predictions (one per line), with accuracy on last line'
    )
    return parser.parse_args()

def train_naive_bayes(train_file, model_out):
    """
    Read training vectors, compute class priors and add-one smoothed likelihoods,
    and save model parameters (priors and likelihoods) to model_out.
    """
    # Read training data
    doc_counts = {}
    token_counts = {}
    total_docs = 0
    V = None
    with open(train_file, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            label = parts[0]
            counts = list(map(int, parts[1:]))
            if V is None:
                V = len(counts)
            doc_counts[label] = doc_counts.get(label, 0) + 1
            if label not in token_counts:
                token_counts[label] = [0]*V
            for i, c in enumerate(counts):
                token_counts[label][i] += c
            total_docs += 1
    # Compute priors
    priors = {label: doc_counts[label]/total_docs for label in doc_counts}
    # Compute smoothed likelihoods
    likelihoods = {}
    for label, counts in token_counts.items():
        total_wc = sum(counts)
        denom = total_wc + V
        likelihoods[label] = [(count + 1)/denom for count in counts]
    # Save model
    model = {'priors': priors, 'likelihoods': likelihoods}
    with open(model_out, 'wb') as mf:
        pickle.dump(model, mf)

def test_naive_bayes(test_file, model_in, pred_out):
    """
    Load model parameters from model_in, predict labels for test_file,
    write one prediction per line to pred_out, and append overall accuracy.
    """
    # TODO: implement testing
    pass

def main():
    args = parse_args()
    train_naive_bayes(args.train_file, args.model_out)
    test_naive_bayes(args.test_file, args.model_out, args.pred_out)

if __name__ == '__main__':
    main()