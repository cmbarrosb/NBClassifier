

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
    # TODO: implement training
    pass

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