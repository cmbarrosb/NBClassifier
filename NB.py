

#!/usr/bin/env python3
"""
NB.py

Train and evaluate a Naïve Bayes classifier using pre-processed BOW vectors.
"""


import argparse
import math
import pickle
import sys

# Load vocabulary to determine feature dimension for sparse vectors
VOCAB_FILE = 'moviereview/aclImdb/imdb.vocab'
vocab = [w.strip() for w in open(VOCAB_FILE, encoding='utf8')]
V = len(vocab)

def parse_args():
    """Parse command-line arguments for training and testing."""
    parser = argparse.ArgumentParser(
        description='Train and test a Naïve Bayes classifier.'
    )
    parser.add_argument(
        '--train-file', '-t',
        help='Path to the training vector file',
        default='train.vectors'
    )
    parser.add_argument(
        '--test-file', '-e',
        help='Path to the test vector file',
        default='test.vectors'
    )
    parser.add_argument(
        '--model-out', '-m',
        help='Output path to save trained model parameters',
        default='model.pkl'
    )
    parser.add_argument(
        '--pred-out', '-p',
        help='Output file for test predictions (one per line), with accuracy on last line',
        default='predictions.txt'
    )
    return parser.parse_args()

def train_nb(train_file, model_out):
    """
    Read training vectors, compute class priors and add-one smoothed likelihoods,
    and save model parameters (priors and likelihoods) to model_out.
    """
    # Read training data
    doc_counts = {}
    token_counts = {}
    total_docs = 0

    with open(train_file, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: # skip empty lines
                continue
            label = parts[0] # first token is pos or neg label
            # Parse sparse "index:count" features into dense count list
            counts = [0] * V
            for spec in parts[1:]:
                idx_str, cnt_str = spec.split(':')
                counts[int(idx_str) - 1] = int(cnt_str)

            doc_counts[label] = doc_counts.get(label, 0) + 1 #NC to compute priors

            if label not in token_counts:
                token_counts[label] = [0]*V

            for i, c in enumerate(counts):
                token_counts[label][i] += c #total number of Wi for each label
            total_docs += 1

    # Compute priors
    priors = {label: doc_counts[label]/total_docs for label in doc_counts}# NC/total_docs

    # Compute smoothed likelihoods
    likelihoods = {}
    for label, counts in token_counts.items():
        total_wc = sum(counts)
        denom = total_wc + V
        likelihoods[label] = [(count + 1)/denom for count in counts]
        # Add-one smoothing: P(Wi|C) = (count(Wi, C) + 1) / (total_wc + V)
        
    # Save model
    model = {'priors': priors, 'likelihoods': likelihoods}
    with open(model_out, 'wb') as mf:
        pickle.dump(model, mf)

def test_nb(test_file, model_in, pred_out):
    """
    Load model parameters from model_in, predict labels for test_file,
    write one prediction per line to pred_out, and append overall accuracy.
    """
    # Load model parameters
    with open(model_in, 'rb') as mf:
        model = pickle.load(mf)
    priors = model['priors']
    likelihoods = model['likelihoods']

    # Precompute logs to avoid overflo to -Inf
    log_priors = {label: math.log(p) for label, p in priors.items()}
    log_likelihoods = {
        label: [math.log(p) for p in probs]
        for label, probs in likelihoods.items()
    }
    # Classify test data
    correct = 0
    total = 0
    with open(pred_out, 'w', encoding='utf8') as out_f:
        with open(test_file, 'r', encoding='utf8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: # skip empty lines
                    continue
                true_label = parts[0] # first token is pos or neg label
                
                # Parse sparse "index:count" features into dense count list
                counts = [0] * V
                for spec in parts[1:]:
                    idx_str, cnt_str = spec.split(':')
                    counts[int(idx_str) - 1] = int(cnt_str)


                # compute scores
                scores = {}
                for label in priors:
                    score = log_priors[label] # log(P(C))
                    for i, c in enumerate(counts):
                        if c:
                            score += c * log_likelihoods[label][i]
                    scores[label] = score

                # predict
                pred = max(scores, key=scores.get) #returns the label whose score is largest
                out_f.write(f"{pred} {true_label}\n")
                if pred == true_label:
                    correct += 1
                total += 1
        # write accuracy
        accuracy = correct/total if total else 0.0
        out_f.write(f"Accuracy: {accuracy:.4f}\n")

def main():
    args = parse_args()
    train_nb(args.train_file, args.model_out)
    test_nb(args.test_file, args.model_out, args.pred_out)

if __name__ == '__main__':
    main()