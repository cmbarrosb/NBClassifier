#!/usr/bin/env python3
"""
Compute and display the top words that most strongly predict
positive vs. negative sentiment in the trained Naïve Bayes model.
"""

import pickle
import math

# CONFIG
MODEL_FILE = "model.pkl"                           #  saved model
VOCAB_FILE = "moviereview/aclImdb/imdb.vocab"      # vocabulary file

# Load model parameters 
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Extract likelihood arrays 
pos_probs = model["likelihoods"]["pos"]
neg_probs = model["likelihoods"]["neg"]

# Load voc
with open(VOCAB_FILE, "r", encoding="utf8") as f:
    vocab = [w.strip() for w in f]

# Compute log‑odds for each word
# log‑odds(w) = log P(w|pos) – log P(w|neg)
log_odds = []
for p, n in zip(pos_probs, neg_probs):
    log_odds.append(math.log(p) - math.log(n))

# Identify top indicators
top_pos_idx = sorted(range(len(log_odds)), key=lambda i: -log_odds[i])[:20]
top_neg_idx = sorted(range(len(log_odds)), key=lambda i: log_odds[i])[:20]

print("Top 20 words indicating POSITIVE sentiment:")
for idx in top_pos_idx:
    print(f"{vocab[idx]:<15} log‑odds = {log_odds[idx]:.4f}")

print("\nTop 20 words indicating NEGATIVE sentiment:")
for idx in top_neg_idx:
    print(f"{vocab[idx]:<15} log‑odds = {log_odds[idx]:.4f}")