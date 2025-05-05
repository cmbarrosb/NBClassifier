#!/usr/bin/env python3
import random
import os

# —— Hardcoded paths ——
BAD_IDX_FILE   = "error_analysis/bad_idx.txt"
VOCAB_FILE     = "moviereview/aclImdb/imdb.vocab"
TEST_VEC_FILE  = "test.vectors"
TEST_DIR       = "moviereview/aclImdb/test"
NUM_SAMPLES    = 10

# Load bad indices
bad_idxs = []
with open(BAD_IDX_FILE, encoding='utf8') as f:
    for line in f:
        line = line.strip()
        if line:
            bad_idxs.append(int(line))

# Load vocab
vocab = [w.strip() for w in open(VOCAB_FILE, encoding='utf8')]

# Load test vectors
test_vecs = [l.strip().split() for l in open(TEST_VEC_FILE, encoding='utf8') if l.strip()]

# Build list of (label, path) for each .txt in test, in the same order
paths = []
for lab in ("pos","neg"):
    folder = os.path.join(TEST_DIR, lab)
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".txt"):
            paths.append((lab, os.path.join(folder, fn)))

# Sample
sampled = random.sample(bad_idxs, min(NUM_SAMPLES, len(bad_idxs)))

for idx in sampled:
    print(f"\n=== Example #{idx} ===")
    # 1) List non-zero tokens
    vec = test_vecs[idx-1]    # idx is 1-based
    print("Tokens:")
    for spec in vec[1:]:
        idx_str, cnt_str = spec.split(':')
        word_index = int(idx_str) - 1
        print(f"  {vocab[word_index]}: {cnt_str}")
    # 2) Show raw file path
    if 1 <= idx <= len(paths):
        lab, path = paths[idx-1]
        print(f"File: {path} (true label: {lab})")
    else:
        print("File: [index out of range]")