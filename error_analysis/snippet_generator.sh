#!/usr/bin/env bash
# Generate snippets of the first 100 words for each misclassified review

SNIPPETS_DIR="error_analysis/snippets"
BAD_FILES="error_analysis/error_files.txt"
NUM_WORDS=100

mkdir -p "$SNIPPETS_DIR"

while read -r filepath; do
  snippet_file="$SNIPPETS_DIR/$(basename "$filepath").snippet"
  echo "=== $filepath ===" > "$snippet_file"
  # Extract first NUM_WORDS words and join back into a single line
  tr -s '[:space:]' '\n' < "$filepath" \
    | head -n "$NUM_WORDS" \
    | paste -sd ' ' - \
    >> "$snippet_file"
done < "$BAD_FILES"
