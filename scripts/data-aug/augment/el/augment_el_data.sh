#!/bin/bash

# This script performs data augmentation for the provided language's entity linking training file, looping through the
# designated distance thresholds

LANG=$1  # [es, en, fr, it, pt]
MODEL_TYPE=$2
DISTANCE_THRESHOLDS=(0.75 0.8 0.85 0.9)

for DISTANCE_THRESHOLD in "${DISTANCE_THRESHOLDS[@]}"; do
  if [ "$MODEL_TYPE" = "fasttext" ]; then
    python3 augment_el_data.py --lang "${LANG}" --distance_threshold "${DISTANCE_THRESHOLD}"
  else
    python3 augment_el_data.py --lang "${LANG}" --distance_threshold "${DISTANCE_THRESHOLD}" --word2vec
  fi
done
