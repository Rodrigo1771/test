#!/bin/bash

# This script performs data augmentation for the provided dataset's named entity recognition training file, looping
# through the designated distance thresholds

DATASET=$1  # [multicardioner, cantemist, symptemist]
MODEL_TYPE=$2
DISTANCE_THRESHOLDS=(0.75 0.8 0.85 0.9)

for DISTANCE_THRESHOLD in "${DISTANCE_THRESHOLDS[@]}"; do
  if [ "$MODEL_TYPE" = "fasttext" ]; then
    python3 augment_ner_data.py --dataset "${DATASET}" --distance_threshold "${DISTANCE_THRESHOLD}"
  else
    python3 augment_ner_data.py --dataset "${DATASET}" --distance_threshold "${DISTANCE_THRESHOLD}" --word2vec
  fi
done
