#!/bin/bash

# This script allocates as many DI Server Cluster nodes as there are languages (5 in the case of symptemist). It then
# passes to each one of those nodes the script "container_handler.sh" with the respective attributes. Change the parameters
# DATASET (if more datasets are added), LANGS, MODE, MODEL_TYPE and ALLOCATION_TIME to your liking.

DATASET="symptemist"
LANGS=("es" "en" "fr" "it" "pt")  # {symptemist: [es, en, fr, it, pt]}
MODE="hyperparameter-search"  # [hyperparameter-search, final-model, final-mode-aug]
MODEL_TYPE=""  # {hyperparameter-search: [], final-model: [], final-mode-aug: [word2vec, fasttext]} - if mode is not final-mode-aug, this parameter will be ignored in the Dockerfile
ALLOCATION_TIME="24"  # in hours

for i in "${!LANGS[@]}"; do
  oarsub -l {"network_address='alakazam-0$((i+1))'"},walltime="${ALLOCATION_TIME}":00:00 "./container_handler.sh ${DATASET} ${LANGS[$i]} ${MODE} ${MODEL_TYPE}"
done

exit
