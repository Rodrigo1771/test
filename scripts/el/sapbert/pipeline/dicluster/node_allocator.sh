#!/bin/bash

# This script allocates as many DI Server Cluster nodes as there are languages (5 in the case of symptemist). It then
# passes to each one of those nodes the script "el_pipeline.sh" with the respective attributes. Change the parameters
# DATASET (if more datasets are added), LANGS, MODE, MODEL_TYPE and ALLOCATION_TIME to your liking. The GITHUB_TOKEN
# is read from the "scripts/config" file.

GITHUB_TOKEN=$(grep 'gh_token' ../../../../config | cut -d'=' -f2)
DATASET="symptemist"
LANGS=("es" "en" "fr" "it" "pt")  # {symptemist: [es, en, fr, it, pt]}
MODE="final-model"  # [hyperparameter-search, final-model, final-mode-aug]
MODEL_TYPE="fasttext"  # {hyperparameter-search: [], final-model: [], final-mode-aug: [word2vec, fasttext]} - if mode is not final-mode-aug, this parameter will be ignored in the Dockerfile
ALLOCATION_TIME="6"  # in hours

for i in "${!LANGS[@]}"; do
  oarsub -l {"network_address='alakazam-0$((i+1))'"},walltime="${ALLOCATION_TIME}":00:00 "./el_pipeline.sh ${GITHUB_TOKEN} ${DATASET} ${LANGS[$i]} ${MODE} ${MODEL_TYPE}"
done

exit
