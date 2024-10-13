#!/bin/bash

# This script handles the spawn and output retrieval of a Docker container that trains a model.
# It builds and runs a container, and then retrieves both the results file and the trained model.

DATASET=$1
MODEL=$2
MODEL_TYPE=$3

# Build and run the container that trains and evaluates the models for NER
docker rm ner-pipeline-container && docker rmi ner-pipeline-image:latest
docker build --no-cache -f Dockerfile -t "ner-pipeline-image" --build-arg DATASET="${DATASET}" --build-arg MODEL="${MODEL}" --build-arg MODEL_TYPE="${MODEL_TYPE}" .
docker run --gpus all --name "ner-pipeline-container" "ner-pipeline-image"

# Source and destination dirs for all results
SOURCE_DIR="/usr/src/app/test/scripts/ner/pipeline/out"
DEST_DIR="out"
mkdir -p ${DEST_DIR}
if [ "$MODEL_TYPE" == "" ]; then
  RESULTS_FILE="${MODEL#*/}-${DATASET}_final_results.json"
else
  RESULTS_FILE="${MODEL#*/}-${DATASET}-${MODEL_TYPE}-*_final_results.json"
fi

# Extract results files from the container
docker start ner-pipeline-container > /dev/null
docker exec ner-pipeline-container find "${SOURCE_DIR}" -name "${RESULTS_FILE}" -print0 | xargs -0 -I {} docker cp "ner-pipeline-container:{}" "${DEST_DIR}"
docker stop ner-pipeline-container > /dev/null
