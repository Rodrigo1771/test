#!/bin/bash

# This script handles the spawn and output retrieval of a Docker container that trains a model.
# It builds and runs a container, and then retrieves both the results file and the trained model.

DATASET=$1
MODEL=$2

# Build and run the container that trains and evaluates the models for NER
docker rm ner-pipeline-container && docker rmi ner-pipeline-image:latest
docker build --no-cache -f Dockerfile -t "ner-pipeline-image" --build-arg DATASET="${DATASET}" --build-arg MODEL="${MODEL}" .
docker run --gpus all --name "ner-pipeline-container" "ner-pipeline-image"

# Source and destination dirs for all results
SOURCE_DIR="/usr/src/app/test/scripts/ner/pipeline/out"
RESULTS_FILE="${MODEL#*/}-${DATASET}_final_results.json"
DEST_DIR="out"
mkdir -p ${DEST_DIR}

# Extract results files from the container
docker start ner-pipeline-container > /dev/null
docker cp "ner-pipeline-container:$SOURCE_DIR/$RESULTS_FILE" "$DEST_DIR"
docker stop ner-pipeline-container > /dev/null
