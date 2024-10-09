#!/bin/bash

# This script handles the spawn and output retrieval of a Docker container that trains a
# word2vec model. It builds and runs a container, and then retrieves the trained model.

GITHUB_TOKEN=$1
LANG=$2

# Build and run the container that trains the word2vec model
docker rm word2vec-training-pipeline-container && docker rmi word2vec-training-pipeline-image:latest
docker build --no-cache -f Dockerfile -t "word2vec-training-pipeline-image" --build-arg GITHUB_TOKEN="${GITHUB_TOKEN}" --build-arg LANG="${LANG}" .
docker run --name "word2vec-training-pipeline-container" "word2vec-training-pipeline-image"

# Source and destination dirs for all results
SOURCE_DIR="/usr/src/app/test/scripts/data-aug/train-word2vec/out/${LANG}"
DEST_DIR="out/${LANG}"
mkdir -p "${DEST_DIR}"

# Extract model from the container
docker start word2vec-training-pipeline-container > /dev/null
docker cp "word2vec-training-pipeline-container:$SOURCE_DIR/." "$DEST_DIR"
docker stop word2vec-training-pipeline-container > /dev/null
