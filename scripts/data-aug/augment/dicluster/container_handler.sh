#!/bin/bash

# This script handles the spawn and output retrieval of a Docker container that augments either a
# NER or an EL dataset. It builds and runs a container, and then retrieves the augmented dataset.

GITHUB_TOKEN=$1
TASK=$1
ARG=$2
MODEL_TYPE=$3

# Build and run the container that augments either a NER or an EL dataset
docker rm augment-data-container && docker rmi augment-data-image:latest
docker build --no-cache -f Dockerfile -t "augment-data-image" --build-arg GITHUB_TOKEN="${GITHUB_TOKEN}" --build-arg TASK="${TASK}" --build-arg ARG="${ARG}" --build-arg MODEL_TYPE="${MODEL_TYPE}" .
docker run --name "augment-data-container" "augment-data-image"

# Source and destination dirs for all results
SOURCE_DIR="/usr/src/app/test/scripts/data-aug/augment/${TASK}/out/${MODEL_TYPE}/${ARG}"
DEST_DIR="out/${TASK}/${MODEL_TYPE}/${ARG}"

# Extract dataset from the container
docker start augment-data-container > /dev/null
docker cp "augment-data-container:$SOURCE_DIR/." "$DEST_DIR"
docker stop augment-data-container > /dev/null
