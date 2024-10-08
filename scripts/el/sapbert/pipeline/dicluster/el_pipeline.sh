#!/bin/bash

# This script handles the spawn and output retrieval of a Docker container that trains a model.
# It builds and runs a container, and then retrieves both the results file and the trained model.

GITHUB_TOKEN=$1
DATASET=$2
LANG=$3
MODE=$4
MODEL_TYPE=$5

# Build and run the container that performs hyperparameter search
docker rm el-pipeline-container && docker rmi el-pipeline-image:latest
docker build --no-cache -f Dockerfile -t "el-pipeline-image" --build-arg GITHUB_TOKEN="${GITHUB_TOKEN}" --build-arg DATASET="${DATASET}" --build-arg LANG="${LANG}" --build-arg MODE="${MODE}" --build-arg MODEL_TYPE="${MODEL_TYPE}" .
docker run --gpus all --name "el-pipeline-container" "el-pipeline-image"

# Source and destination dirs for all results
if [ "$MODE" == "final-model-aug" ]; then
  MODEL_TYPE_MODIFIER="/$MODEL_TYPE"
else
  MODEL_TYPE_MODIFIER=""
fi
SOURCE_DIR="/usr/src/app/dissertation/scripts/el/sapbert/pipeline/out/${DATASET}/${MODE}${MODEL_TYPE_MODIFIER}/${LANG}"
DEST_DIR="out/${DATASET}/${MODE}${MODEL_TYPE_MODIFIER}/${LANG}"
if [ "$MODE" == "hyperparameter-search" ]; then
	RESULTS_FILE="*_training_results.json"
elif [ "$MODE" == "final-model" ]; then
	RESULTS_FILE="*_*_*_*_final_results.json"
else
	RESULTS_FILE="*_*_*_*_*_final_results.json"
fi

# Extract results files from the container
docker start el-pipeline-container > /dev/null
docker exec el-pipeline-container find "${SOURCE_DIR}" -name "${RESULTS_FILE}" -print0 | xargs -0 -I {} docker cp "el-pipeline-container:{}" "${DEST_DIR}"
docker stop el-pipeline-container > /dev/null
