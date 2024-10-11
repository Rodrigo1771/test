# This Dockerfile builds an image that is able to train a model with CUDA. It installs the necessary packages,
# dependencies, and GitHub repository, and then runs a script that trains the model, either for hyperparameter
# search, of the actual final model. It is NOT meant to be interactive.

# Use the NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Install the necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir torch tqdm numpy pandas pytorch_metric_learning transformers sentencepiece protobuf

# Get arguments
ARG DATASET
ARG LANG
ARG MODE
ARG MODEL_TYPE

# Set environment variables
ENV DATASET=$DATASET
ENV LANG=$LANG
ENV MODE=$MODE
ENV MODEL_TYPE=$MODEL_TYPE

# Clone GitHub repo
RUN git clone https://github.com/Rodrigo1771/test

# Run either the hyperparameter search or the final training and evaluation script
CMD if [ "$MODE" = "hyperparameter-search" ]; then \
        cd test/scripts/el/sapbert/pipeline && \
        ./sapbert_hyperparameter_search.sh "$DATASET" "$LANG"; \
    elif [ "$MODE" = "final-model" ]; then \
        cd test/scripts/el/sapbert/pipeline && \
        ./sapbert_train_and_evaluate_final_model.sh "$DATASET" "$LANG"; \
    else \
        cd test/scripts/el/sapbert/pipeline && \
        ./sapbert_train-aug_and_evaluate_final_model.sh "$DATASET" "$LANG" "$MODEL_TYPE"; \
    fi
