# This Dockerfile builds an image that is able to train a model with CUDA. It installs the necessary packages,
# dependencies, and GitHub repository, and then runs a script that trains the model for NER. It is NOT meant to
# be interactive.

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
RUN pip3 install --no-cache-dir transformers[torch] datasets evaluate seqeval pandas tqdm

# Get arguments
ARG DATASET
ARG MODEL
ARG MODEL_TYPE

# Set environment variables
ENV DATASET=$DATASET
ENV MODEL=$MODEL
ENV MODEL_TYPE=$MODEL_TYPE

# Clone GitHub repo
RUN git clone https://github.com/Rodrigo1771/test

# Run the training script
CMD if [ "$MODEL_TYPE" = "" ]; then \
        cd test/scripts/ner/pipeline && \
        ./ner_train_and_evaluate.sh "$DATASET" "$MODEL"; \
    else \
        cd test/scripts/ner/pipeline && \
        ./ner_train-aug_and_evaluate.sh "$DATASET" "$MODEL" "$MODEL_TYPE"; \
    fi
