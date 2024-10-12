#!/bin/bash

DATASET=$1
MODEL_ID=$2

# Verify inputs (MODEL_ID is not checked because theoretically you could try any model)
DATASETS="symptemist cantemist distemist drugtemist-es drugtemist-en drugtemist-it"
if ! echo "$DATASETS" | grep -qw "$DATASET"; then
  echo "[ERROR] $(readlink -f "$0"): $DATASET dataset option not supported or wrongly spelled."
  exit 1
fi

# Define hyperparameters
# (the actual batch size is 64 because PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS = TRUE_BATCH_SIZE)
PER_DEVICE_TRAIN_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=5e-5
EPOCHS=10

echo ""
echo ">>> [1] TRAINING USING THE NER PIPELINE"
echo ">>> [1]   MODEL: ${MODEL_ID}"
echo ">>> [1]   DATASET: ${DATASET}"
echo ">>> [1]   TRUE BATCH SIZE: $(( PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS ))"
echo ">>> [1]   LEARNING RATE: ${LEARNING_RATE}"
echo ">>> [1]   EPOCHS: ${EPOCHS}"
echo ""

# Train model
python3 ner_train.py \
  --model_name_or_path "$MODEL_ID" \
  --dataset_name "$DATASET-ner" \
  --output_dir "/out/${MODEL_ID#*/}/${DATASET}/" 2>&1 | tee "/out/${MODEL_ID#*/}/${DATASET}/train.log" \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$EPOCHS" \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --disable_tqdm \
  --seed 42

# Upload the model to Hugging Face
BEST_MODEL_CHECKPOINT_DIR=$(python3 get_best_checkpoint_dir.py --output_dir "/out/${MODEL_ID#*/}/${DATASET}/")
echo "$BEST_MODEL_CHECKPOINT_DIR"
cd ../../../scripts/utils/ && python3 upload_model_to_huggingface.py \
  --local_model_dir "$BEST_MODEL_CHECKPOINT_DIR"

