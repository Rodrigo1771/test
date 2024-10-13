#!/bin/bash

DATASET=$1
MODEL_ID=$2


# Verify inputs (MODEL_ID is not checked because theoretically you could try any model)
DATASETS="symptemist cantemist distemist drugtemist-es drugtemist-en drugtemist-it"
if ! echo "${DATASETS}" | grep -qw "${DATASET}"; then
  echo "[ERROR] $(readlink -f "$0"): ${DATASET} dataset option not supported or wrongly spelled."
  exit 1
fi


# Define hyperparameters
# (the actual batch size is 64 because PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS = TRUE_BATCH_SIZE)
PER_DEVICE_TRAIN_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=5e-5
EPOCHS=10


# Create output directories
mkdir -p "out/${MODEL_ID#*/}/${DATASET}/"
mkdir -p "out/predictions/"


echo ""
echo ">>> [1] TRAINING USING THE NER PIPELINE"
echo ">>> [1]   MODEL: ${MODEL_ID}"
echo ">>> [1]   DATASET: ${DATASET}"
echo ">>> [1]   TRUE BATCH SIZE: $(( PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS ))"
echo ">>> [1]   LEARNING RATE: ${LEARNING_RATE}"
echo ">>> [1]   EPOCHS: ${EPOCHS}"
echo ""

# Train model
HF_USERNAME=$(grep 'hf_username' ../../config | cut -d'=' -f2)
python3 ner_train.py \
  --model_name_or_path "${MODEL_ID}" \
  --dataset_name "${HF_USERNAME}/${DATASET}-ner" \
  --output_dir "out/${MODEL_ID#*/}/${DATASET}/" \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs "${EPOCHS}" \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --disable_tqdm true \
  --seed 42 \
   2>&1 | tee "out/${MODEL_ID#*/}/${DATASET}/train.log"

Upload the model to Hugging Face
cd ../../utils/ && python3 upload_model_to_huggingface.py \
  --local_model_dir "../ner/pipeline/out/${MODEL_ID#*/}/${DATASET}" \
  --task "ner"


echo ""
echo ">>> [2] EVALUATING USING THE OFFICIAL EVALUATION LIBRARY"
echo ">>> [2]   MODEL: ${MODEL_ID}"
echo ">>> [2]   DATASET: ${DATASET}"
echo ""

echo ">>> [2.1] OBTAINING PREDICTIONS"
echo ""

# Obtain the predictions
HF_REPO_NAME="${HF_USERNAME}/${MODEL_ID#*/}-${DATASET}-ner"
cd ../ner/pipeline && python3 ner_predict.py \
  --model_name "${HF_REPO_NAME}" \
  --input_file_path_json "../phrase-parse/out/${DATASET}_testset_phrases.json" \
  --output_file_path "out/${MODEL_ID#*/}-${DATASET}_final_results.json" \
  --predictions_file_path "out/predictions/${MODEL_ID#*/}-${DATASET}_predictions.tsv"

echo ""
echo ">>> [2.2] EVALUATING MODEL"
echo ""

# Evaluate the model on the official evaluation library
cd ../../../eval-libs/ner/ && python3 evaluate.py \
  -r "testset-reference-tsvs/${DATASET}_testset_reference.tsv" \
  -p "../../scripts/ner/pipeline/out/predictions/${MODEL_ID#*/}-${DATASET}_predictions.tsv" \
  -o "../../scripts/ner/pipeline/out/${MODEL_ID#*/}-${DATASET}_final_results.json"
