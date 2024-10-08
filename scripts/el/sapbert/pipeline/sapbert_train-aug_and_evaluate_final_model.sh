#!/bin/bash
# This script trains a model with the SapBERT pipeline for the SympTEMIST EL subtasks, but with augmented files. It then
# evaluates it on the official evaluation library. It is easily extensible with other datasets, just parse the dataset
# in the same way as SympTEMIST (with the same file structures and names), and then pass the dataset name as input.
# Likewise, to test other models, add their Hugging Face IDs to the MODEL_IDS variable.

DATASET=$1
LANG=$2
MODEL_TYPE=$3


# Verify inputs
SYMPTEMIST_LANGS="es en fr it pt"
if [ "$DATASET" = "symptemist" ]; then
  if ! echo "$SYMPTEMIST_LANGS" | grep -qw "$LANG"; then
    echo "[ERROR] $(readlink -f "$0"): $LANG language option not supported for SympTEMIST."
    exit 1
  fi
else
  echo "[ERROR] $(readlink -f "$0"): $DATASET dataset not supported or wrongly spelled."
  exit 1
fi

MODEL_TYPES="word2vec fasttext"
if ! echo "$MODEL_TYPES" | grep -qw "$MODEL_TYPE"; then
  echo "[ERROR] $(readlink -f "$0"): $MODEL_TYPE not supported or wrongly spelled."
  exit 1
fi


# Define hyperparameters and models
DISTANCE_THRESHOLDS=(0.75 0.8 0.85 0.9)
case $LANG in
  "es")
    MODEL_IDS=("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR" "PlanTL-GOB-ES/bsc-bio-ehr-es")
    # HYPERPARAMETERS=("3 64 1e-5" "20 128 1e-4")
    HYPERPARAMETERS=("2 64 1e-5" "2 128 1e-4")
    ;;
  "en")
    MODEL_IDS=("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR" "michiyasunaga/BioLinkBERT-base" "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" "emilyalsentzer/Bio_ClinicalBERT")
    # HYPERPARAMETERS=("2 64 2e-5" "17 128 2e-5" "15 256 5e-5" "10 64 2e-5")
    HYPERPARAMETERS=("2 64 2e-5" "2 128 2e-5" "2 256 5e-5" "2 64 2e-5")
    ;;
  "fr")
    MODEL_IDS=("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR" "almanach/camembert-bio-base" "quinten-datalab/AliBERT-7GB" "Dr-BERT/DrBERT-7GB")
    # HYPERPARAMETERS=("3 64 5e-5" "20 64 1e-5" "2 64 1e-4" "2 64 1e-5")
    HYPERPARAMETERS=("2 64 5e-5" "2 64 1e-5" "2 64 1e-4" "2 64 1e-5")
    ;;
  "it")
    MODEL_IDS=("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR" "IVN-RIN/bioBIT" "IVN-RIN/medBIT" "IVN-RIN/MedPsyNIT")
    # HYPERPARAMETERS=("5 64 5e-5" "5 64 5e-5" "17 256 1e-4" "14 64 1e-4")
    HYPERPARAMETERS=("2 64 5e-5" "2 64 5e-5" "2 256 1e-4" "2 64 1e-4")
    ;;
  "pt")
    MODEL_IDS=("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR" "neuralmind/bert-base-portuguese-cased" "PORTULAN/albertina-100m-portuguese-ptpt-encoder" "pucpr/biobertpt-all" "pucpr/biobertpt-clin")
    # HYPERPARAMETERS=("1 128 1e-4" "3 64 5e-5" "14 64 2e-5" "8 64 5e-5" "13 256 2e-5")
    HYPERPARAMETERS=("2 128 1e-4" "2 64 5e-5" "2 64 2e-5" "2 64 5e-5" "2 256 2e-5")
    ;;
esac


# Train and eval loop
for DISTANCE_THRESHOLD in "${DISTANCE_THRESHOLDS[@]}"; do
  for IDX in "${!MODEL_IDS[@]}"; do
    # Parse model id and hyperparameters
    MODEL_ID="${MODEL_IDS[IDX]}"
    HYPERPARAMETERS_FOR_MODEL_STRING="${HYPERPARAMETERS[IDX]}"
    read -ra HYPERPARAMETERS_FOR_MODEL <<< "$HYPERPARAMETERS_FOR_MODEL_STRING"
    EPOCHS="${HYPERPARAMETERS_FOR_MODEL[0]}"
    BATCH_SIZE="${HYPERPARAMETERS_FOR_MODEL[1]}"
    LEARNING_RATE="${HYPERPARAMETERS_FOR_MODEL[2]}"

    echo ""
    echo ">>> [1] TRAINING USING THE SAPBERT PIPELINE"
    echo ">>> [1]   MODEL: ${MODEL_ID}"
    echo ">>> [1]   LANGUAGE: ${LANG}"
    echo ">>> [1]   DATASET: AUGMENTED ${DATASET}"
    echo ">>> [1]   DISTANCE THRESHOLD: ${DISTANCE_THRESHOLD}"
    echo ">>> [1]   BATCH SIZE: ${BATCH_SIZE}"
    echo ">>> [1]   LEARNING RATE: ${LEARNING_RATE}"
    echo ">>> [1]   EPOCHS: ${EPOCHS}"
    echo ""

    # Train model
    cd ../../../../models/sapbert/train/ && CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
      --model_dir "$MODEL_ID" \
      --training_file_path "../../../scripts/data-aug/augment/el/sapbert/out/${MODEL_TYPE}/${LANG}/sapbert_${DATASET}_${LANG}_training_file_aug_3_${DISTANCE_THRESHOLD}.txt" \
      --output_dir_for_best_model "../../../scripts/el/sapbert/pipeline/out/${DATASET}/final-model-aug/${MODEL_TYPE}/${LANG}/${MODEL_ID#*/}_EPOCH_${BATCH_SIZE}_${LEARNING_RATE}_${DISTANCE_THRESHOLD}/" \
      --results_file_path "../../../scripts/el/sapbert/pipeline/out/${DATASET}/final-model-aug/${MODEL_TYPE}/${LANG}/${MODEL_ID#*/}_training_results.json" \
      --epoch "$EPOCHS" \
      --train_batch_size "$BATCH_SIZE" \
      --learning_rate "$LEARNING_RATE" \
      --distance_threshold "$DISTANCE_THRESHOLD" \
      --random_seed 33 \
      --checkpoint_step 999999 \
      --amp \
      --use_cuda \
      --parallel \
      --pairwise \
      --use_miner

    # Upload the model to Hugging Face
    cd ../../../scripts/utils/ && python3 upload_model_to_huggingface.py \
      --local_model_dir "../pipeline/out/${DATASET}/final-model-aug/${MODEL_TYPE}/${LANG}/${MODEL_ID#*/}_${EPOCHS}_${BATCH_SIZE}_${LEARNING_RATE}_${DISTANCE_THRESHOLD}" \
      --augmented_dataset "$MODEL_TYPE"


    echo ""
    echo ">>> [2] EVALUATING USING THE OFFICIAL EVALUATION LIBRARY"
    echo ">>> [2]   MODEL: ${MODEL_ID}"
    echo ">>> [2]   LANGUAGE: ${LANG}"
    echo ">>> [2]   DATASET: AUGMENTED ${DATASET}"
    echo ""

    echo ">>> [2.1] OBTAINING PREDICTIONS"
    echo ""

    # Get the full Hugging Face repository name
    HF_USERNAME=$(grep 'hf_username' config | cut -d'=' -f2)
    HF_REPO_NAME="${HF_USERNAME}/${MODEL_ID#*/}-${DATASET}-${LANG}-${MODEL_TYPE}-${DISTANCE_THRESHOLD}-el"

    # Obtain the predictions
    cd ../../models/sapbert/evaluation/ && CUDA_VISIBLE_DEVICES=0,1 python3 evaluate.py \
      --model_dir "${HF_REPO_NAME}" \
      --test_file_path "../../../scripts/el/sapbert/${DATASET}-parse/out/final-model/${LANG}/sapbert_${DATASET}_${LANG}_test_file.txt" \
      --complete_dictionary_path "../../../scripts/el/sapbert/${DATASET}-parse/out/final-model/${LANG}/sapbert_${DATASET}_${LANG}_dictionary_file_complete.txt" \
      --only_test_codes_dictionary_path "../../../scripts/el/sapbert/${DATASET}-parse/out/final-model/${LANG}/sapbert_${DATASET}_${LANG}_dictionary_file_only_test_codes.txt" \
      --output_file_path "../../../scripts/el/sapbert/pipeline/out/${DATASET}/final-model-aug/${MODEL_TYPE}/${LANG}/${MODEL_ID#*/}_${EPOCHS}_${BATCH_SIZE}_${LEARNING_RATE}_${DISTANCE_THRESHOLD}_final_results.json" \
      --predictions_tsv_file_path "../../../scripts/el/sapbert/pipeline/out/${DATASET}/final-model-aug/${MODEL_TYPE}/${LANG}/${MODEL_ID#*/}_${EPOCHS}_${BATCH_SIZE}_${LEARNING_RATE}_${DISTANCE_THRESHOLD}_predictions.tsv"

    echo ""
    echo ">>> [2.2] EVALUATING MODEL"
    echo ""

    # Evaluate the model on the official evaluation library
    cd "../../../eval-libs/el/" && python3 "evaluate.py" \
      -r "../../../eval-libs/el/test-file-reference-tsvs/${LANG}_test_file_reference.tsv" \
      -p "../../../scripts/el/sapbert/pipeline/out/${DATASET}/final-model-aug/${MODEL_TYPE}/${LANG}/${MODEL_ID#*/}_${EPOCHS}_${BATCH_SIZE}_${LEARNING_RATE}_${DISTANCE_THRESHOLD}_predictions.tsv" \
      -o "../../../scripts/el/sapbert/pipeline/out/${DATASET}/final-model-aug/${MODEL_TYPE}/${LANG}/${MODEL_ID#*/}_${EPOCHS}_${BATCH_SIZE}_${LEARNING_RATE}_${DISTANCE_THRESHOLD}_final_results.json"

  done
done
