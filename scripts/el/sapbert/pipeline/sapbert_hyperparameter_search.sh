#!/bin/bash
# This script performs hyperparameter grid-search on a model for the SympTEMIST dataset, with the SapBERT pipeline. It
# is easily extensible with other datasets, just parse the dataset in the way as SympTEMIST (with the same file
# structures and names), and then pass the dataset name as input. Likewise, to test other models, add their Hugging Face
# IDs to the MODEL_IDS variable.

DATASET=$1
LANG=$2


# Verify inputs
SYMPTEMIST_LANGS="es en fr it pt"
if [ "$DATASET" = "SYMPTEMIST" ]; then
  if ! echo "$SYMPTEMIST_LANGS" | grep -qw "$LANG"; then
    echo "[ERROR] $(readlink -f "$0"): $LANG language option not supported for SympTEMIST."
    exit 1
  fi
else
  echo "[ERROR] $(readlink -f "$0"): $DATASET dataset not supported or wrongly spelled."
  exit 1
fi


# Define hyperparameters and models
# EPOCHS=20
# BATCH_SIZES=(64 128 256 512)
# LEARNING_RATES=(1e-4 5e-5 2e-5 1e-5)
EPOCHS=3
BATCH_SIZES=(64 128)
LEARNING_RATES=(1e-4)
case $LANG in
  "es")
    MODEL_IDS=("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR" "PlanTL-GOB-ES/bsc-bio-ehr-es")
    ;;
  "en")
    MODEL_IDS=("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR" "michiyasunaga/BioLinkBERT-base" "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" "emilyalsentzer/Bio_ClinicalBERT")
    ;;
  "fr")
    MODEL_IDS=("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR" "almanach/camembert-bio-base" "quinten-datalab/AliBERT-7GB" "Dr-BERT/DrBERT-7GB")
    ;;
  "it")
    MODEL_IDS=("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR" "IVN-RIN/bioBIT" "IVN-RIN/medBIT" "IVN-RIN/MedPsyNIT")
    ;;
  "pt")
    MODEL_IDS=("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR" "neuralmind/bert-base-portuguese-cased" "PORTULAN/albertina-100m-portuguese-ptpt-encoder" "pucpr/biobertpt-all" "pucpr/biobertpt-clin")
    ;;
esac


# Hyperparameter search
for MODEL_ID in "${MODEL_IDS[@]}"; do

  echo ""
  echo ">>> [1] HYPERPARAMETER SEARCH USING THE SAPBERT PIPELINE"
  echo ">>> [1]   MODEL: ${MODEL_ID}"
  echo ">>> [1]   LANGUAGE: ${LANG}"
  echo ">>> [1]   DATASET: ${DATASET}"
  echo ""

  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
      echo ""
      echo ">>> [1]   BATCH SIZE: ${BATCH_SIZE}"
      echo ">>> [1]   LEARNING RATE: ${LEARNING_RATE}"
      echo ">>> [1]   EPOCHS: ${EPOCHS}"
      echo ""

      # Train model
      cd ../../../../models/sapbert/train/ && CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
        --model_dir "$MODEL_ID" \
        --training_file_path "../../../scripts/el/sapbert/${DATASET}-parse/out/hyperparameter-search/${LANG}/sapbert_${DATASET}_${LANG}_training_file.txt" \
        --validation_file_path "../../../scripts/el/sapbert/${DATASET}-parse/out/hyperparameter-search/${LANG}/sapbert_${DATASET}_${LANG}_validation_file.txt" \
        --complete_dictionary_path "../../../scripts/el/sapbert/${DATASET}-parse/out/hyperparameter-search/${LANG}/sapbert_${DATASET}_${LANG}_dictionary_file_complete.txt" \
        --only_test_codes_dictionary_path "../../../scripts/el/sapbert/${DATASET}-parse/out/hyperparameter-search/${LANG}/sapbert_${DATASET}_${LANG}_dictionary_file_only_test_codes.txt" \
        --results_file_path "../../../scripts/el/sapbert/pipeline/out/${DATASET}/hyperparameter-search/${LANG}/${MODEL_ID#*/}_training_results.json" \
        --epoch "$EPOCHS" \
        --train_batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --random_seed 33 \
        --checkpoint_step 999999 \
        --amp \
        --use_cuda \
        --parallel \
        --pairwise \
        --use_miner
    done
  done

  # Identify and log the best model's hyperparameters in the results file
  cd ../../../scripts/el/sapbert/pipeline && python3 log_best_overall_model.py --dataset "${DATASET}" --language "${LANG}" --model_id "${MODEL_ID#*/}"

done


