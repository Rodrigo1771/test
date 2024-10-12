#!/bin/bash

# This script allocates as many DI Server Cluster nodes as there are datasets. It then passes to each one of those
# nodes the script "container_handler.sh" with the respective attributes. Change the parameters DATASETS, MODELS and
# ALLOCATION_TIME to your liking (MODELS has to have a model for every dataset in DATASETS, even if it repeats models).

HF_USERNAME=$(grep 'hf_username' config | cut -d'=' -f2)
DATASETS=("symptemist" "cantemist" "distemist" "drugtemist-es" "drugtemist-en" "drugtemist-it")
MODELS=("PlanTL-GOB-ES/bsc-bio-ehr-es" "PlanTL-GOB-ES/bsc-bio-ehr-es" "PlanTL-GOB-ES/bsc-bio-ehr-es" "PlanTL-GOB-ES/bsc-bio-ehr-es" "michiyasunaga/BioLinkBERT-base" "IVN-RIN/bioBIT")
ALLOCATION_TIME="24"  # in hours

for i in "${!DATASETS[@]}"; do
  oarsub -l {"network_address='alakazam-0$((i+1))'"},walltime="${ALLOCATION_TIME}":00:00 "./container_handler.sh ${HF_USERNAME}/${DATASETS[$i]} ${MODELS[$i]}"
done

exit
