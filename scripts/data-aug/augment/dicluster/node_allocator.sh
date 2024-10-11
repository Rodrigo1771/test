#!/bin/bash

# This script allocates as many DI Server Cluster nodes as languages. It then passes to
# each of them the script "container_handler.sh" with the respective language as attribute.

TASK="ner"  # [ner, el]
ARGS=("cantemist" "symptemist" "multicardioner")  # {ner: [cantemist, symptemist, multicardioner], el: [es, en, fr, it, pt]}
MODEL_TYPE="word2vec"  # [word2vec, fasttext]
ALLOCATION_TIME="24"  # in hours

for i in "${!ARGS[@]}"; do
  oarsub -l {"network_address='alakazam-0$((i+1))'"},walltime="${ALLOCATION_TIME}":00:00 "./container_handler.sh ${TASK} ${ARGS[$i]} ${MODEL_TYPE}"
done

exit
