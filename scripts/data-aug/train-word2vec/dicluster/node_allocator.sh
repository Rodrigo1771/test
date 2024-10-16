#!/bin/bash

# This script allocates as many DI Server Cluster nodes as languages. It then passes to
# each of them the script "container_handler.sh" with the respective language as attribute.

LANGS=("es" "en" "fr" "it" "pt")  # [es, en, fr, it, pt]
ALLOCATION_TIME="24"  # in hours

for i in "${!LANGS[@]}"; do
  oarsub -l {"network_address='alakazam-0$((i+1))'"},walltime="${ALLOCATION_TIME}":00:00 "./container_handler.sh ${LANGS[$i]}"
done

exit
