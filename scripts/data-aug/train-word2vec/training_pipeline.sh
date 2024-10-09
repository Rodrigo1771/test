#!/bin/bash

# For the provided language, this script downloads the 20240801 (latest at time of development) wikipedia
# dump, decompresses it, builds a corpus with its phrases, and finally trains a word2vec model.

LANG=$1  # [es, en, fr, it, pt]
UPPERCASE_LANG=$(echo "$LANG" | tr '[:lower:]' '[:upper:]')
MAX_CORPUS_SIZE=50000000000  # 50gb
WORD_VECTOR_SIZE=300
WINDOW_SIZE=10  # the maximum distance between the current and predicted word within a sentence.
VOCAB_SIZE=500000
NUM_NEGATIVES=5  # the int for negative specifies how many “noise words” should be drawn

echo ">>> [0] TRAINING ${UPPERCASE_LANG} WORD2VEC MODEL"
echo ">>> [0]   MAX_CORPUS_SIZE: ${MAX_CORPUS_SIZE}"
echo ">>> [0]   WORD_VECTOR_SIZE: ${WORD_VECTOR_SIZE}"
echo ">>> [0]   WINDOW_SIZE: ${WINDOW_SIZE}"
echo ">>> [0]   VOCAB_SIZE: ${VOCAB_SIZE}"
echo ">>> [0]   NUM_NEGATIVES: ${NUM_NEGATIVES}"

echo ">>> [1] DOWNLOADING ${UPPERCASE_LANG} WIKI"
mkdir -p "data/${LANG}"
cd "data/${LANG}" || exit
wget "https://dumps.wikimedia.org/${LANG}wiki/20240801/${LANG}wiki-20240801-pages-articles-multistream.xml.bz2"

echo ">>> [2] DECOMPRESSING ${UPPERCASE_LANG} WIKI"
bzip2 -dk < "${LANG}wiki-20240801-pages-articles-multistream.xml.bz2" | pv -ptb > "${LANG}_wikidump.xml"

echo ">>> [3] BUILDING ${UPPERCASE_LANG} CORPUS"
cd ../..
python3 build_corpus.py --lang="${LANG}" --max_corpus_size=${MAX_CORPUS_SIZE}

echo ">>> [4] TRAINING ${UPPERCASE_LANG} MODEL"
python3 train_word2vec_model.py --lang="${LANG}" --word_vector_size=${WORD_VECTOR_SIZE} --window_size=${WINDOW_SIZE} --vocab_size=${VOCAB_SIZE} --num_negative=${NUM_NEGATIVES}
