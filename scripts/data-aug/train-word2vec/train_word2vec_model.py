import os
import sys
import gensim
import argparse
import numpy as np
from itertools import chain
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', help='ISO 639-1 code of target language.')
    parser.add_argument('--word_vector_size', type=int, default=100, help='the size of a word vector')
    parser.add_argument('--window_size', type=int, default=5,
                        help='the maximum distance between the current and predicted word within a sentence.')
    parser.add_argument('--vocab_size', type=int, default=10000, help='the maximum vocabulary size')
    parser.add_argument('--num_negative', type=int, default=5,
                        help='the int for negative specifies how many “noise words” should be drawn')
    parser.add_argument('--save_vocab_and_vectors_tsv', type=bool, default=False,
                        help='whether to save the vocab and vector representations in a tsv file')
    args = parser.parse_args()
    return args


def get_min_count(sents, vocab_size):
    fdist = Counter(chain.from_iterable(sents))
    min_count = fdist.most_common(vocab_size)[-1][1]  # the count of the top-kth word
    return min_count


def make_wordvectors(args):
    sents = []
    with open(f'data/{args.lang}/{args.lang}_corpus.txt', 'r', encoding='utf-8') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            words = line.split()
            sents.append(words)

    min_count = get_min_count(sents, args.vocab_size)
    model = gensim.models.Word2Vec(
        sents,
        vector_size=args.word_vector_size,
        min_count=min_count,
        negative=args.num_negative,
        window=args.window_size,
        workers=64,
        sg=1
    )
    print(f'Vocab size: {len(model.wv.index_to_key)}')

    os.makedirs('out', exist_ok=True)
    model.save(f'out/{args.lang}/{args.lang}_word2vec_model')

    if args.save_vocab_and_vectors_tsv:
        with open(f'data/{args.lang}/{args.lang}_vocab_and_vectors.tsv', 'w', encoding='utf-8') as fout:
            for i, word in enumerate(model.wv.index_to_key):
                vector_str = np.array2string(model.wv[word], separator=',', max_line_width=sys.maxsize)
                fout.write(f"{i}\t{word}\t{vector_str}\n")


if __name__ == "__main__":
    args = parse_args()
    make_wordvectors(args)
