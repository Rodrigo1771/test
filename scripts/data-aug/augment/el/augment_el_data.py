import os
import copy
import gensim
import random
import argparse
import humanize
from tqdm import tqdm


random.seed(0)
structure_words = {
    'es': {'y', 'en', 'lo', 'la', 'el', 'que', 'con', 'sin', 'para', 'por', 'un', 'una', 'unos', 'unas', 'de', 'a', 'del', 'al', 'los', 'las', 'se', 'me', 'te', 'le', 'nos', 'os', 'les', 'mi', 'tu', 'su'},
    'en': {'and', 'in', 'it', 'the', 'that', 'which' 'with', 'without', 'for', 'by', 'a', 'an', 'some', 'of', 'to', 'the', 'to', 'the', 'the', 'the', 'it', 'me', 'you', 'him', 'us', 'you', 'them', 'my', 'your', 'his'},
    'fr': {'et', 'en', 'le', 'la', 'le', 'que', 'avec', 'sans', 'pour', 'par', 'un', 'une', 'des', 'des', 'de', 'à', 'du', 'au', 'les', 'les', 'se', 'me', 'te', 'lui', 'nous', 'vous', 'leur', 'mon', 'ton', 'son'},
    'it': {'e', 'in', 'lo', 'la', 'il', 'che', 'con', 'senza', 'per', 'da', 'un', 'una', 'uni', 'une', 'di', 'a', 'del', 'al', 'i', 'le', 'si', 'mi', 'ti', 'gli', 'noi', 'voi', 'loro', 'mio', 'tuo', 'suo'},
    'pt': {'e', 'em', 'o', 'a', 'o', 'que', 'com', 'sem', 'para', 'por', 'um', 'uma', 'uns', 'umas', 'de', 'a', 'do', 'ao', 'os', 'as', 'se', 'me', 'te', 'lhe', 'nos', 'vos', 'lhes', 'meu', 'teu', 'seu'}
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', required=True, help='Language of the dataset')
    parser.add_argument('--augment_factor', type=int, required=False, default=3,
                        help='How many times to augment each entity')
    parser.add_argument('--word2vec', action="store_true",
                        help='Use own Word2Vec models instead of the default FastText ones')
    parser.add_argument('--distance_threshold', type=float, required=False, default=0.7,
                        help='The max distance from a query entity to a synonym')
    args = parser.parse_args()
    return args


def load_data(file_path):
    codes_and_entities = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            code, entity1, entity2 = line.strip().split('||')
            if code in codes_and_entities:
                codes_and_entities[code].extend([entity1, entity2])
            else:
                codes_and_entities[code] = [entity1, entity2]
        for code, entities in codes_and_entities.items():
            codes_and_entities[code] = remove_duplicates_and_preserve_order(entities)
    return codes_and_entities


def save_data(augmented_data, new_entities_aux, augmented_data_file_path, new_entities_aux_file_path=None):
    with open(augmented_data_file_path, 'w', encoding='utf-8') as f:
        for code, entities in augmented_data.items():
            for i, entity_a in enumerate(entities):
                for entity_b in entities[i + 1:]:
                    f.write(f"{code}||{entity_a}||{entity_b}\n")
    if new_entities_aux_file_path:
        with open(new_entities_aux_file_path, 'w', encoding='utf-8') as f:
            for line in new_entities_aux:
                f.write(line)


def remove_duplicates_and_preserve_order(lst):
    seen = set()
    result = []
    for elem in lst:
        if elem not in seen:
            seen.add(elem)
            result.append(elem)
    return result


def get_file_size(file_path):
    stat_info = os.stat(file_path)
    file_size = stat_info.st_size
    file_size_human_readable = humanize.naturalsize(file_size)
    return file_size_human_readable


def get_similar_words(model, word, distance_threshold, topn=5):
    try:
        similar_words = model.most_similar(word, topn=topn)
        return [w for w, d in similar_words if d > distance_threshold]
    except KeyError:
        return []


def synonym_replacement(args, model, entity, previously_replaced_word_idxs, distance_threshold):
    words = entity.split()
    all_structured_words = True
    for i, word in enumerate(words):
        if i not in previously_replaced_word_idxs and word not in structure_words[args.lang]:
            all_structured_words = False
            break
    if all_structured_words:
        return []
    new_words = []
    while True:
        replace_idx = random.randint(0, len(words) - 1)
        if replace_idx in previously_replaced_word_idxs:
            continue
        replace_word = words[replace_idx]
        if replace_word not in structure_words[args.lang]:
            break
    previously_replaced_word_idxs.append(replace_idx)
    similar_words = get_similar_words(model, replace_word, distance_threshold)
    for similar_word in similar_words:
        words[replace_idx] = similar_word
        new_words.append(' '.join(words))
    return remove_duplicates_and_preserve_order(new_words)


def augment_data(args, model, data, augment_factor, distance_threshold):
    augmented_data = {}
    new_entities_aux = []
    for code, entities in tqdm(data.items(), desc="Augmenting EL data"):
        augmented_data[code] = copy.deepcopy(entities)
        for entity in entities:
            replaced_idxs = []
            for i in range(augment_factor):
                if i == len(entity.split(' ')):
                    break
                new_entities = synonym_replacement(args, model, entity, replaced_idxs, distance_threshold)
                for new_entity in new_entities:
                    if new_entity in augmented_data[code]:
                        continue
                    augmented_data[code].append(new_entity)
                    new_entities_aux.append(f'{code}: {entity} -> {new_entity}\n')
    return augmented_data, new_entities_aux


def main():
    args = parse_args()
    model = gensim.models.Word2Vec.load(f'../../train-word2vec/out/{args.lang}/{args.lang}_word2vec_model').wv \
        if args.word2vec else gensim.models.KeyedVectors.load_word2vec_format(
        f'../../fasttext-models/cc.{args.lang}.300.vec.gz', binary=False)
    input_file_path = f"../../../el/sapbert/symptemist-parse/out/final-model/{args.lang}/sapbert_symptemist_{args.lang}_training_file.txt"
    output_dir = 'out/word2vec' if args.word2vec else 'out/fasttext'
    output_file_path = f"{output_dir}/{args.lang}/sapbert_symptemist_{args.lang}_training_file_aug_{args.augment_factor}_{args.distance_threshold}.txt"
    aux_file_path = f"{output_dir}/{args.lang}/aux/new_entities_{args.lang}_{args.augment_factor}_{args.distance_threshold}.txt"
    log_file_path = f"{output_dir}/{args.lang}/aux/log.txt"
    os.makedirs(os.path.dirname(aux_file_path), exist_ok=True)

    original_data = load_data(input_file_path)
    augmented_data, new_entities_aux = augment_data(args, model, original_data, args.augment_factor, args.distance_threshold)
    save_data(augmented_data, new_entities_aux, output_file_path, aux_file_path)

    log = f"Language: {args.lang}\nDistance Threshold: {args.distance_threshold}\nOriginal data size: {get_file_size(input_file_path)}\nAugmented data size: {get_file_size(output_file_path)}\n\n"
    with open(log_file_path, "a+") as f:
        f.write(log)
    print(log)


if __name__ == "__main__":
    main()
