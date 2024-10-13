import os
import random
import gensim
import argparse
import humanize
from tqdm import tqdm


random.seed(0)
data_paths = {
    'multicardioner': [
        {
            'input_file_path': '../../../ner/conll-parse/multicardioner-parse/out/distemist_train.conll',
            'output_file_path': 'out/multicardioner/distemist_train_aug.conll',
            'lang': 'es'
        },
        {
            'input_file_path': '../../../ner/conll-parse/multicardioner-parse/out/drugtemist_es_train.conll',
            'output_file_path': 'out/multicardioner/drugtemist_es_train_aug.conll',
            'lang': 'es'
        },
        {
            'input_file_path': '../../../ner/conll-parse/multicardioner-parse/out/drugtemist_en_train.conll',
            'output_file_path': 'out/multicardioner/drugtemist_en_train_aug.conll',
            'lang': 'en'
        },
        {
            'input_file_path': '../../../ner/conll-parse/multicardioner-parse/out/drugtemist_it_train.conll',
            'output_file_path': 'out/multicardioner/drugtemist_it_train_aug.conll',
            'lang': 'it'
        },
    ],
    'symptemist': [
        {
            'input_file_path': '../../../ner/conll-parse/symptemist-parse/out/train.conll',
            'output_file_path': 'out/symptemist/train_aug.conll',
            'lang': 'es'
        }
    ],
    'cantemist': [
        {
            'input_file_path': '../../../ner/conll-parse/cantemist-parse/out/train.conll',
            'output_file_path': 'out/cantemist/train_aug.conll',
            'lang': 'es'
        }
    ],
}
structure_words = {
    'es': {'y', 'en', 'lo', 'la', 'el', 'que', 'con', 'sin', 'para', 'por', 'un', 'una', 'unos', 'unas', 'de', 'a', 'del', 'al', 'los', 'las', 'se', 'me', 'te', 'le', 'nos', 'os', 'les', 'mi', 'tu', 'su'},
    'en': {'and', 'in', 'it', 'the', 'that', 'with', 'without', 'for', 'by', 'a', 'an', 'some', 'of', 'to', 'the', 'to', 'the', 'the', 'the', 'it', 'me', 'you', 'him', 'us', 'you', 'them', 'my', 'your', 'his'},
    'it': {'e', 'in', 'lo', 'la', 'il', 'che', 'con', 'senza', 'per', 'da', 'un', 'una', 'uni', 'une', 'di', 'a', 'del', 'al', 'i', 'le', 'si', 'mi', 'ti', 'gli', 'noi', 'voi', 'loro', 'mio', 'tuo', 'suo'},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='The name of the dataset to augment.')
    parser.add_argument('--augment_factor', type=int, required=False, default=3,
                        help='How many times to augment each sentence')
    parser.add_argument('--word2vec', action="store_true",
                        help='Use own Word2Vec models instead of the default FastText ones')
    parser.add_argument('--distance_threshold', type=float, required=False, default=0.75,
                        help='The max distance from a query word to a synonym')
    return parser.parse_args()


def load_data(file_path):
    sentences = []
    current_sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                current_sentence.append(line.split('\t'))
            elif current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
    if current_sentence:
        sentences.append(current_sentence)
    return sentences


def save_data(augmented_data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in augmented_data:
            for word_info in sentence:
                f.write('\t'.join(word_info) + '\n')
            f.write('\n')


def get_file_size(file_path):
    stat_info = os.stat(file_path)
    file_size = stat_info.st_size
    file_size_human_readable = humanize.naturalsize(file_size)
    return file_size_human_readable


def fix_spans(data):
    new_data = []
    for _, sentences in data.items():
        for sentence in sentences:
            new_sentence = []
            offset = 0
            for (word, filename, span, tag) in sentence:
                start_span, end_span = map(int, span.split('_'))
                word_len = len(word)
                new_start_span = f'{start_span-offset}'
                offset += end_span - (start_span + word_len)
                new_end_span = f'{end_span-offset}'
                new_span = f'{new_start_span}_{new_end_span}'
                new_sentence.append([word, filename, new_span, tag])
            new_data.append(new_sentence)
    return new_data


def get_similar_words(model, word, distance_threshold, topn=5):
    try:
        similar_words = model.most_similar(word, topn=topn)
        return [w for w, d in similar_words if d > distance_threshold]
    except KeyError:
        return []


def synonym_replacement(model, entity, lang, previously_replaced_word_idxs_and_best_synonym_idx, distance_threshold):
    words = [w for (w, _, _, _) in entity]
    previously_replaced_word_idxs, best_synonym_idx = previously_replaced_word_idxs_and_best_synonym_idx
    all_structured_words = True
    for i, word in enumerate(words):
        if i not in previously_replaced_word_idxs and word not in structure_words[lang]:
            all_structured_words = False
            break
    if all_structured_words:
        return entity, False
    while True:
        replace_idx = random.randint(0, len(words) - 1)
        if replace_idx in previously_replaced_word_idxs:
            continue
        replace_word = words[replace_idx]
        if replace_word not in structure_words[lang]:
            break
    previously_replaced_word_idxs.append(replace_idx)
    similar_words = get_similar_words(model, replace_word, distance_threshold)
    augmented = len(similar_words) >= best_synonym_idx+1
    if augmented:
        words[replace_idx] = similar_words[best_synonym_idx]
    return [[w, f, s, t] for w, (_, f, s, t) in zip(words, entity)], augmented


def augment_sentence(model, sentence, lang, replaced_idxs_dict, distance_threshold):
    augmented_sentence = []
    current_entity = []
    current_entity_number = 0
    augmented_phrase = False
    for word_info in sentence:
        word, filename, span, tag = word_info
        if tag.startswith('B-'):
            current_entity_number += 1
            if current_entity:
                augmented_entity, augmented = synonym_replacement(model, current_entity, lang, replaced_idxs_dict[current_entity_number], distance_threshold)
                if augmented:
                    augmented_phrase = True
                augmented_sentence.extend(augmented_entity if augmented else current_entity)
                current_entity = []
            current_entity.append(word_info)
        elif tag.startswith('I-'):
            if not current_entity:
                current_entity_number += 1
                current_entity.append([word_info[0], word_info[1], word_info[2], f'B-{word_info[3][3:]}'])
            else:
                current_entity.append(word_info)
        else:  # 'O' tag
            if current_entity:
                augmented_entity, augmented = synonym_replacement(model, current_entity, lang, replaced_idxs_dict[current_entity_number], distance_threshold)
                if augmented:
                    augmented_phrase = True
                augmented_sentence.extend(augmented_entity if augmented else current_entity)
                current_entity = []
            augmented_sentence.append(word_info)

    if current_entity:
        augmented_entity, augmented = synonym_replacement(model, current_entity, lang, replaced_idxs_dict[current_entity_number], distance_threshold)
        if augmented:
            augmented_phrase = True
        augmented_sentence.extend(augmented_entity if augmented else current_entity)

    return augmented_sentence, augmented_phrase


def get_entities_from_sentence(sentence):
    entities = []
    curr_entity = []
    curr_entity_idx = -1
    for i, word_info in enumerate(sentence):
        word, filename, span, tag = word_info
        if tag.startswith('B-'):
            curr_entity_idx = i
            if curr_entity:
                entities.append((' '.join(curr_entity), curr_entity_idx))
                curr_entity = []
                curr_entity_idx = -1
            curr_entity.append(word)
        elif tag.startswith('I-'):
            curr_entity.append(word)
        else:  # 'O' tag
            if curr_entity:
                entities.append((' '.join(curr_entity), curr_entity_idx))
                curr_entity = []
                curr_entity_idx = -1
    if curr_entity:
        entities.append((' '.join(curr_entity), curr_entity_idx))
    return entities


def augment_data(model, sentences, lang, dataset, iteration, augment_factor, distance_threshold):
    augmented_data = {}
    for n, sentence in enumerate(tqdm(sentences, desc=f'Augmenting sentences for {dataset} {iteration}')):
        augmented_data[n] = [sentence]
        entities = get_entities_from_sentence(sentence)
        replaced_idxs_dict = {i+1: [[], 0] for i, _ in enumerate(entities)}
        for _ in range(augment_factor):
            augmented_sentence, augmented = augment_sentence(model, sentence, lang, replaced_idxs_dict, distance_threshold)
            if augmented:
                augmented_data[n].append(augmented_sentence)
            for (i, (idxs, best_synonym_idx)), (entity, _) in zip(replaced_idxs_dict.items(), entities):
                if len(idxs) == len(entity.split(' ')):
                    replaced_idxs_dict[i] = [[], best_synonym_idx + 1]
    return augmented_data


def fix_bnfermedad_bug(data):
    for sentence_info in data:
        for word_info in sentence_info:
            if word_info[3] == 'B-NFERMEDAD':
                word_info[3] = 'I-ENFERMEDAD'
    return data


def main():
    args = parse_args()
    paths_list = data_paths[args.dataset]
    augmented_dicts = []
    for i, path_and_lang in enumerate(paths_list):
        lang = path_and_lang['lang']
        model = gensim.models.Word2Vec.load(f'../../train-word2vec/out/{lang}/{lang}_word2vec_model').wv \
            if args.word2vec else gensim.models.KeyedVectors.load_word2vec_format(
            f'../../fasttext-models/cc.{lang}.300.vec.gz', binary=False)

        input_file_path = path_and_lang['input_file_path']
        output_dir = 'out/word2vec' if args.word2vec else 'out/fasttext'
        output_file_path = path_and_lang['output_file_path'].replace('out', output_dir)
        output_file_path = f'{output_file_path[:-6]}_{args.augment_factor}_{args.distance_threshold}{output_file_path[-6:]}'
        log_file_path = f"{output_dir}/{args.dataset}/aux/log.txt"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        original_data = load_data(input_file_path)
        augmented_data = augment_data(model, original_data, lang, args.dataset, i, args.augment_factor, args.distance_threshold)
        augmented_dicts.append(augmented_data)
        augmented_data = fix_spans(augmented_data)
        if args.dataset == 'multicardioner':
            augmented_data = fix_bnfermedad_bug(augmented_data)
        save_data(augmented_data, output_file_path)

        log = f"Language: {lang}\nDistance Threshold: {args.distance_threshold}\nOriginal data size: {get_file_size(input_file_path)}\nAugmented data size: {get_file_size(output_file_path)}\n\n"
        with open(log_file_path, "a+") as f:
            f.write(log)
        print(log)


if __name__ == "__main__":
    main()
