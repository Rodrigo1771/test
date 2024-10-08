import os
import json
import random
import pandas as pd


mode = 'hyperparameter-search'  # ['hyperparameter-search', 'final-model']
languages = ['es', 'en', 'it', 'fr', 'pt']
paths_per_language = {
    lang: {
        'training_file_path':
            '../../../../datasets/symptemist/symptemist_train/subtask2-linking/symptemist_tsv_train_subtask2.tsv'
            if lang == 'es' else f'../../../../datasets/symptemist/symptemist_train/subtask3-experimental_multilingual/symptemist_task3_{lang}.tsv',
        'test_file_path': '../../../../datasets/symptemist/symptemist_test/subtask2-linking/symptemist_tsv_test_subtask2.tsv'
            if lang == 'es' else f'../../../../datasets/symptemist/symptemist_test/subtask3-experimental_multilingual/symptemist_task3_{lang}_test.tsv',
        'gazetteer_file_path': '../../../../datasets/symptemist/symptemist_gazetteer/symptemist_gazetter_snomed_ES_v2.tsv'
            if lang == 'es' else None,
        'output_dir': f'out/{mode}/{lang}',
        'gold_standard_file_path': f'../../../../eval-libs/el/test-file-reference-tsvs/symptemist_{lang}_test_file_reference.tsv',
    }
    for lang in languages
}


def build_training_file(training_file_path, output_dir, lang, mode):
    # If mode = hyperparameter-search, split 80% of the training data for the training file, and 20% for the validation
    # file. (this function only builds the training file tho). Else, use all the training data for the training file
    with open(training_file_path, 'r') as f:
        if mode == "hyperparameter-search":
            random.seed(0)
            lines = f.readlines()
            num_lines = len(lines)
            num_lines_to_select = int(num_lines * 0.8)
            selected_lines = random.sample(lines, num_lines_to_select)
        else:
            selected_lines = f.readlines()

    # Organize training file codes and entities into a dict with the following structure:
    # {'code1': ['entity11', 'entity12', etc], 'code2': ['entity21', 'entity22', etc]}
    simple_training_set_parsing = []
    training_codes_and_entities_json = {}
    for line in selected_lines:
        split_line = line.strip().split('\t')
        entity, code = split_line[4].lower(), split_line[5]
        if code == 'code':
            continue
        simple_training_set_parsing.append(f'{code}||{entity}\n')
        if code in training_codes_and_entities_json:
            training_codes_and_entities_json[code].add(entity)
        else:
            training_codes_and_entities_json[code] = {entity}

    # Write training file
    with open(f"{output_dir}/sapbert_symptemist_{lang}_training_file.txt", 'w') as f:
        for code, entities_set in training_codes_and_entities_json.items():
            if code == 'NO_CODE':
                continue
            entities = list(entities_set)
            for i, entity_a in enumerate(entities):
                for entity_b in entities[i + 1:]:
                    f.write(f"{code}||{entity_a}||{entity_b}\n")

    # Write an auxiliary file with the training data in the gazetteer
    # format, that will be appended to the final dictionary file
    with open(f"{output_dir}/aux/symptemist_{lang}_simple_training_set_parsing_{mode}.txt", 'w') as file:
        file.writelines(simple_training_set_parsing)


def build_test_file(test_file_path, output_dir, lang, mode):
    # If mode = hyperparameter-search, split 80% of the training data for the training file, and 20% for the validation
    # file. (this function only builds the validation file tho). Else, use all the test data for the test file
    with open(test_file_path, 'r') as f:
        if mode == "hyperparameter-search":
            random.seed(0)
            lines = f.readlines()
            num_lines = len(lines)
            num_lines_to_select = int(num_lines * 0.8)
            training_lines = random.sample(lines, num_lines_to_select)
            selected_lines = [line for line in lines if line not in training_lines]
        else:
            selected_lines = f.readlines()

    # Parse validation/test file data
    test_file_all_info = []
    test_codes_and_entities = []
    for line in selected_lines:
        filename, label, start_span, end_span, entity, code, _, _, _, _ = line.strip().split('\t')
        if code == 'code':
            continue
        test_codes_and_entities.append((code, entity))
        test_file_all_info.append({
            'filename': filename,
            'label': label,
            'start_span': start_span,
            'end_span': end_span,
            'entity': entity,
            'code': code
        })

    # Write validation/test file
    filename = 'validation' if mode == 'hyperparameter-search' else 'test'
    with open(f"{output_dir}/sapbert_symptemist_{lang}_{filename}_file.txt", 'w') as f:
        for code, entity in test_codes_and_entities:
            f.write(f"{code}||{entity}\n")

    # Write an auxiliary file with detailed info from the test file, that will be used to build the evaluation file
    with open(f"{output_dir}/aux/test_file_all_info_{lang}_{mode}.json", 'w') as f:
        json.dump(test_file_all_info, f, indent=4)

    # Return codes present in the validation / test file. This will be used to build one of the dictionaries
    return set([code for code, _ in test_codes_and_entities])


def __build_dictionary_files_not_spanish(parsed_simple_training_file, output_dir, test_file_codes, lang):
    # Write dictionary files
    with (
        open(f"{output_dir}/sapbert_symptemist_{lang}_dictionary_file_complete.txt", 'w') as dictionary_file_complete,
        open(f"{output_dir}/sapbert_symptemist_{lang}_dictionary_file_only_test_codes.txt", 'w') as dictionary_file_only_test_codes
    ):
        for line in parsed_simple_training_file:
            dictionary_file_complete.write(line)
            code, _ = line.split('||')
            if code in test_file_codes:
                dictionary_file_only_test_codes.write(line)


def __build_dictionary_files_spanish(parsed_simple_training_file, gazetteer_file_path, output_dir, test_file_codes, lang):
    # Parse dictionary data
    dictionary = {}
    with open(gazetteer_file_path, 'r') as gazetteer_file:
        next(gazetteer_file)  # skip the first line
        for line in gazetteer_file:
            code, _, entity, _, _ = line.strip().split('\t')
            if code in dictionary:
                dictionary[code].append(entity)
            else:
                dictionary[code] = [entity]

    # Write dictionary files
    with (
        open(f"{output_dir}/sapbert_symptemist_{lang}_dictionary_file_complete.txt", 'w') as dictionary_file_complete,
        open(f"{output_dir}/sapbert_symptemist_{lang}_dictionary_file_only_test_codes.txt", 'w') as dictionary_file_only_test_codes,
    ):
        for line in parsed_simple_training_file:
            dictionary_file_complete.write(line)
            code, _ = line.split('||')
            if code in test_file_codes:
                dictionary_file_only_test_codes.write(line)
        for code, entities in dictionary.items():
            for entity in entities:
                dictionary_file_complete.write(f"{code}||{entity}\n")
            if code in test_file_codes:
                for entity in entities:
                    dictionary_file_only_test_codes.write(f"{code}||{entity}\n")


def build_dictionary_files(gazetteer_file_path, output_dir, test_file_codes, lang, mode):
    # Check if the auxiliary file with the training data in the
    # gazetteer format exists, as it is essential for the dictionary
    parsed_training_set_file = f"{output_dir}/aux/symptemist_{lang}_simple_training_set_parsing_{mode}.txt"
    if not os.path.exists(parsed_training_set_file):
        print(f"Training set file ({output_dir}/aux/symptemist_{lang}_simple_training_set_parsing_{mode}.txt) does "
              f"not exist. Building the training file will create this auxiliary file. Then, try to build the "
              f"dictionary file again.")
        return
    with open(parsed_training_set_file, 'r') as f:
        parsed_simple_training_file = f.readlines()

    if lang == 'es':
        __build_dictionary_files_spanish(parsed_simple_training_file, gazetteer_file_path, output_dir, test_file_codes, lang)
    else:
        __build_dictionary_files_not_spanish(parsed_simple_training_file, output_dir, test_file_codes, lang)


def build_el_test_file_reference_tsv(paths):
    df = pd.read_csv(paths['test_file_path'], delimiter='\t')
    df = df.iloc[:, :-4]
    df = df.rename(columns={'span_ini': 'start_span'})
    df = df.rename(columns={'span_end': 'end_span'})
    os.makedirs(os.path.dirname(paths['gold_standard_file_path']), exist_ok=True)
    df.to_csv(paths['gold_standard_file_path'], sep='\t', index=False)
    return df


for lang, paths in paths_per_language.items():
    os.makedirs(os.path.join(paths['output_dir'], 'aux'), exist_ok=True)
    build_training_file(paths['training_file_path'], paths['output_dir'], lang, mode)
    if mode == "hyperparameter-search":
        codes = build_test_file(paths['training_file_path'], paths['output_dir'], lang, mode)
    else:
        codes = build_test_file(paths['test_file_path'], paths['output_dir'], lang, mode)
    build_dictionary_files(paths['gazetteer_file_path'], paths['output_dir'], codes, lang, mode)
    build_el_test_file_reference_tsv(paths)
