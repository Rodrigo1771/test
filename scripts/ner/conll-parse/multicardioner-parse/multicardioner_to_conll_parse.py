import json
import os
import re
import random
random.seed(0)

tracks = [1, 2]
modes = ['train', 'dev', 'test']
args = {
    1: {
        'BIO': {"beginning": 'B-ENFERMEDAD', "inside": 'I-ENFERMEDAD', "outside": 'O'},
        'train_data_dirs': [f"../../../../datasets/multicardioner/track1/distemist_train/brat"],
        'dev_data_dirs': [f"../../../../datasets/multicardioner/track1/cardioccc_dev/brat"],
        'test_data_dirs': [f"../../../../datasets/multicardioner/track1/cardioccc_test/brat"],
    },
    2: {
        'BIO': {"beginning": 'B-FARMACO', "inside": 'I-FARMACO', "outside": 'O'},
        'train_data_dirs': [
            f"../../../../datasets/multicardioner/track2/drugtemist_train/en/brat",
            f"../../../../datasets/multicardioner/track2/drugtemist_train/es/brat",
            f"../../../../datasets/multicardioner/track2/drugtemist_train/it/brat"
        ],
        'dev_data_dirs': [
            f"../../../../datasets/multicardioner/track2/cardioccc_dev/en/brat",
            f"../../../../datasets/multicardioner/track2/cardioccc_dev/es/brat",
            f"../../../../datasets/multicardioner/track2/cardioccc_dev/it/brat"
        ],
        'test_data_dirs': [
            f"../../../../datasets/multicardioner/track2/cardioccc_test/en/brat",
            f"../../../../datasets/multicardioner/track2/cardioccc_test/es/brat",
            f"../../../../datasets/multicardioner/track2/cardioccc_test/it/brat"
        ],
    }
}


# For a dict in which each value is a list of lists, turn them into lists of tuples
def turn_inner_lists_into_tuples(dict):
    for _, list_of_lists in dict.items():
        for i, list in enumerate(list_of_lists):
            list_of_lists[i] = tuple(list)


# Get formatted dataset name from dir path.
def get_set_name(path):
    path_parts = path.split('/')
    track = path_parts[6]
    split = f"{path_parts[7].split('_')[-1]}"
    (dataset, subset) = ('distemist', '') if track == 'track1' else ('drugtemist', '_' + path_parts[8])
    return f'{dataset}{subset}_{split}'


# Parse a directory of .ann and .txt files into a list of tuples of the following format:
#       (token, filename, span, label)
# This format resembles the CONLL file format. Also, the hard coded cases are specific to MultiCardioNER
def parse_brat_dir(mode, data_dir, files, labels):
    lang = data_dir.split('/')[8]
    n_err = 0
    ret = []
    for i in range(0, len(files), 2):
        # open ann and txt file together
        ann_file = files[i]
        txt_file = files[i+1]

        # obtain phrases
        with open(data_dir + '/' + txt_file, 'r') as f:
            txt_lines = f.readlines()
            splits = [line.split('. ') for line in txt_lines]
            phrases = [item for sublist in splits for item in sublist]
            for j, phrase in enumerate(phrases):
                if phrase and phrase[-1] not in ['.', '\n'] or len(phrase) > 1 and phrase[-2:] == '..':
                    phrases[j] = phrase + '.'

        # solve weird bug where lines end with '. \n' instead of '.\n' and it messes things up
        for line in txt_lines:
            if line != '\n' and line != '\u00A0\n' and line != '.\n' and line[-3:] == '. \n':
                ending = line[-30:-3].lstrip()
                aux = ending.split('.')
                if len(aux) > 1:
                    ending = aux[-1].lstrip()
                ending += '.'
                for a, phrase in enumerate(phrases):
                    if mode == 'dev' and txt_file == 'casos_clinicos_cardiologia77.txt' and a == 72:
                        continue
                    if ending in phrase:
                        index = phrases[a].rfind('.')
                        if index != -1:
                            phrases[a] = phrases[a][:index] + ' .' + phrases[a][index+1:]
                        else:
                            print("ERROOOOOO")
                        break

        # solve weird bug where lines have '..' instead of '.' and it messes things up
        for line in txt_lines:
            if '..' in line and '...' not in line:
                index = line.find('..')
                ending = line[index-20:index]
                for a, phrase in enumerate(phrases):
                    if ending in phrase and phrase[-1] != '\n':
                        index = phrases[a].rfind('.')
                        if index != -1:
                            phrases[a] = phrases[a][:index] + ' .' + phrases[a][index+1:]
                        else:
                            print("ERROOOOOO")
                        break

        # obtain annotations and spans
        with open(data_dir + '/' + ann_file, 'r') as f:
            ann_lines = f.readlines()
            annotations = {}
            for line in ann_lines:
                if line[0] == 'T':
                    _, span, text = line.split('\t')
                    _, start, end = span.split(' ')
                    annotations[start] = (text.strip(), end)

        # split phrases into words
        words = [re.split(r'(\s*\S+)', phrase) for phrase in phrases]
        words = [item for sublist in words for item in sublist if item not in [' ', '']]

        # separate words and symbols
        new_words = []
        for word in words:
            if word[-1] in [',', ';', ':', '.', '/', '\'', '?']:
                new_words.append(word[:-1])
                new_words.append(word[-1])
            elif word[0:2] in [' \'', ' «', ' \"', ' ¿', ' “', ' –']:
                new_words.append(word[0:2])
                new_words.append(word[2:])
            elif word[0] == '–':
                new_words.append(word[0])
                new_words.append(word[1:])
            else:
                new_words.append(word)
        words = new_words

        # useful for other processing steps down the line (the spaces step)
        for k, word in enumerate(words):
            if k == 0:
                continue
            if word != '\n' and words[k-1] == '.':
                words[k] = ' ' + word

        # further separate word from symbols
        new_words = []
        for word in words:
            if '(' in word or ')' in word or '[' in word or ']' in word:
                sp = re.findall(r'\w+|[^\w\s]', word)
                c = 0
                for s in sp:
                    if c == 0 and word[0] == ' ':
                        for w in word:
                            if w == ' ':
                                c += 1
                            else:
                                break
                        for q in range(c):
                            s = ' ' + s
                    new_words.append(s)
            else:
                new_words.append(word)
        words = new_words
        new_words = []
        for word in words:
            if len(word) > 2 and '-' in word:
                sp = word.split('-')
                for ind, s in enumerate(sp):
                    if s != '':
                        new_words.append(s)
                    if ind < len(sp) - 1:
                        new_words.append('-')
            elif '+' in word:
                sp = word.split('+')
                for ind, s in enumerate(sp):
                    if s != '':
                        new_words.append(s)
                    if ind < len(sp) - 1:
                        new_words.append('+')
            else:
                new_words.append(word)
        words = new_words
        new_words = []
        for word in words:
            if '/' in word:
                sp = word.split('/')
                for ind, s in enumerate(sp):
                    if s != '':
                        new_words.append(s)
                    if ind < len(sp) - 1:
                        new_words.append('/')
            else:
                new_words.append(word)
        words = new_words
        if lang == 'it':
            new_words = []
            for word in words:
                if '\'' in word:
                    sp = word.split('\'')
                    for ind, s in enumerate(sp):
                        if s != '':
                            new_words.append(s)
                        if ind < len(sp) - 1:
                            new_words.append('\'')
                else:
                    new_words.append(word)
            words = new_words

        # solve specific edge cases without a pattern that have to be hardcoded, namely when two
        # words have no space char between them and the second word is the beginning of an entity
        new_words = []
        for word in words:
            if re.search(r't.*?(nomo|n0m0|n2cmx|n2bm0)', word, re.IGNORECASE):
                new_words.append(word[:-2])
                new_words.append(word[-2:])
            elif txt_file == 'es-S0212-71992000001200006-1.txt' and word == ' positivo.VIH':
                new_words.append(' positivo')
                new_words.append('.')
                new_words.append('VIH')
            elif txt_file == 'es-S0210-48062004000200011-1.txt' and word == ' nohematuria':
                new_words.append(' no')
                new_words.append('hematuria')
            elif mode == 'dev' and txt_file == 'casos_clinicos_cardiologia207.txt' and word == ' conshock':
                new_words.append(' con')
                new_words.append('shock')
            elif mode == 'dev' and txt_file == 'casos_clinicos_cardiologia230.txt' and word == ' pseudonormalA.I':
                new_words.append(' pseudonormal')
                new_words.append('A.I')
            elif mode == 'dev' and txt_file == 'casos_clinicos_cardiologia234.txt' and word == ' deshock':
                new_words.append(' de')
                new_words.append('shock')
            elif mode == 'dev' and txt_file == 'casos_clinicos_cardiologia242.txt' and word == ' presentashock':
                new_words.append(' presenta')
                new_words.append('shock')
            elif mode == 'dev' and txt_file == 'casos_clinicos_cardiologia247.txt' and word == ' mínimoshunt':
                new_words.append(' mínimo')
                new_words.append('shunt')
            elif mode == 'dev' and txt_file == 'casos_clinicos_cardiologia27.txt' and word == ' sigue:"Válvulas':
                new_words.append(' sigue')
                new_words.append(':')
                new_words.append('"')
                new_words.append('Válvulas')
            elif mode == 'dev' and txt_file == 'casos_clinicos_cardiologia293.txt' and word == ' deflutter':
                new_words.append(' de')
                new_words.append('flutter')
            elif mode == 'dev' and txt_file == 'casos_clinicos_cardiologia436.txt' and word == ' 20%.Trombo':
                new_words.append(' 20%')
                new_words.append('.')
                new_words.append('Trombo')
            elif mode == 'train' and txt_file == 'es-S0004-06142010000500014-1.txt' and word == ' Ampicilina,Amoxicilina':
                new_words.append(' Ampicilina')
                new_words.append(',')
                new_words.append('Amoxicilina')
            elif mode == 'train' and txt_file == 'es-S0004-06142010000500014-1.txt' and word == ' Eritromicina,Tetraciclina':
                new_words.append(' Eritromicina')
                new_words.append(',')
                new_words.append('Tetraciclina')
            elif mode == 'train' and txt_file == 'es-S0213-12852006000600002-1.txt' and word == ' ·3TC':
                new_words.append(' ·')
                new_words.append('3TC')
            elif mode == 'train' and txt_file == 'es-S0376-78922012000200008-1.txt' and word == ' adriamicinaciclofosfamida':
                new_words.append(' adriamicina')
                new_words.append('ciclofosfamida')
            else:
                new_words.append(word)
        words = new_words

        # assign spans to all words
        words_and_spans = []
        start = 0
        for word in words:
            end = len(word)
            words_and_spans.append((word, start, start+end))
            start += end

        # take out spaces in the beginning of words
        for n, (word, start, end) in enumerate(words_and_spans):
            if word and word[0] == ' ':
                n_spaces = 0
                for w in word:
                    if w == ' ':
                        n_spaces += 1
                words_and_spans[n] = (word[n_spaces:], start+n_spaces, end)

        # delete artifact empty strings
        for o, (word, _, _) in enumerate(words_and_spans):
            if word == '':
                del words_and_spans[o]

        # build final structure (word, filename, span, tag)
        end2 = -1
        annotation_check = []
        for (word, start, end) in words_and_spans:
            if word == '\n':
                continue
            # check for invisible characters (omg...)
            if '\u00A0' in word:
                continue
            if annotations.get(f'{start}'):
                (_, end2) = annotations.get(f'{start}')
                annotation_check.append(start)
                ret.append((word, txt_file[:-4], f'{start}_{end}', labels['beginning']))
            elif end <= int(end2):
                ret.append((word, txt_file[:-4], f'{start}_{end}', labels['inside']))
            else:
                ret.append((word, txt_file[:-4], f'{start}_{end}', labels['outside']))

        # check for errors
        if len(annotation_check) != len(annotations.keys()):
            b = [ann for ann in list(annotations.keys()) if int(ann) not in annotation_check]
            n_err += len(b)
            print(txt_file, b)

    if lang != 'brat':
        print(f"n_err_{data_dir.split('/')[7]}_{lang}: ", n_err)
    else:
        print(f"n_err_{data_dir.split('/')[7]}: ", n_err)
    print("------------------------------------------------")
    return ret


# Parse a CONLL file into a list of tuples of the following format:
#       (token, filename, span, label)
def parse_conll_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        ret = []
        for line in lines:
            if line == '\n':
                continue
            (word, filename, span, label) = line.split('\t')
            ret.append((word, filename, span, label.rstrip('\n')))
        return ret


# Builds the combined dataset's training file tuples. The tuples have the following format:
#       (token, filename, span, label)
# These tuples only come from the training files themselves, and are meant to be combined with the tuples from the dev
# files in order to make an 80/20 split, 80 for the training file and 20 for the validation file.
def build_combined_training_file_tuples(distemist_train, drugtemist_train, symptemist_train, medprocner_train):
    train_file_tuples = []
    for distemist_line, drugtemist_line, symptemist_line, medprocner_line in zip(
            distemist_train, drugtemist_train, symptemist_train, medprocner_train
    ):
        (word, filename, span) = distemist_line[:3]
        drugtemist_label = drugtemist_line[3]
        if drugtemist_label != 'O':
            final_label = drugtemist_label
        else:
            distemist_label = distemist_line[3]
            if distemist_label != 'O':
                final_label = distemist_label
            else:
                symptemist_label = symptemist_line[3]
                medprocner_label = medprocner_line[3]
                if symptemist_label != 'O' and medprocner_label != 'O':
                    final_label = random.choice([symptemist_label, medprocner_label])
                elif symptemist_label == 'O':
                    final_label = medprocner_label
                elif medprocner_label == 'O':
                    final_label = symptemist_label
                else:
                    final_label = 'O'
        train_file_tuples.append((word, filename, span, final_label))
    return train_file_tuples


# Builds the combined dataset's validation file tuples. The tuples have the following format:
#       (token, filename, span, label)
# These tuples only come from the validation files themselves, and are meant to be combined with the tuples from the
# training files in order to make an 80/20 split, 80 for the training file and 20 for the validation file.
def build_combined_validation_file_tuples(distemist_dev, drugtemist_dev):
    dev_final_lines = []
    for distemist_line, drugtemist_line in zip(distemist_dev, drugtemist_dev):
        (word, filename, span) = distemist_line[:3]
        drugtemist_label = drugtemist_line[3]
        if drugtemist_label != 'O':
            final_label = drugtemist_label
        else:
            distemist_label = distemist_line[3]
            if distemist_label != 'O':
                final_label = distemist_label
            else:
                final_label = 'O'
        dev_final_lines.append((word, filename, span, final_label))
    return dev_final_lines


# Builds the final combined dataset's training and validation file tuples, as it divides 80% of exampled for training,
# and the remaining 20% for validation. The tuples have the following format:
#       (token, filename, span, label)
def built_80_20_split_of_combined_dataset(
        combined_training_file_tuples, combined_validation_file_tuples, selected_examples_idxs=None
):
    # Divide list of all tuples into list of examples, themselves lists of tuples
    def get_examples(tuples):
        examples = [[]]
        example_counter = 0
        for t in tuples:
            examples[example_counter].append(t)
            if t[0] == '.':
                examples.append([])
                example_counter += 1
        return examples[:-1]

    # If a list with the selected indexes is available, use it to select 80% of all examples (for the training file).
    # If not, sample those indexes randomly with `random.sample`.
    train_examples, dev_examples = get_examples(combined_training_file_tuples), get_examples(combined_validation_file_tuples)
    all_examples = train_examples + dev_examples
    n_lines = len(all_examples)
    if selected_examples_idxs:
        selected_train_examples_idxs = selected_examples_idxs['selected_train_examples_idxs']
    else:
        n_train_lines = int(n_lines * 0.8)
        selected_train_examples_idxs = random.sample(range(n_lines), n_train_lines)

    # Selects the opposite values of a list, up to a given threshold.
    # For example: with [1, 2, 4] and threshold 7, it returns [0, 3, 5, 6]
    def opposite_values(original_list, threshold):
        opposite = [i for i in range(threshold) if i not in original_list]
        random.shuffle(opposite)
        return opposite

    # If a list with the selected indexes is available, use it to select the remaining 20% of all examples (for the
    # validation file). If not, get those indexes from the sampled training example indexes and the `opposite_values`
    # function.
    if selected_examples_idxs:
        selected_dev_examples_idxs = selected_examples_idxs['selected_dev_examples_idxs']
    else:
        selected_dev_examples_idxs = opposite_values(selected_train_examples_idxs, n_lines)
    selected_train_examples, selected_dev_examples = [], []
    for idx in selected_train_examples_idxs:
        selected_train_examples.append(all_examples[idx])
    for idx in selected_dev_examples_idxs:
        selected_dev_examples.append(all_examples[idx])

    # Just flattens a list of lists into a list
    def flatten(lst):
        return [item for sublist in lst for item in (flatten(sublist) if isinstance(sublist, list) else [sublist])]

    # Flatten lists of examples back into lists of tuples
    selected_train_lines, selected_dev_lines = flatten(selected_train_examples), flatten(selected_dev_examples)

    # Insert '.' when there are some missing in the original files
    new_selected_train_lines = []
    for i, train_line in enumerate(selected_train_lines):
        new_selected_train_lines.append(train_line)
        (word, filename, span, label) = train_line
        if i < len(selected_train_lines)-1:
            if filename != selected_train_lines[i+1][1] and word != '.':
                new_span = f"{int(span.split('_')[1])}_{int(span.split('_')[1])+1}"
                new_selected_train_lines.append(('.', filename, new_span, 'O'))
    selected_train_lines = new_selected_train_lines

    new_selected_dev_lines = []
    for i, dev_line in enumerate(selected_dev_lines):
        (word, filename, span, label) = dev_line
        # Only keep relevant labels for the validation file
        if label in ['B-SINTOMA', 'I-SINTOMA', 'B-PROCEDIMIENTO', 'I-PROCEDIMIENTO']:
            new_selected_dev_lines.append((word, filename, span, 'O'))
        else:
            new_selected_dev_lines.append(dev_line)
        # Insert '.' when there are some missing in the original files
        if i < len(selected_dev_lines) - 1:
            if filename != selected_dev_lines[i + 1][1] and word != '.':
                new_span = f"{int(span.split('_')[1])}_{int(span.split('_')[1]) + 1}"
                new_selected_dev_lines.append(('.', filename, new_span, 'O'))
    selected_dev_lines = new_selected_dev_lines

    return selected_train_lines, selected_dev_lines


# Builds the final distemist and drugtemist's training and validation file tuples, from the combined dataset's ones, by
# simply discarding the labels that are not relevant from those files.
def build_distemist_and_drugtemist_training_and_validation_tuples(
        combined_dataset_tuples, distemist_labels, drugtemist_labels
):
    distemist_tuples, drugtemist_tuples = [], []
    for multi_dataset_tuple in combined_dataset_tuples:
        (word, filename, span, label) = multi_dataset_tuple
        if label in distemist_labels:
            distemist_tuples.append(multi_dataset_tuple)
        else:
            distemist_tuples.append((word, filename, span, 'O'))
        if label in drugtemist_labels:
            drugtemist_tuples.append(multi_dataset_tuple)
        else:
            drugtemist_tuples.append((word, filename, span, 'O'))
    return distemist_tuples, drugtemist_tuples


# Writes the CONLL files
def write_files(datasets_and_line_tuples_processed):
    for dataset, tuples in datasets_and_line_tuples_processed.items():
        with open(f'out/{dataset}.conll', 'w') as f:
            for i, (word, filename, span, final_label) in enumerate(tuples):
                f.write(f'{word}\t{filename}\t{span}\t{final_label}\n')
                if i < len(tuples)-1 and word == '.' and not tuples[i+1][3].startswith('I-'):
                    f.write('\n')


# Some entities get spread across two separate phrases like:
#
# ...
# word1 -- B-
# word2 -- I-
# .
#
# word3 -- I-
# ...
#
# This messes up some entities, and correcting them in each separate file messes up the example count in the files that
# are related to eachother, so this function ensures that those files are completely equal by taking the "combined" file
# as the anchor, and the drugtemist and distemist files as the ones to process.
def write_files_correct_imperfect_entities(tuples_dict, mode):
    combined_tuples = tuples_dict[0]
    indexes_to_skip = []
    with open(f'out/combined_{mode}.conll', 'w') as f:
        for i, (word, filename, span, final_label) in enumerate(combined_tuples):
            f.write(f'{word}\t{filename}\t{span}\t{final_label}\n')
            if i < len(combined_tuples) - 1 and word == '.':
                if not combined_tuples[i + 1][3].startswith('I-'):
                    f.write('\n')
                else:
                    indexes_to_skip.append(i)
    for name, tuples in zip(['distemist', 'drugtemist_es'], tuples_dict[1:]):
        with open(f'out/{name}_{mode}.conll', 'w') as f:
            for i, (word, filename, span, final_label) in enumerate(tuples):
                f.write(f'{word}\t{filename}\t{span}\t{final_label}\n')
                if i < len(tuples) - 1 and word == '.' and i not in indexes_to_skip:
                    f.write('\n')


def main():
    # This dictionary holds every train, dev and test split from every dataset used to compute the final splits
    datasets_and_line_tuples_preprocessed = {}

    # Parse all MultiCardioNER directories into a format similar to CONLL (check 'parse_dir' documentation). If it's
    # the first time, parse it and save it in 'aux/dirs_and_line_tuples.json'. If not, check for the existence of that
    # file. This just speeds things up.
    if os.path.isfile('aux/dirs_and_line_tuples.json'):
        print("Using existing dirs_and_line_tuples.json")
        with open('aux/dirs_and_line_tuples.json', 'r') as f:
            datasets_and_line_tuples_preprocessed = json.load(f)
            turn_inner_lists_into_tuples(datasets_and_line_tuples_preprocessed)
    else:
        print("Parsing MultiCardioNER\n------------------------------------------------")
        for track in tracks:
            BIO = args[track]['BIO']
            for mode in modes:
                data_dirs = args[track][f'{mode}_data_dirs']
                dirs_and_files = {data_dir: sorted(os.listdir(data_dir)) for data_dir in data_dirs}
                for data_dir, files in dirs_and_files.items():
                    datasets_and_line_tuples_preprocessed[get_set_name(data_dir)] = parse_brat_dir(mode, data_dir, files, BIO)
        with open('aux/dirs_and_line_tuples.json', 'w') as f:
            json.dump(datasets_and_line_tuples_preprocessed, f)

    # PARSING OF DISTEMIST AND DRUGTEMIST-ES

    # Parse SympTEMIST and MedProcNER as well
    datasets_and_line_tuples_preprocessed["symptemist_train"] = parse_conll_file("../symptemist-parse/out/train-full.conll")
    datasets_and_line_tuples_preprocessed["medprocner_train"] = parse_conll_file("../medprocner-parse/out/train-full.conll")

    # Parse the training and validation files of the combined dataset, as they are (without the 80/20 split)
    combined_training_file_tuples = build_combined_training_file_tuples(
        datasets_and_line_tuples_preprocessed["distemist_train"], datasets_and_line_tuples_preprocessed["drugtemist_es_train"],
        datasets_and_line_tuples_preprocessed["symptemist_train"], datasets_and_line_tuples_preprocessed["medprocner_train"]
    )
    combined_validation_file_tuples = build_combined_validation_file_tuples(
        datasets_and_line_tuples_preprocessed["distemist_dev"], datasets_and_line_tuples_preprocessed["drugtemist_es_dev"],
    )

    # Parse the training and validation files of the combined dataset, but now with the 80/20 split
    with open('aux/selected_examples_idxs.json', 'r') as f:
        selected_examples_idxs = json.load(f)
        combined_training_file_tuples, combined_validation_file_tuples = built_80_20_split_of_combined_dataset(
            combined_training_file_tuples, combined_validation_file_tuples, selected_examples_idxs
        )

    # Parse the training and validation files of distemist and drugtemist, from the combined dataset files
    (distemist_train_tuples, drugtemist_train_tuples), (distemist_dev_tuples, drugtemist_dev_tuples) = (
        build_distemist_and_drugtemist_training_and_validation_tuples(
            combined_training_file_tuples, list(args[1]['BIO'].values())[:-1], list(args[2]['BIO'].values())[:-1]
        ), build_distemist_and_drugtemist_training_and_validation_tuples(
            combined_validation_file_tuples, list(args[1]['BIO'].values())[:-1], list(args[2]['BIO'].values())[:-1]
        )
    )

    # PARSING OF DRUGTEMIST-EN AND DRUGTEMIST-IT

    drugtemist_train_en, drugtemist_dev_en = built_80_20_split_of_combined_dataset(
        datasets_and_line_tuples_preprocessed['drugtemist_en_train'],
        datasets_and_line_tuples_preprocessed['drugtemist_en_dev'],
    )

    drugtemist_train_it, drugtemist_dev_it = built_80_20_split_of_combined_dataset(
        datasets_and_line_tuples_preprocessed['drugtemist_it_train'],
        datasets_and_line_tuples_preprocessed['drugtemist_it_dev'],
    )

    # This dictionary holds the final splits
    training_tuples = [combined_training_file_tuples, distemist_train_tuples, drugtemist_train_tuples]
    dev_tuples = [combined_validation_file_tuples, distemist_dev_tuples, drugtemist_dev_tuples]
    independent_datasets_and_line_tuples_processed = {
        "distemist_test": datasets_and_line_tuples_preprocessed["distemist_test"],
        "drugtemist_es_test": datasets_and_line_tuples_preprocessed["drugtemist_es_test"],
        "drugtemist_en_train": drugtemist_train_en,
        "drugtemist_en_dev": drugtemist_dev_en,
        "drugtemist_en_test": datasets_and_line_tuples_preprocessed["drugtemist_en_test"],
        "drugtemist_it_train": drugtemist_train_it,
        "drugtemist_it_dev": drugtemist_dev_it,
        "drugtemist_it_test": datasets_and_line_tuples_preprocessed["drugtemist_it_test"],
    }

    # Save into CONLL files
    os.makedirs('out', exist_ok=True)
    write_files_correct_imperfect_entities(training_tuples, 'train')
    write_files_correct_imperfect_entities(dev_tuples, 'dev')
    write_files(independent_datasets_and_line_tuples_processed)


if __name__ == '__main__':
    main()
