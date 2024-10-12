import os
import re
import random

modes = ['train', 'dev', 'test']
BIO = {
    "beginning": 'B-MORFOLOGIA_NEOPLASIA',
    "inside": 'I-MORFOLOGIA_NEOPLASIA',
    "outside": 'O'
}

for mode in modes:
    if mode in ['train', 'dev']:
        data_dirs = [
            '../../../../datasets/cantemist/train-set/cantemist-ner/',
            '../../../../datasets/cantemist/dev-set1/cantemist-ner/',
            '../../../../datasets/cantemist/dev-set2/cantemist-ner/'
        ]
    else:
        data_dirs = ['../../../../datasets/cantemist/test-set/cantemist-ner/']

    files_and_data_dirs = []
    for data_dir in data_dirs:
        for file in sorted(os.listdir(data_dir)):
            if os.path.isfile(data_dir + file):
                files_and_data_dirs.append((file, data_dir))

    if mode in ['train', 'dev']:
        random.seed(0)
        num_file_pairs = len(files_and_data_dirs) // 2
        num_file_pairs_to_select = int(num_file_pairs * 0.8)
        selected_file_pairs = random.sample(range(num_file_pairs), num_file_pairs_to_select)
        selected_files = []
        for ann, txt in zip([files_and_data_dirs[i*2] for i in selected_file_pairs], [files_and_data_dirs[i*2+1] for i in selected_file_pairs]):
            selected_files.append(ann)
            selected_files.append(txt)
        all_files = []
        if mode == 'dev':
            selected_file_names = [selected_file[0] for selected_file in selected_files]
            for file in files_and_data_dirs:
                if file[0] not in selected_file_names:
                    all_files.append(file)
        else:
            all_files = selected_files
    else:
        all_files = files_and_data_dirs

    final_structure = []
    n_err = 0
    for i in range(0, len(all_files), 2):
        # open ann and txt file together
        ann_file = all_files[i]
        txt_file = all_files[i+1]

        # obtain phrases
        with open(txt_file[1] + txt_file[0], 'r') as f:
            txt_lines = f.readlines()
            splits = [line.split('. ') for line in txt_lines]
            phrases = [item for sublist in splits for item in sublist if item != '']
            for j, phrase in enumerate(phrases):
                if phrase[-1] not in ['.', '\n'] or len(phrase) > 1 and phrase[-2:] == '..':
                    phrases[j] = phrase + '.'

        # solve weird bug where lines end with '. \n' instead of '.\n' and it messes things up
        for line in txt_lines:
            if line != '\n' and line != '\u00A0\n' and line != '.\n' and line[-3:] == '. \n':
                ending = line[-30:-3]
                aux = ending.split('.')
                if len(aux) > 1:
                    ending = aux[-1].lstrip()
                ending += '.'
                for a, phrase in enumerate(phrases):
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
        with open(ann_file[1] + ann_file[0], 'r') as f:
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

        # further separate word from symbols, in this case slashes and parenthesis
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

        # solve specific edge cases without a pattern that have to be hardcoded, aka spelling and orthographic errors
        new_words = []
        for word in words:
            if txt_file[0] == 'cc_onco267.txt' and word == ' MIcarcinoma':
                new_words.append(' MI')
                new_words.append('carcinoma')
            elif txt_file[0] == 'cc_onco1181.txt' and word == ' unactumoración':
                new_words.append(' una')
                new_words.append(' tumoración')
            elif txt_file[0] == 'cc_onco174.txt' and word == ' conmetaplasia':
                new_words.append(' con')
                new_words.append('metaplasia')
            elif txt_file[0] == 'cc_onco195.txt' and word == ' hallazgos:tumoración':
                new_words.append(' hallazgos')
                new_words.append(':')
                new_words.append('tumoración')
            elif txt_file[0] == 'cc_onco1397.txt' and word == ' uncarcinoma':
                new_words.append(' un')
                new_words.append('carcinoma')
            elif txt_file[0] == 'cc_onco1427.txt' and word == ' G3pT3N2Mx':
                new_words.append(' G3')
                new_words.append('pT3N2Mx')
            elif txt_file[0] == 'cc_onco1254.txt' and word == ' G2pT2N0M0':
                new_words.append(' G2')
                new_words.append('pT2N0M0')
            elif txt_file[0] == 'cc_onco202.txt' and word == ' restostumorales':
                new_words.append(' restos')
                new_words.append('tumorales')
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
            if annotations.get(f'{start}'):
                (_, end2) = annotations.get(f'{start}')
                annotation_check.append(start)
                final_structure.append((word, txt_file[0][:-4], f'{start}_{end}', BIO['beginning']))
            elif end <= int(end2):
                final_structure.append((word, txt_file[0][:-4], f'{start}_{end}', BIO['inside']))
            else:
                final_structure.append((word, txt_file[0][:-4], f'{start}_{end}', BIO['outside']))

        # check for errors
        if len(annotation_check) != len(annotations.keys()):
            a = sorted(list(annotations.keys()))
            b = [ann for ann in list(annotations.keys()) if int(ann) not in annotation_check]
            n_err += len(b)
            print(txt_file, b)

    print(f'n_err_{mode}: ', n_err)

    # construct lines and check for freaking invisible characters omfg
    final_strings = []
    for (word, filename, span, tag) in final_structure:
        if '\u00A0' in word:
            continue
        final_strings.append(f"{word}\t{filename}\t{span}\t{tag}\n")
        if word == '.':
            final_strings.append('\n')

    os.makedirs('out', exist_ok=True)
    with open(f'out/{mode}.conll', 'w') as f:
        for i, str in enumerate(final_strings):
            if i in [289869, 469376, len(final_strings)-1]:
                continue
            f.write(str)
