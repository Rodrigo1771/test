import os
import re
import random

modes = ['train-full', 'train', 'dev', 'test']
BIO = {
    "beginning": 'B-SINTOMA',
    "inside": 'I-SINTOMA',
    "outside": 'O'
}

for mode in modes:
    if mode == 'train-full':
        data_dir = 'train+test/'
    elif mode == 'test':
        data_dir = "../../../../datasets/symptemist/symptemist_test/subtask1-ner/brat/"
    else:
        data_dir = "../../../../datasets/symptemist/symptemist_train/subtask1-ner/brat/"

    files = sorted(os.listdir(data_dir))

    if mode in ['train', 'dev']:
        random.seed(0)
        num_file_pairs = len(files) // 2
        num_file_pairs_to_select = int(num_file_pairs * 0.8)
        selected_file_pairs = random.sample(range(num_file_pairs), num_file_pairs_to_select)
        selected_files = []
        for ann, txt in zip([files[i*2] for i in selected_file_pairs], [files[i*2+1] for i in selected_file_pairs]):
            selected_files.append(ann)
            selected_files.append(txt)
        files = [file for file in files if file not in selected_files] if mode == 'dev' else selected_files

    final_structure = []
    n_err = 0
    for i in range(0, len(files), 2):
        # open ann and txt file together
        ann_file = files[i]
        txt_file = files[i+1]

        # obtain phrases
        with open(data_dir + txt_file, 'r') as f:
            txt_lines = f.readlines()
            splits = [line.split('. ') for line in txt_lines]
            phrases = [item for sublist in splits for item in sublist]
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
        with open(data_dir + ann_file, 'r') as f:
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
            if txt_file == 'es-S0212-71992000001200006-1.txt' and word == ' positivo.VIH':
                new_words.append(' positivo')
                new_words.append('.')
                new_words.append('VIH')
            elif txt_file == 'es-S0210-48062004000200011-1.txt' and word == ' nohematuria':
                new_words.append(' no')
                new_words.append('hematuria')
            elif re.search(r't.*?(nomo|n0m0|n2cmx|n2bm0)', word, re.IGNORECASE):
                new_words.append(word[:-2])
                new_words.append(word[-2:])
            elif mode in ['train', 'train-full'] and txt_file == 'es-S0004-06142010000500014-1.txt' and word == ' Ampicilina,Amoxicilina':
                new_words.append(' Ampicilina')
                new_words.append(',')
                new_words.append('Amoxicilina')
            elif mode in ['train', 'train-full'] and txt_file == 'es-S0004-06142010000500014-1.txt' and word == ' Eritromicina,Tetraciclina':
                new_words.append(' Eritromicina')
                new_words.append(',')
                new_words.append('Tetraciclina')
            elif mode in ['train', 'train-full'] and txt_file == 'es-S0213-12852006000600002-1.txt' and word == ' ·3TC':
                new_words.append(' ·')
                new_words.append('3TC')
            elif mode in ['train', 'train-full'] and txt_file == 'es-S0376-78922012000200008-1.txt' and word == ' adriamicinaciclofosfamida':
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
            if annotations.get(f'{start}'):
                (_, end2) = annotations.get(f'{start}')
                annotation_check.append(start)
                final_structure.append((word, txt_file[:-4], f'{start}_{end}', BIO['beginning']))
            elif end <= int(end2):
                final_structure.append((word, txt_file[:-4], f'{start}_{end}', BIO['inside']))
            else:
                final_structure.append((word, txt_file[:-4], f'{start}_{end}', BIO['outside']))

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
            if mode == 'train' and i in [
                2827, 7446, 21097, 30508, 30510, 30512, 34006, 34461, 37806, 45358, 46919, 68248, 91476, 91505, 95676,
                134667, 146038, 149435, 150203, 150672, 150680, 150688, 152859, 152864, 161045, 162033, 162038, 180795,
                190364, 190393, 191056, 195573, 197180, 204650, 210981, 214183, 216216, 243057, 253672, 253682,
                len(final_strings)-1
            ] or mode == 'train-full' and i in [
                8252, 12434, 13989, 38653, 38661, 38669, 48336, 49547, 59262, 63292, 63535, 68485, 69896, 80424, 83584,
                86021, 105413, 105418, 107425, 107454, 110399, 117435, 128401, 128411, 131547, 134808, 134813, 138840,
                142128, 146292, 147326, 159773, 165517, 176145, 205734, 229742, 233350, 237260, 242698, 260902, 298405,
                305379, 343385, 345977, 347160, 347162, 347164, 366796, 395299, 406158, 406187, 424138, 7074, 15411, 22385,
                28863, 39445, 39448, 39698, 39701, 39704, 39707, 39916, 71710, 71926, 140773, 175022, 176145, 207022,
                207036, 207418, 207422, 212702, 212714, 212739, 212790, 213492, 217930, 222539, 223873, 230514, 232456,
                248330, 249458, 256874, 264132, 268924, 270182, 299097, 299728, 302507, 325877, 389185, 389604,
                len(final_strings)-1
            ]:
                continue
            f.write(str)
