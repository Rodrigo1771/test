import os
import re
import json
import glob


# Parse a directory of .ann and .txt files into a list of tuples of the following format:
#       (phrase, filename, start_span of the first word)
# So, the structure of a CONLL file
def parse_dir(data_dir, files):
    ret = []
    for i in range(0, len(files), 2):
        txt_file = files[i+1]
        tmp = []

        # Obtain lines
        with open(data_dir + '/' + txt_file, 'r') as f:
            txt_lines = f.readlines()

        # Obtain phrases with the appropriate amount of '.'s in the end
        splits = [re.split(r'(?<=[^\d])\.(?=\d)|(?<=\d)\.(?=[^\d])|(?<=[^\d])\.(?=[^\d])', line) for line in txt_lines]
        phrases = [item for sublist in splits for item in sublist]
        for j, phrase in enumerate(phrases):
            if not phrase:
                if phrases[j-1]:
                    phrases[j-1] = f'{phrases[j-1]}.'
                else:
                    phrases[j-2] = f'{phrases[j-2]}.'
            elif phrase[-1] != '\n':
                phrases[j] = f'{phrase}.'

        # Obtain stars span of each phrase
        current_len = 0
        for phrase in phrases:
            tmp.append((phrase, txt_file[:-4], current_len))
            current_len += len(phrase)

        # Filter new lines, empty strings, and join phrase that doesn't end with '.' with the next
        new_tmp = []
        skip = False
        for k, (phrase, filename, start_span) in enumerate(tmp):
            if skip:
                skip = False
                continue
            elif not phrase or phrase in ['\n', ' \n']:
                continue
            elif phrase[-1] == '\n':
                spaces = ' '
                l = 0
                for l in range(k+1, len(tmp)):
                    if tmp[l][0] != '\n':
                        break
                    spaces += ' '
                new_tmp.append((f'{phrase[:-1]}{spaces}{tmp[l][0]}', filename, start_span))
                skip = True
            elif phrase[-4:] == ' no.' and tmp[k+1][0][0] != '\n':
                new_tmp.append((f'{phrase}{tmp[k+1][0]}', filename, start_span))
                skip = True
            elif phrase[-3:] == ' n.' and tmp[k+1][0][0] != '\n':
                new_tmp.append((f'{phrase}{tmp[k+1][0]}', filename, start_span))
                skip = True
            elif phrase[-4:] == ' St.' and tmp[k+1][0][0] != '\n':
                new_tmp.append((f'{phrase}{tmp[k+1][0]}', filename, start_span))
                skip = True
            else:
                new_tmp.append((phrase, filename, start_span))
        tmp = new_tmp

        # Erase spaces in the beginning of words
        new_tmp = []
        for (phrase, filename, start_span) in tmp:
            if phrase[0] == ' ':
                new_tmp.append((phrase[1:], filename, start_span+1))
            else:
                new_tmp.append((phrase, filename, start_span))

        ret.extend(new_tmp)

    return ret


def main():
    data_dirs = {
        'cantemist': "../../../datasets/cantemist/test-set/cantemist-ner",
        'symptemist': "../../../datasets/symptemist/test/subtask1-ner/brat",
        'distemist': "../../../datasets/multicardioner/track1/cardioccc_test/brat",
        'drugtemist': "../../../datasets/multicardioner/track2/cardioccc_test/es/brat",
        'drugtemist-en': "../../../datasets/multicardioner/track2/cardioccc_test/en/brat",
        'drugtemist-it': "../../../datasets/multicardioner/track2/cardioccc_test/it/brat",
    }
    for name, data_dir in data_dirs.items():
        if name == 'symptemist':
            files = glob.glob(os.path.join(data_dir, '*.ann')) + glob.glob(os.path.join(data_dir, '*.txt'))
            files = [filename.split('/')[-1] for filename in sorted(files, key=lambda x: x.lower())]
        else:
            files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
            files = list(sorted(files, key=lambda x: (int(re.search(r'\d+', x).group()), x.split('.')[-1])))
        tuples = parse_dir(data_dir, files)
        os.makedirs('out', exist_ok=True)
        with open(f"out/{name}_testset_phrases.json", 'w') as f:
            json.dump(tuples, f, indent=4)


if __name__ == '__main__':
    main()
