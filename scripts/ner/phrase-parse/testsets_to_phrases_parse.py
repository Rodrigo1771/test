import os
import re
import json
import glob
import pandas as pd


data_dirs = {
    'cantemist': {
        'brat': '../../../datasets/cantemist/test-set/cantemist-ner',
    },
    'symptemist': {
        'brat': '../../../datasets/symptemist/symptemist_test/subtask1-ner/brat',
        'tsv': '../../../datasets/symptemist/symptemist_test/subtask1-ner/tsv/symptemist_tsv_test_subtask1.tsv'
    },
    'distemist': {
        'brat': '../../../datasets/multicardioner/track1/cardioccc_test/brat',
        'tsv': '../../../datasets/multicardioner/track1/cardioccc_test/tsv/multicardioner_track1_cardioccc_test.tsv'
    },
    'drugtemist-es': {
        'brat': "../../../datasets/multicardioner/track2/cardioccc_test/es/brat",
        'tsv': '../../../datasets/multicardioner/track2/cardioccc_test/es/tsv/multicardioner_track2_cardioccc_test_es.tsv'
    },
    'drugtemist-en': {
        'brat': "../../../datasets/multicardioner/track2/cardioccc_test/en/brat",
        'tsv': '../../../datasets/multicardioner/track2/cardioccc_test/en/tsv/multicardioner_track2_cardioccc_test_en.tsv'
    },
    'drugtemist-it': {
        'brat': "../../../datasets/multicardioner/track2/cardioccc_test/it/brat",
        'tsv': '../../../datasets/multicardioner/track2/cardioccc_test/it/tsv/multicardioner_track2_cardioccc_test_it.tsv'
    },
}


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


def build_cantemist_tsv(brat_path):
    ann_files = glob.glob(os.path.join(brat_path, '*.ann'))
    ann_files = [filename.split('/')[-1] for filename in sorted(ann_files, key=lambda x: x.lower())]

    data = []
    for ann_file in ann_files:
        filename = os.path.basename(ann_file)[:-4]
        with open(os.path.join(brat_path, ann_file), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    mark, label_and_spans, text = parts
                    label, start_span, end_span = label_and_spans.split(' ')
                    data.append([filename, label, start_span, end_span, text])

    df = pd.DataFrame(data, columns=['filename', 'label', 'start_span', 'end_span', 'text'])

    # Extract the numeric part of the filename for sorting, save it as an auxiliary column, and then drop it when sorted
    df['file_num'] = df['filename'].apply(lambda x: int(re.search(r'\d+', x).group()))
    df = df.sort_values(by=['file_num', 'start_span'])
    df = df.drop(columns=['file_num'])

    return df


def build_ner_test_file_reference_tsv(name, paths):
    if name == 'cantemist':
        df = build_cantemist_tsv(paths['brat'])
    else:
        tsv_file_path = paths['tsv']
        df = pd.read_csv(tsv_file_path, delimiter='\t')
        if name == 'symptemist':
            df = df.drop('ann_id', axis=1)

    reference_tsvs_dir_path = '../../../eval-libs/ner/testset-reference-tsvs/'
    os.makedirs(os.path.dirname(reference_tsvs_dir_path), exist_ok=True)
    df.to_csv(os.path.join(reference_tsvs_dir_path, f'{name}_testset_reference.tsv'), sep='\t', index=False)
    return df


def main():
    for name, paths in data_dirs.items():
        if name == 'symptemist':
            files = glob.glob(os.path.join(paths['brat'], '*.ann')) + glob.glob(os.path.join(paths['brat'], '*.txt'))
            files = [filename.split('/')[-1] for filename in sorted(files, key=lambda x: x.lower())]
        else:
            files = [f for f in os.listdir(paths['brat']) if os.path.isfile(os.path.join(paths['brat'], f))]
            files = list(sorted(files, key=lambda x: (int(re.search(r'\d+', x).group()), x.split('.')[-1])))
        tuples = parse_dir(paths['brat'], files)
        os.makedirs('out', exist_ok=True)
        with open(f"out/{name}_testset_phrases.json", 'w') as f:
            json.dump(tuples, f, indent=4)
        build_ner_test_file_reference_tsv(name, paths)


if __name__ == '__main__':
    main()
