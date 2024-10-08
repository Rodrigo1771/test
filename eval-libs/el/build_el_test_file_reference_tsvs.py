import os
import pandas as pd

symptemist_languages = ['es', 'en', 'it', 'fr', 'pt']
symptemist_paths_per_language = {
    lang: {
        'test_file_path': '../../../../datasets/symptemist/test/subtask2-linking/symptemist_tsv_test_subtask2.tsv'
        if lang == 'es' else f'../../../../datasets/symptemist/test/subtask3-experimental_multilingual/symptemist_task3_{lang}_test.tsv',
        'output_file_path': f'test-file-reference-tsvs/{lang}_test_file_reference.tsv',
    }
    for lang in symptemist_languages
}


# To support other datasets, create the dictionaries <DATASET>_languages and <DATASET>_paths_per_language, in the same
# way that it is done for SympTEMIST
def build_el_test_file_reference_tsv(dataset, language):
    if dataset not in ['symptemist']:
        print(f"[ERROR] build_el_test_file_reference_tsvs.py: {dataset} dataset not supported or wrongly spelled.")
        return
    paths = symptemist_paths_per_language[language]
    df = pd.read_csv(paths['test_file_path'], delimiter='\t')
    df = df.iloc[:, :-4]
    df = df.rename(columns={'span_ini': 'start_span'})
    df = df.rename(columns={'span_end': 'end_span'})
    os.makedirs(paths['output_file_path'], exist_ok=True)
    df.to_csv(paths['output_file_path'], sep='\t', index=False)
