import utils
import pandas as pd
from argparse import ArgumentParser


def main(argv=None):
    # Parse options
    parser = ArgumentParser()
    parser.add_argument("-r", "--reference", help=".TSV file with Gold Standard or reference annotations", required=True)
    parser.add_argument("-p", "--prediction", help=".TSV file with your predictions", required=True)
    parser.add_argument("-t", "--task",
                        choices=['track1', 'track2_es', 'track2_en', 'track2_it', 'symptemist', 'cantemist'],
                        help="Task that you want to evaluate (track1, track2_es, track2_en or track2_it)",
                        required=True)
    parser.add_argument("-o", "--output_file", help="Path to save the scoring results", required=True)
    args = parser.parse_args(argv)

    # Read gold_standard and predictions
    df_gs = pd.read_csv(args.reference, sep="\t")
    df_preds = pd.read_csv(args.prediction, sep="\t")
    df_preds = df_preds.drop_duplicates(subset=["filename", "label", "start_span", "end_span"]).reset_index(drop=True)

    # Calculate results
    calculate_ner(df_gs, df_preds, args.output_file, args.task)


def calculate_ner(df_gs, df_preds, output_path, task):
    # Group annotations by filename
    list_gs_per_doc = df_gs.groupby('filename').apply(
        lambda x: x[["filename", 'start_span', 'end_span', "text",  "label"]].values.tolist())
    list_preds_per_doc = df_preds.groupby('filename').apply(
        lambda x: x[["filename", 'start_span', 'end_span', "text", "label"]].values.tolist())
    scores = utils.calculate_fscore(list_gs_per_doc, list_preds_per_doc)
    utils.write_results(task, scores, output_path)


if __name__ == "__main__":
    main()

# OLD / LEGACY

# python3 ner_evaluation.py -r ./data/gold_standard/distemist-dev-gold-standard.tsv -p ./data/predictions/devset/bsc-bio-ehr-es-combined-train-distemist-dev-ner.tsv -t track1 -o ./out/devset_results
# python3 ner_evaluation.py -r ./data/gold_standard/distemist-dev-gold-standard.tsv -p ./data/predictions/devset/bsc-bio-ehr-es-distemist-ner.tsv -t track1 -o ./out/devset_results
# python3 ner_evaluation.py -r ./data/gold_standard/drugtemist-dev-gold-standard.tsv -p ./data/predictions/devset/bsc-bio-ehr-es-combined-train-drugtemist-dev-ner.tsv -t track2_es -o ./out/devset_results
# python3 ner_evaluation.py -r ./data/gold_standard/drugtemist-dev-gold-standard.tsv -p ./data/predictions/devset/bsc-bio-ehr-es-drugtemist-ner.tsv -t track2_es -o ./out/devset_results

# python3 ner_evaluation.py -r ../multicardioner/track1/cardioccc_test/tsv/multicardioner_track1_cardioccc_test.tsv -m ../multicardioner/test+background/multicardioner_test+background_fname-mapping.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-combined-train-distemist-dev-ner-documents.tsv -t track1 -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track1/cardioccc_test/tsv/multicardioner_track1_cardioccc_test.tsv -m ../multicardioner/test+background/multicardioner_test+background_fname-mapping.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-distemist-ner-documents.tsv -t track1 -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/es/tsv/multicardioner_track2_cardioccc_test_es.tsv -m ../multicardioner/test+background/multicardioner_test+background_fname-mapping.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-combined-train-drugtemist-dev-ner-documents.tsv -t track2_es -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/es/tsv/multicardioner_track2_cardioccc_test_es.tsv -m ../multicardioner/test+background/multicardioner_test+background_fname-mapping.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-drugtemist-ner-documents.tsv -t track2_es -o ./out/testset_results



# NORMAL DATASETS

# python3 ner_evaluation.py -r ../multicardioner/track1/cardioccc_test/tsv/multicardioner_track1_cardioccc_test.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-combined-train-distemist-dev-ner-phrases.tsv -t track1 -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track1/cardioccc_test/tsv/multicardioner_track1_cardioccc_test.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-distemist-ner-phrases.tsv -t track1 -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/es/tsv/multicardioner_track2_cardioccc_test_es.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-combined-train-drugtemist-dev-ner-phrases.tsv -t track2_es -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/es/tsv/multicardioner_track2_cardioccc_test_es.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-drugtemist-ner-phrases.tsv -t track2_es -o ./out/testset_results

# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/it/tsv/multicardioner_track2_cardioccc_test_it.tsv -p ./data/predictions/testset/bioBIT-drugtemist-it-ner-phrases.tsv -t track2_it -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/en/tsv/multicardioner_track2_cardioccc_test_en.tsv -p ./data/predictions/testset/BioLinkBERT-base-drugtemist-en-ner-phrases.tsv -t track2_en -o ./out/testset_results

# python3 ner_evaluation.py -r ../symptemist/test/subtask1-ner/tsv/symptemist_tsv_test_subtask1.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-symptemist-ner-phrases.tsv -t symptemist -o ./out/testset_results
# python3 ner_evaluation.py -r ../cantemist/test-set/cantemist-ner/tsv/cantemist_ner_test.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-cantemist-ner-phrases.tsv -t cantemist -o ./out/testset_results



# AUGMENTED DATASETS

## Word2Vec

# python3 ner_evaluation.py -r ../symptemist/test/subtask1-ner/tsv/symptemist_tsv_test_subtask1.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-symptemist-word2vec-75-ner-phrases.tsv -t symptemist -o ./out/testset_results
# python3 ner_evaluation.py -r ../symptemist/test/subtask1-ner/tsv/symptemist_tsv_test_subtask1.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-symptemist-word2vec-8-ner-phrases.tsv -t symptemist -o ./out/testset_results
# python3 ner_evaluation.py -r ../symptemist/test/subtask1-ner/tsv/symptemist_tsv_test_subtask1.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-symptemist-word2vec-85-ner-phrases.tsv -t symptemist -o ./out/testset_results
# python3 ner_evaluation.py -r ../symptemist/test/subtask1-ner/tsv/symptemist_tsv_test_subtask1.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-symptemist-word2vec-9-ner-phrases.tsv -t symptemist -o ./out/testset_results

# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/en/tsv/multicardioner_track2_cardioccc_test_en.tsv -p ./data/predictions/testset/BioLinkBERT-base-drugtemist-en-word2vec-75-ner-phrases.tsv -t track2_en -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/en/tsv/multicardioner_track2_cardioccc_test_en.tsv -p ./data/predictions/testset/BioLinkBERT-base-drugtemist-en-word2vec-8-ner-phrases.tsv -t track2_en -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/en/tsv/multicardioner_track2_cardioccc_test_en.tsv -p ./data/predictions/testset/BioLinkBERT-base-drugtemist-en-word2vec-85-ner-phrases.tsv -t track2_en -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/en/tsv/multicardioner_track2_cardioccc_test_en.tsv -p ./data/predictions/testset/BioLinkBERT-base-drugtemist-en-word2vec-9-ner-phrases.tsv -t track2_en -o ./out/testset_results

# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/it/tsv/multicardioner_track2_cardioccc_test_it.tsv -p ./data/predictions/testset/bioBIT-drugtemist-it-word2vec-75-ner-phrases.tsv -t track2_it -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/it/tsv/multicardioner_track2_cardioccc_test_it.tsv -p ./data/predictions/testset/bioBIT-drugtemist-it-word2vec-8-ner-phrases.tsv -t track2_it -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/it/tsv/multicardioner_track2_cardioccc_test_it.tsv -p ./data/predictions/testset/bioBIT-drugtemist-it-word2vec-85-ner-phrases.tsv -t track2_it -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/it/tsv/multicardioner_track2_cardioccc_test_it.tsv -p ./data/predictions/testset/bioBIT-drugtemist-it-word2vec-9-ner-phrases.tsv -t track2_it -o ./out/testset_results

# python3 ner_evaluation.py -r ../cantemist/test-set/cantemist-ner/tsv/cantemist_ner_test.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-cantemist-word2vec-85-ner-phrases.tsv -t cantemist -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track1/cardioccc_test/tsv/multicardioner_track1_cardioccc_test.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-combined-train-distemist-dev-word2vec-85-ner-phrases.tsv -t track1 -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track1/cardioccc_test/tsv/multicardioner_track1_cardioccc_test.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-distemist-word2vec-85-ner-phrases.tsv -t track1 -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/es/tsv/multicardioner_track2_cardioccc_test_es.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-combined-train-drugtemist-dev-word2vec-85-ner-phrases.tsv -t track2_es -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/es/tsv/multicardioner_track2_cardioccc_test_es.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-drugtemist-word2vec-85-ner-phrases.tsv -t track2_es -o ./out/testset_results


## FastText

# python3 ner_evaluation.py -r ../symptemist/test/subtask1-ner/tsv/symptemist_tsv_test_subtask1.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-symptemist-fasttext-75-ner-phrases.tsv -t symptemist -o ./out/testset_results
# python3 ner_evaluation.py -r ../symptemist/test/subtask1-ner/tsv/symptemist_tsv_test_subtask1.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-symptemist-fasttext-8-ner-phrases.tsv -t symptemist -o ./out/testset_results
# python3 ner_evaluation.py -r ../symptemist/test/subtask1-ner/tsv/symptemist_tsv_test_subtask1.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-symptemist-fasttext-85-ner-phrases.tsv -t symptemist -o ./out/testset_results
# python3 ner_evaluation.py -r ../symptemist/test/subtask1-ner/tsv/symptemist_tsv_test_subtask1.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-symptemist-fasttext-9-ner-phrases.tsv -t symptemist -o ./out/testset_results

# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/en/tsv/multicardioner_track2_cardioccc_test_en.tsv -p ./data/predictions/testset/BioLinkBERT-base-drugtemist-en-fasttext-75-ner-phrases.tsv -t track2_en -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/en/tsv/multicardioner_track2_cardioccc_test_en.tsv -p ./data/predictions/testset/BioLinkBERT-base-drugtemist-en-fasttext-8-ner-phrases.tsv -t track2_en -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/en/tsv/multicardioner_track2_cardioccc_test_en.tsv -p ./data/predictions/testset/BioLinkBERT-base-drugtemist-en-fasttext-85-ner-phrases.tsv -t track2_en -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/en/tsv/multicardioner_track2_cardioccc_test_en.tsv -p ./data/predictions/testset/BioLinkBERT-base-drugtemist-en-fasttext-9-ner-phrases.tsv -t track2_en -o ./out/testset_results

# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/it/tsv/multicardioner_track2_cardioccc_test_it.tsv -p ./data/predictions/testset/bioBIT-drugtemist-it-fasttext-75-ner-phrases.tsv -t track2_it -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/it/tsv/multicardioner_track2_cardioccc_test_it.tsv -p ./data/predictions/testset/bioBIT-drugtemist-it-fasttext-8-ner-phrases.tsv -t track2_it -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/it/tsv/multicardioner_track2_cardioccc_test_it.tsv -p ./data/predictions/testset/bioBIT-drugtemist-it-fasttext-85-ner-phrases.tsv -t track2_it -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/it/tsv/multicardioner_track2_cardioccc_test_it.tsv -p ./data/predictions/testset/bioBIT-drugtemist-it-fasttext-9-ner-phrases.tsv -t track2_it -o ./out/testset_results

# python3 ner_evaluation.py -r ../cantemist/test-set/cantemist-ner/tsv/cantemist_ner_test.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-cantemist-fasttext-75-ner-phrases.tsv -t cantemist -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track1/cardioccc_test/tsv/multicardioner_track1_cardioccc_test.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-distemist-fasttext-75-ner-phrases.tsv -t track1 -o ./out/testset_results
# python3 ner_evaluation.py -r ../multicardioner/track2/cardioccc_test/es/tsv/multicardioner_track2_cardioccc_test_es.tsv -p ./data/predictions/testset/bsc-bio-ehr-es-drugtemist-fasttext-75-ner-phrases.tsv -t track2_es -o ./out/testset_results
