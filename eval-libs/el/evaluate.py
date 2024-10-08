import pandas as pd
from argparse import ArgumentParser
from utils import calculate_scores, write_results
from build_el_test_file_reference_tsvs import build_el_test_file_reference_tsv


def main(argv=None):
    # Parse options
    parser = ArgumentParser()
    parser.add_argument("-d", "--reference", help="name of the dataset. OPTIONAL: only used to build the Gold Standard TSV when the path passed in `-r` is invalid (does not exist)")
    parser.add_argument("-l", "--reference", help="language of the dataset. OPTIONAL: only used to build the Gold Standard TSV when the path passed in `-r` is invalid (does not exist)")
    parser.add_argument("-r", "--reference", help="TSV file with Gold Standard or reference annotations", required=True)
    parser.add_argument("-p", "--prediction", help="TSV file with your predictions", required=True)
    parser.add_argument("-o", "--output_file", help="Path to save the scoring results", required=True)
    args = parser.parse_args(argv)

    # Read gold_standard and predictions
    try:
        df_gs = pd.read_csv(args.reference, sep="\t", engine='python', quoting=3, dtype=str)
    except FileNotFoundError:
        df_gs = build_el_test_file_reference_tsv(args.dataset, args.language)
    df_preds = pd.read_csv(args.prediction, sep="\t", engine='python', quoting=3, dtype=str)
    df_preds = df_preds.drop_duplicates(subset=["filename", "label", "start_span", "end_span"]).reset_index(drop=True)

    calculate_norm(df_gs, df_preds, args.output_file)


def calculate_norm(df_gs, df_preds, output_path):
    # Group annotations by filename
    list_gs_per_doc = df_gs.groupby('filename').apply(lambda x: x[[
        "filename", 'start_span', 'end_span', "text", "label", "code"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds.groupby('filename').apply(
        lambda x: x[["filename", 'start_span', 'end_span', "text", "label", "code"]].values.tolist()).to_list()
    scores = calculate_scores(list_gs_per_doc, list_preds_per_doc, 'norm')
    write_results(scores, output_path)


if __name__ == "__main__":
    main()
