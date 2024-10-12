import pandas as pd
from argparse import ArgumentParser
from utils import calculate_f1score, write_results


def main(argv=None):
    # Parse options
    parser = ArgumentParser()
    parser.add_argument("-r", "--reference", help=".TSV file with Gold Standard or reference annotations", required=True)
    parser.add_argument("-p", "--prediction", help=".TSV file with your predictions", required=True)
    parser.add_argument("-o", "--output_file", help="Path to save the scoring results", required=True)
    args = parser.parse_args(argv)

    # Read gold_standard and predictions
    df_gs = pd.read_csv(args.reference, sep="\t")
    df_preds = pd.read_csv(args.prediction, sep="\t")
    df_preds = df_preds.drop_duplicates(subset=["filename", "label", "start_span", "end_span"]).reset_index(drop=True)

    # Calculate results
    calculate_ner(df_gs, df_preds, args.output_file)


def calculate_ner(df_gs, df_preds, output_path):
    # Group annotations by filename
    list_gs_per_doc = df_gs.groupby('filename').apply(
        lambda x: x[["filename", 'start_span', 'end_span', "text",  "label"]].values.tolist())
    list_preds_per_doc = df_preds.groupby('filename').apply(
        lambda x: x[["filename", 'start_span', 'end_span', "text", "label"]].values.tolist())
    scores = calculate_f1score(list_gs_per_doc, list_preds_per_doc)
    write_results(scores, output_path)


if __name__ == "__main__":
    main()
