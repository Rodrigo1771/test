import os
import csv
import json
import logging
import argparse

import sys
sys.path.append("../")

from utils import predict_and_evaluate
from src.model_wrapper import Model_Wrapper
from src.data_loader import DictionaryDataset, QueryDataset_custom

LOGGER = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(description='sapbert evaluation')

    # Paths
    parser.add_argument('--model_dir', required=True, help='Directory for model')
    parser.add_argument('--test_file_path', type=str, required=True, help='data set to evaluate')
    parser.add_argument('--complete_dictionary_path', type=str, required=True,
                        help='Path of dictionary file with every code')
    parser.add_argument('--only_test_codes_dictionary_path', type=str, required=True,
                        help='Path with dictionary file woth only the test codes')
    parser.add_argument('--output_file_path', type=str, default='./output/', help='Directory for output')
    parser.add_argument('--predictions_tsv_file_path', type=str, help="path to save predictions")

    args = parser.parse_args()
    return args


def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%I:%M:%S %p')
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


def main(args):
    init_logging()

    # load model, dictionary, and data queries
    model_wrapper = Model_Wrapper().load_model(path=args.model_dir, max_length=25, use_cuda=True)
    eval_dictionary_complete = DictionaryDataset(dictionary_path=args.complete_dictionary_path).data
    eval_dictionary_only_test_codes = DictionaryDataset(dictionary_path=args.only_test_codes_dictionary_path).data
    eval_queries = QueryDataset_custom(data_dir=args.test_file_path, filter_duplicate=False).data

    LOGGER.info("Evaluating")

    predictions, accuracy, _, _ = predict_and_evaluate(
        model_wrapper=model_wrapper,
        eval_dictionary_complete=eval_dictionary_complete,
        eval_dictionary_only_test_codes=eval_dictionary_only_test_codes,
        eval_queries=eval_queries,
        agg_mode='cls'
    )

    LOGGER.info(f"Accuracy: {accuracy}")
    result = {"unofficial_accuracy": accuracy, "predictions": predictions}

    with open(args.output_file_path, 'w') as f:
        json.dump(result, f, indent=4)

    if args.predictions_tsv_file_path:
        dir_path, filename = os.path.split(args.test_file_path)
        mode = 'final-model' if 'final-model' in dir_path else 'hyperparameter-search'
        _, _, lang, _, _ = filename.split('_')
        reference_info_file_path = os.path.join(dir_path, f'aux/test_file_all_info_{lang}_{mode}.json')
        with open(reference_info_file_path, 'r') as info, open(args.predictions_tsv_file_path, 'w') as out:
            info_dict = json.load(info)
            tsv_writer = csv.writer(out, delimiter='\t')
            tsv_writer.writerow(['filename', 'label', 'start_span', 'end_span', 'text', 'code'])
            for entry, prediction in zip(info_dict, predictions):
                tsv_writer.writerow(
                    [
                        entry['filename'],
                        entry['label'],
                        entry['start_span'],
                        entry['end_span'],
                        entry['entity'],
                        prediction['candidate_label']
                    ]
                )

    del model_wrapper
    del eval_dictionary_complete


if __name__ == '__main__':
    args = parse_args()
    main(args)
