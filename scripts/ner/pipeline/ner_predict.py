import json
import os
import re
import csv
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import pipeline


# Initialize the tokenizer, model, and pipeline
def initialize_pipeline(model_name):
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise use CPU
    pipe = pipeline(task='token-classification', model=model_name, device=device)
    return pipe


def get_labels(argument):
    if 'distemist' in argument.lower():
        return ['B-ENFERMEDAD', 'I-ENFERMEDAD']
    elif 'drugtemist' in argument.lower():
        return ['B-FARMACO', 'I-FARMACO']
    elif 'cantemist' in argument.lower():
        return ['B-MORFOLOGIA_NEOPLASIA', 'I-MORFOLOGIA_NEOPLASIA']
    elif 'symptemist' in argument.lower():
        return ['B-SINTOMA', 'I-SINTOMA']
    else:
        raise (
            "If --input_file_path_conll or --input_file_path_json are set, the option 'dataset' has to be one of the "
            "following: 'distemist', 'drugtemist-es', 'drugtemist-it', 'drugtemist-en', 'cantemist', 'symptemist'. "
            "If --input_dir_path is set, the --model_name has to have either 'distemist' or 'drugtemist' in its name.")


# Process predictions for each example
def process_predictions(preprocessed_results, text, filename, labels, start_span_inside_document=None):
    processed_results = []
    for result in preprocessed_results:
        if result['entity'] not in labels:
            continue
        result['word'] = text[result['start']:result['end']]
        if start_span_inside_document:
            result['start'] += start_span_inside_document
            result['end'] += start_span_inside_document
        result['filename'] = filename
        processed_results.append(result)
    return processed_results


# Combines predictions according to the BIO tagging scheme, into the whole entity
def combine_BIO_entities(extracted_entities):
    combined_entities = []
    current_entity = None
    current_entity_text = ""
    current_entity_start = ""
    for entity in extracted_entities:
        if entity['entity'].startswith("B-"):  # Start of a new entity
            if current_entity is not None:
                combined_entities.append({
                    'start': current_entity_start,
                    'end': current_entity['end'],
                    'entity': current_entity['entity'][2:],
                    'word': current_entity_text.strip(),
                    'filename': current_entity['filename']
                })
            current_entity = entity
            current_entity_text = entity['word']
            current_entity_start = entity['start']
        elif entity['entity'].startswith("I-"):  # Part of the current entity
            if entity['start'] == current_entity['end']:
                current_entity_text += entity['word']
            else:
                current_entity_text += f" {entity['word']}"
            current_entity = entity
    # Add the last entity if it exists
    if current_entity is not None:
        combined_entities.append({
            'start': current_entity_start,
            'end': current_entity['end'],
            'entity': current_entity['entity'][2:],
            'word': current_entity_text.strip(),
            'filename': current_entity['filename']
        })
    return combined_entities


# Combines predictions when the entity is split into more than one entry
# (end_span of an entity is the same as the start_span of the next one)
def combine_split_entities(entities):
    df = pd.DataFrame(entities)

    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    df['word'] = df['word'].astype(str)

    merged_records = []
    current_record = None
    for _, row in df.iterrows():
        if current_record is None:
            current_record = row
        else:
            if (current_record['filename'] == row['filename'] and
                    current_record['end'] == row['start']):
                # Merge the current row with the current record
                current_record['end'] = row['end']
                current_record['word'] += row['word']
            else:
                # Save the current record and start a new one
                merged_records.append(current_record)
                current_record = row

    # Add the last record
    if current_record is not None:
        merged_records.append(current_record)

    # Convert merged records to a list of dictionaries
    merged_dicts = [row.to_dict() for _, row in pd.DataFrame(merged_records).iterrows()]

    return merged_dicts


# Returns the filenames and the spans from each example in the conll file
def get_examples_filenames_and_spans_from_conll_file(dev_file_path):
    with open(dev_file_path, 'r') as f:
        lines = f.readlines()
    filenames = []
    spans = {lines[0].split('\t')[1]: [{'start': int(lines[0].split('\t')[2].split('_')[0])}]}
    for i, line in enumerate(lines):
        if line.startswith('.'):
            for d in spans[line.split('\t')[1]]:
                if 'end' not in d:
                    d['end'] = int(line.split('\t')[2].split('_')[1])
        elif line == '\n':
            if i < len(lines) - 1:
                if lines[i + 1].split('\t')[1] in spans:
                    spans[lines[i + 1].split('\t')[1]].append({'start': int(lines[i + 1].split('\t')[2].split('_')[0])})
                else:
                    spans[lines[i + 1].split('\t')[1]] = [{'start': int(lines[i + 1].split('\t')[2].split('_')[0])}]
            filename = lines[i - 1].split('\t')[1]
            if f"{filename}.txt" not in filenames:
                filenames.append(f"{filename}.txt")
    filename = lines[i - 1].split('\t')[1]
    if f"{filename}.txt" not in filenames:
        filenames.append(f"{filename}.txt")
    return filenames, spans


def main(args):
    pipe = initialize_pipeline(args.model_name)
    if args.input_dir_path:
        labels = get_labels(args.model_name)
    elif args.input_file_path_json:
        labels = get_labels(args.input_file_path_json)
    else:
        labels = get_labels(args.input_file_path_conll)

    # Ensure output directory exists
    output_dir = '/'.join(args.output_file_path.split('/')[:-1])
    os.makedirs(output_dir, exist_ok=True)

    # Accumulate all predictions
    all_results = []
    if args.input_file_path_json:
        with open(args.input_file_path_json, 'r') as f:
            examples = json.load(f)
        with tqdm(total=len(examples), desc="Obtaining predictions") as pbar:
            for example in examples:
                preprocessed_predictions = pipe(example[0])
                results = process_predictions(preprocessed_predictions, example[0], example[1], labels,
                                              start_span_inside_document=example[2])
                all_results.extend(results)
                pbar.update(1)
    else:
        if args.input_file_path_conll:
            files, _ = get_examples_filenames_and_spans_from_conll_file(args.input_file_path_conll)
            dirs = [
                '../../datasets/multicardioner/track2/drugtemist_train/es/brat',
                '../../datasets/multicardioner/track2/cardioccc_dev/es/brat'
            ]
        else:
            files = list(sorted(os.listdir(args.input_dir_path), key=lambda x: int(re.search(r'\d+', x).group())))
            dirs = [args.input_dir_path]

        with tqdm(total=len(files), desc="Obtaining predictions") as pbar:
            for filename in files:
                for dir in dirs:
                    file_path = os.path.join(dir, filename)
                    if os.path.exists(file_path):
                        break
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                preprocessed_predictions = pipe(text)
                results = process_predictions(preprocessed_predictions, text, filename, labels)
                all_results.extend(results)
                pbar.update(1)

    print("Combining tokens into entities")
    all_results = combine_split_entities(all_results)
    all_results = combine_BIO_entities(all_results)

    if args.input_file_path_conll:
        _, dev_set_example_spans = get_examples_filenames_and_spans_from_conll_file(args.input_file_path_conll)
        new_all_results = []
        for result in all_results:
            for example_span in dev_set_example_spans[result['filename'].removesuffix('.txt')]:
                if result['start'] > example_span['start'] and result['end'] < example_span['end']:
                    new_all_results.append(result)
                    continue
        all_results = new_all_results

    # Write TSV and JSON files
    all_results = sorted(all_results, key=lambda x: (x['filename'], x['start']))
    with open(args.predictions_file_path, 'w', newline='', encoding='utf-8') as csvfile, \
            open(args.output_file_path, 'w') as jsonfile:
        fieldnames = ['filename', 'label', 'start_span', 'end_span', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        with tqdm(total=len(all_results), desc="Saving results") as pbar:
            all_predictions_dict = {'predictions': []}
            for result in all_results:
                prediction_dict = {
                    'filename': result['filename'],
                    'label': result['entity'].split('-')[-1],
                    'start_span': result['start'],
                    'end_span': result['end'],
                    'text': result['word']
                }
                all_predictions_dict['predictions'].append(prediction_dict)
                writer.writerow(prediction_dict)
                pbar.update(1)
            json.dump(all_predictions_dict, jsonfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform token classification inference on a collection of files")
    parser.add_argument("--model_name", type=str, help="Name of the model to use for inference", required=True)
    parser.add_argument("--input_file_path_conll", type=str,
                        help="Validation file (a devset) containing examples to perform inference on")
    parser.add_argument("--input_file_path_json", type=str,
                        help="Json file (a testset) containing examples (phrases) to perform inference on")
    parser.add_argument("--input_dir_path", type=str,
                        help="Directory (the test+background folder) containing input files to perform inference on")
    parser.add_argument("--output_file_path", type=str, help="File path of the final results file", required=True)
    parser.add_argument("--predictions_file_path", type=str, help="File path of the predictions table", required=True)
    args = parser.parse_args()
    main(args)
