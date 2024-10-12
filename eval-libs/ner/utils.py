"""
MultiCardioNER evaluation library main script.
Heavily based on the MedProcNER library, which is in turn based on the DisTEMIST and MEDDOPLACE evaluation scripts.
@author: salva
"""
import json
import os


def inside_threshold(gold_start_span, pred_start_span, gold_end_span, pred_end_span, threshold_1=20, threshold_2=5):
    start_difference = abs(gold_start_span - pred_start_span)
    end_difference = abs(gold_end_span - pred_end_span)
    if (((start_difference != 0) ^ (end_difference != 0)) and start_difference < threshold_1 and end_difference < threshold_1
        or (start_difference != 0) and (end_difference != 0) and start_difference < threshold_2 and end_difference < threshold_2):
        return True
    return False


def word_frequency(data):
    word_freq = {}
    for item in data:
        word = item[3]
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
    return sorted_word_freq


def average_start(data):
    all_starts = []
    for item in data:
        all_starts.append(item[1])
    avg_start = round(sum(all_starts) / len(all_starts)) if len(all_starts) > 0 else 0
    return avg_start


def calculate_fscore(gold_standard, predictions):
    """
    Calculate micro-averaged precision, recall and f-score from two pandas dataframe
    Depending on the task, do some different pre-processing to the data
    """
    # Cumulative true positives, false positives, false negatives
    total_tp, total_fp, total_fn = 0, 0, 0
    almost, total_ga, total_p = 0, 0, 0
    total_tp_entities, total_fp_entities, total_fn_entities = [], [], []
    tp_entities_per_doc, fp_entities_per_doc, fn_entities_per_doc = {}, {}, {}
    # Dictionary to store files in gold and prediction data.
    gs_files = {}
    pred_files = {}
    for document in gold_standard:
        document_id = document[0][0]
        gs_files[document_id] = document
    for document in predictions:
        document_id = document[0][0]
        pred_files[document_id] = document

    # Dictionary to store scores
    scores = {}

    # Iterate through documents in the Gold Standard
    for document_id in gs_files.keys():
        doc_tp, doc_fp, doc_fn = 0, 0, 0
        gold_doc = gs_files[document_id]
        #  Check if there are predictions for the current document, default to empty document if false
        if document_id not in pred_files.keys():
            predicted_doc = []
        else:
            predicted_doc = pred_files[document_id]
        total_ga += len(gold_doc)
        total_p += len(predicted_doc)
        # Iterate through a copy of our gold mentions
        for gold_annotation in gold_doc[:]:
            # Iterate through predictions looking for a match
            for prediction in predicted_doc[:]:
                if inside_threshold(gold_annotation[1], prediction[1], gold_annotation[2], prediction[2]):
                    almost += 1
                if set(gold_annotation) == set(prediction):
                    # Add a true positive
                    doc_tp += 1
                    # Remove elements from list to calculate later false positives and false negatives
                    predicted_doc.remove(prediction)
                    gold_doc.remove(gold_annotation)
                    total_tp_entities.append(prediction)
                    if document_id in tp_entities_per_doc:
                        tp_entities_per_doc[document_id].append(prediction)
                    else:
                        tp_entities_per_doc[document_id] = [prediction]
                    break
        # Get the number of false positives and false negatives from the items remaining in our lists
        doc_fp += len(predicted_doc)
        doc_fn += len(gold_doc)
        total_fp_entities.extend(predicted_doc)
        total_fn_entities.extend(gold_doc)
        fp_entities_per_doc[document_id] = predicted_doc
        fn_entities_per_doc[document_id] = gold_doc
        # Calculate document score
        try:
            precision = doc_tp / (doc_tp + doc_fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = doc_tp / (doc_tp + doc_fn)
        except ZeroDivisionError:
            recall = 0
        if precision == 0 or recall == 0:
            f_score = 0
        else:
            f_score = 2 * precision * recall / (precision + recall)
        # Add to dictionary
        scores[document_id] = {"recall": round(recall, 4), "precision": round(precision, 4), "f_score": round(f_score, 4)}
        # Update totals
        total_tp += doc_tp
        total_fn += doc_fn
        total_fp += doc_fp

    wf = word_frequency(total_fn_entities)
    p_avg_start, fn_avg_start = average_start(total_tp_entities+total_fp_entities), average_start(total_fn_entities)
    # Now let's calculate the micro-averaged score using the cumulative TP, FP, FN
    try:
        precision = total_tp / (total_tp + total_fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = total_tp / (total_tp + total_fn)
    except ZeroDivisionError:
        recall = 0
    if precision == 0 or recall == 0:
        f_score = 0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    scores['total'] = {"recall": round(recall, 4), "precision": round(precision, 4), "f_score": round(f_score, 4)}
    scores['statistics'] = {
        "total_gold_annotations": total_ga,
        "total_predictions": total_p,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "fn_almost_tp": almost,
        "p_average_start_span": p_avg_start,
        "fn_average_start_span": fn_avg_start,
    }
    scores["fn_freq"] = wf
    scores["entities"] = {
        "true_positives": tp_entities_per_doc,
        "false_positives": fp_entities_per_doc,
        "false_negatives": fn_entities_per_doc
    }

    return scores


def check_top_false_negative_presence_in_training_data(data, task):
    conll_file_path, dir_paths = '', []
    if task in ['track1', 'track2_es']:
        conll_file_path = '../../scripts/ner/conll-parsing/multicardioner-parse/out/drugtemist_train.conll'
        dir_paths = ['../multicardioner/track1/distemist_train/brat', '../multicardioner/track1/cardioccc_dev/brat']
    elif task == 'track2_it':
        conll_file_path = '../../scripts/ner/conll-parsing/multicardioner-parse/out/drugtemist_train_it.conll'
        dir_paths = ['../multicardioner/track2/drugtemist_train/it/brat', '../multicardioner/track2/cardioccc_dev/it/brat']
    elif task == 'track2_en':
        conll_file_path = '../../scripts/ner/conll-parsing/multicardioner-parse/out/drugtemist_train_en.conll'
        dir_paths = ['../multicardioner/track2/drugtemist_train/en/brat', '../multicardioner/track2/cardioccc_dev/en/brat']
    elif task == 'symptemist':
        conll_file_path = '../../scripts/ner/conll-parsing/symptemist-parse/out/train.conll'
        dir_paths = ['../symptemist/train/subtask1-ner/brat']
    elif task == 'cantemist':
        conll_file_path = '../../scripts/ner/conll-parsing/cantemist-parse/out/train.conll'
        dir_paths = ['../cantemist/train-set/cantemist-ner']
    else:
        print('IMPOSSIBLEEEEE')

    # Get filenames from training file.
    with open(conll_file_path, 'r') as f:
        lines = f.readlines()
    filenames = []
    for i, line in enumerate(lines):
        if line == '\n':
            filenames.append(lines[i - 1].split('\t')[1])
    filenames = list(set(filenames))

    # Read the contents of the files in filenames.
    contents = []
    for filename in filenames:
        for dir_path in dir_paths:
            path = os.path.join(dir_path, filename + '.txt')
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as file:
                    contents.append(file.read())
                    break

    # Check entity presence in files
    result = {}
    for entity, count in data['fn_freq'].items():
        if count > 10:
            found = any(entity in content for content in contents)
            presence = 'Present' if found else 'Not Present'
            result[entity] = {'frequency': count, 'present_in_training_data': presence}
    data['fn_freq'] = result


def write_results(task, scores, output_path, verbose=False):
    """
    Helper function to write the results for each of the tasks
    """
    headers_dict = {'track1': 'MultiCardioNER Shared Task: Subtask 1 (Disease Named Entity Recognition) Results',
                    'track2_es': 'MultiCardioNER Shared Task: Subtask 2 (Multilingual Drug Named Entity Recognition) Spanish Results',
                    'track2_en': 'MultiCardioNER Shared Task: Subtask 2 (Multilingual Drug Named Entity Recognition) English Results',
                    'track2_it': 'MultiCardioNER Shared Task: Subtask 2 (Multilingual Drug Named Entity Recognition) Italian Results',
                    'symptemist': 'SympTEMIST Shared Task: Subtask 1 (Symptom Named Entity Recognition) Results',
                    'cantemist': 'CANTEMIST Shared Task: Subtask 1 (Tumor Named Entity Recognition) Results'
                    }
    if task in ['track1', 'track2_es']:
        model_name = f'{os.path.basename(output_path)}'.split('_')[-1].split('.')[0]
        with open(f'out/testset_multicardioner_comparison_files/{model_name}_comparison-file.json', 'w') as f1:
            f1.write(json.dumps(scores['entities'], indent=4))
    with open(output_path, 'w') as f_out:
        # This looks super ugly, but if we keep the indentation it will also appear in the output file
        f_out.write("""-------------------------------------------------------------------
{}
-------------------------------------------------------------------
""".format(headers_dict[task]))
        if verbose:
            for k in scores.keys():
                if k != 'total':
                    f_out.write("""-------------------------------------------------------------------
Results for document: {}
-------------------------------------------------------------------
Precision: {}
Recall: {}
F-score: {}
""".format(k, scores[k]["precision"], scores[k]["recall"], scores[k]["f_score"]))

        f_out.write("""-------------------------------------------------------------------
Overall results:
-------------------------------------------------------------------
Micro-average precision: {}
Micro-average recall: {}
Micro-average F-score: {}
""".format(scores['total']["precision"], scores['total']["recall"], scores['total']["f_score"]))
        f_out.write("""-------------------------------------------------------------------
Statistics:
-------------------------------------------------------------------
Total gold annotations: {}
Total predictions: {}
True positives: {}
False positives: {}
False negatives: {}
False negatives that were almost true positives: {}
Average "span start" of true + false positives: {}
Average "span start" of false negatives: {}
""".format(
            scores['statistics']["total_gold_annotations"],
            scores['statistics']["total_predictions"],
            scores['statistics']["true_positives"],
            scores['statistics']["false_positives"],
            scores['statistics']["false_negatives"],
            scores['statistics']["fn_almost_tp"],
            scores['statistics']["p_average_start_span"],
            scores['statistics']["fn_average_start_span"]
        ))
        f_out.write(f'-------------------------------------------------------------------\n')
        f_out.write(f'Frequency and Presence in Training Data:\n')
        f_out.write(f'-------------------------------------------------------------------\n')
        for entity, stats in scores['fn_freq'].items():
            f_out.write(f'{stats["frequency"]} | {stats["present_in_training_data"]} | {entity}\n')
    print("Written MultiCardioNER {} scores to {}".format(task, output_path))
