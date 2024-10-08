import os
import json
import copy


# METRICS
def calculate_scores(gold_standard, predictions, task):
    """
    Calculate micro-averaged precision, recall and f-score from two pandas dataframe
    Depending on the task, do some different pre-processing to the data
    """
    # Cumulative true positives, false positives, false negatives
    total_tp, total_fp, total_fn = 0, 0, 0
    total_overlap_tp, total_overlap_fp, total_overlap_fn = 0, 0, 0
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
        doc_overlap_tp, doc_overlap_fp, doc_overlap_fn = 0, 0, 0
        gold_doc = gs_files[document_id]
        #  Check if there are predictions for the current document, default to empty document if false
        if document_id not in pred_files.keys():
            predicted_doc = []
        else:
            predicted_doc = pred_files[document_id]

        if task == 'ner':
            # Create copy of document for overlapping matches calculation
            overlapping_predicted_doc = copy.deepcopy(predicted_doc)
            overlapping_gold_doc = copy.deepcopy(gold_doc)
            # No need to filter using label names as we only have one: SINTOMA; leaving this in for future reference just in case
            # overlapping_predicted_doc = list(filter(lambda x: x[-1] == label, overlapping_predicted_doc))
            # overlapping_gold_doc = list(filter(lambda x: x[-1] == label, overlapping_gold_doc))

        # Iterate through a copy of our gold mentions (strict matching)
        for gold_annotation in gold_doc[:]:
            # Iterate through predictions looking for a match
            for prediction in predicted_doc[:]:
                if set(gold_annotation) == set(prediction):
                    # Add a true positive
                    doc_tp += 1
                    # Remove elements from list to calculate later false positives and false negatives
                    predicted_doc.remove(prediction)
                    gold_doc.remove(gold_annotation)
                    break

        if task == 'ner':
            # Calculate overlaps
            for gold_top in overlapping_gold_doc[:]:
                for predicted_top in overlapping_predicted_doc[:]:
                    if is_overlap_match(gold_top, predicted_top):
                        doc_overlap_tp += 1
                        overlapping_predicted_doc.remove(predicted_top)
                        overlapping_gold_doc.remove(gold_top)
                        break

        # Get the number of false positives and false negatives from the items remaining in our lists
        doc_fp += len(predicted_doc)
        doc_fn += len(gold_doc)

        # Same but for overlapping
        if task == 'ner':
            doc_overlap_fp += len(overlapping_predicted_doc)
            doc_overlap_fn += len(overlapping_gold_doc)

        # Calculate document score (strict)
        doc_precision, doc_recall, doc_fscore = (
            calculate_precision_recall_f1(doc_tp, doc_fp, doc_fn))

        if task == 'ner':
            # Same but for overlap
            doc_overl_precision, doc_overl_recall, doc_overl_fscore = (
                calculate_precision_recall_f1(doc_overlap_tp, doc_overlap_fp, doc_overlap_fn))

            # Add to dictionary
            scores[document_id] = {"recall": round(doc_recall, 4),
                                   "precision": round(doc_precision, 4),
                                   "f_score": round(doc_fscore, 4)}
            scores[document_id].update({"recall_overlap": round(doc_overl_recall, 4),
                                        "precision_overlap": round(doc_overl_precision, 4),
                                        "f_score_overlap": round(doc_overl_fscore, 4)})

        elif task in ['norm', 'multi']:
            # Since we don't have true negatives in this evaluation setting,
            # we can just re-use the recall value as it's equivalent to accuracy
            scores[document_id] = {"accuracy": round(doc_recall, 4)}

        # Update totals
        total_tp += doc_tp
        total_fn += doc_fn
        total_fp += doc_fp

        if task == 'ner':
            total_overlap_tp += doc_overlap_tp
            total_overlap_fp += doc_overlap_fp
            total_overlap_fn += doc_overlap_fn

    # Now let's calculate the micro-averaged score using the cumulative TP, FP, FN
    total_precision, total_recall, total_fscore = calculate_precision_recall_f1(total_tp, total_fp, total_fn)
    if task == 'ner':
        total_overlap_precision, total_overlap_recall, total_overlap_fscore = (
            calculate_precision_recall_f1(total_overlap_tp, total_overlap_fp, total_overlap_fn))

        scores['total'] = {"recall": round(total_recall, 4),
                           "precision": round(total_precision, 4),
                           "f_score": round(total_fscore, 4)}
        scores['total'].update({"recall_overlap": round(total_overlap_recall, 4),
                                "precision_overlap": round(total_overlap_precision, 4),
                                "f_score_overlap": round(total_overlap_fscore, 4)})

    if task in ['norm', 'multi']:
        scores['total'] = {'accuracy': round(total_recall, 4)}

    return scores


def calculate_precision_recall_f1(tp, fp, fn):
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    if precision == 0 or recall == 0:
        f_score = 0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score


def is_overlap(a, b):
    return b[1] <= a[1] <= b[2] or a[1] <= b[1] <= a[2]


def is_overlap_match(a, b):
    # return is_overlap(a, b) and a.ptid == b.ptid
    # index 0 = filename, index 4 = label
    return is_overlap(a, b) and a[0] == b[0] and a[4] == b[4]


# HELPER
def write_results(scores, output_path):
    data = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    data['official_accuracy'] = scores['total']['accuracy']
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Official accuracy: {data['official_accuracy']}")
