import torch
import numpy as np
from tqdm import tqdm
from pytorch_metric_learning.losses import MultiSimilarityLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner


def calculate_loss_batch(ms_loss, miner, all_embeddings, all_labels, batch_size, desc):
    total_loss = 0
    num_batches = (len(all_embeddings) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), total=num_batches, desc=desc):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(all_embeddings))

        batch_embeddings = all_embeddings[start_idx:end_idx]
        batch_labels = all_labels[start_idx:end_idx]

        # Get hard pairs for this batch
        hard_pairs = miner(batch_embeddings, batch_labels)

        # Calculate loss for this batch
        batch_loss = ms_loss(batch_embeddings, batch_labels, hard_pairs)
        total_loss += batch_loss.item()

    return total_loss / num_batches


def calculate_losses(embeds_info, labels_info, batch_size):
    # deconstruct arguments
    query_embeds, complete_dict_dense_embeds, only_test_codes_dict_dense_embeds = embeds_info
    query_labels, complete_dict_labels, only_test_codes_dict_labels = labels_info

    # obtain query and dict embeddings
    query_embeddings = torch.from_numpy(query_embeds)
    dict_embeddings_complete = torch.from_numpy(complete_dict_dense_embeds)
    dict_embeddings_only_test_codes = torch.from_numpy(only_test_codes_dict_dense_embeds)

    # obtain query and dict labels
    query_labels = torch.tensor([hash(label) for label in query_labels])
    dict_labels_complete = torch.tensor([hash(row[1]) for row in complete_dict_labels])
    dict_labels_only_test_codes = torch.tensor([hash(row[1]) for row in only_test_codes_dict_labels])

    # join query and dict embeddings and labels
    all_embeddings_complete = torch.cat([query_embeddings, dict_embeddings_complete], dim=0)
    all_embeddings_only_test_codes = torch.cat([query_embeddings, dict_embeddings_only_test_codes], dim=0)
    all_labels_complete = torch.cat([query_labels, dict_labels_complete])
    all_labels_only_test_codes = torch.cat([query_labels, dict_labels_only_test_codes])

    # initialize loss and miner
    ms_loss = MultiSimilarityLoss()
    miner = MultiSimilarityMiner()

    # calculate loss by batches
    validation_loss_complete = calculate_loss_batch(ms_loss, miner, all_embeddings_complete, all_labels_complete, batch_size, "- Calculating Val Loss 1")
    validation_loss_only_test_codes = calculate_loss_batch(ms_loss, miner, all_embeddings_only_test_codes, all_labels_only_test_codes, batch_size, "- Calculating Val Loss 2")

    return validation_loss_complete, validation_loss_only_test_codes


def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|")))) > 0)


def predict_and_evaluate(model_wrapper, eval_dictionary_complete, eval_dictionary_only_test_codes, eval_queries, agg_mode="cls", batch_size=1024):
    complete_dict_names = [row[0] for row in eval_dictionary_complete]
    only_test_codes_dict_names = [row[0] for row in eval_dictionary_only_test_codes]
    complete_dict_dense_embeds = model_wrapper.embed_dense(
        names=complete_dict_names, show_progress=True, agg_mode=agg_mode, description="- Embedding Dictionary 1"
    )
    only_test_codes_dict_dense_embeds = model_wrapper.embed_dense(
        names=only_test_codes_dict_names, show_progress=True, agg_mode=agg_mode, description="- Embedding Dictionary 2"
    )

    mean_centering = False
    if mean_centering:
        complete_tgt_space_mean_vec = complete_dict_dense_embeds.mean(0)
        complete_dict_dense_embeds -= complete_tgt_space_mean_vec
        only_test_codes_tgt_space_mean_vec = only_test_codes_dict_dense_embeds.mean(0)
        only_test_codes_dict_dense_embeds -= only_test_codes_tgt_space_mean_vec

    predictions = []
    query_embeds = []
    query_labels = []
    for eval_query in tqdm(eval_queries, total=len(eval_queries), desc="- Obtaining Predictions"):
        text = eval_query[0]
        golden_cui = eval_query[1]

        text_dense_embeds = model_wrapper.embed_dense(names=[text], agg_mode=agg_mode)

        if mean_centering:
            text_dense_embeds -= complete_tgt_space_mean_vec

        query_embeds.append(text_dense_embeds[0])
        query_labels.append(golden_cui)

        # get score matrix
        dense_score_matrix = model_wrapper.get_score_matrix(
            query_embeds=text_dense_embeds,
            dict_embeds=complete_dict_dense_embeds,
        )
        score_matrix = dense_score_matrix

        candidate_id = model_wrapper.retrieve_candidate_cuda(score_matrix=score_matrix, topk=1, batch_size=16)[0].tolist()[0]
        np_candidate = eval_dictionary_complete[candidate_id]

        predictions.append({
            'text': text,
            'golden_cui': golden_cui,
            'candidate_name': np_candidate[0],
            'candidate_label': np_candidate[1],
            'correct': check_label(np_candidate[1], golden_cui)
        })

    accuracy = sum(item["correct"] for item in predictions) / len(predictions)

    query_embeds = np.vstack(query_embeds)
    complete_dict_labels = [row[1] for row in eval_dictionary_complete]
    only_test_codes_dict_labels = [row[1] for row in eval_dictionary_only_test_codes]
    validation_loss_complete, validation_loss_only_test_codes = calculate_losses(
        (query_embeds, complete_dict_dense_embeds, only_test_codes_dict_dense_embeds),
        (query_labels, complete_dict_labels, only_test_codes_dict_labels),
        batch_size
    )

    torch.cuda.empty_cache()

    return predictions, accuracy, validation_loss_complete, validation_loss_only_test_codes
