#!/usr/bin/env python

import time
import copy
import json
import torch
import logging
import argparse
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import sys
sys.path.append("../")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.model_wrapper import Model_Wrapper
from evaluation.utils import predict_and_evaluate
from src.metric_learning import Sap_Metric_Learning
from src.data_loader import MetricLearningDataset, MetricLearningDataset_pairwise, DictionaryDataset, QueryDataset_custom

LOGGER = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(description='sapbert train')

    # Required
    parser.add_argument('--model_dir', required=True, help='Directory or Huggingface ID for pretrained model')
    parser.add_argument('--training_file_path', type=str, required=True, help='Training file path')
    parser.add_argument('--validation_file_path', type=str, help='Validation file path')
    parser.add_argument('--complete_dictionary_path', type=str, help='Path of dictionary file with every code')
    parser.add_argument('--only_test_codes_dictionary_path', type=str,
                        help='Path with dictionary file woth only the test codes')
    parser.add_argument('--output_dir_for_best_model', type=str,
                        help='Directory in which the best model will be saved')
    parser.add_argument('--results_file_path', type=str, required=True, help='Path for the file with all results')

    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)

    # Train config
    parser.add_argument('--use_cuda', action="store_true")
    parser.add_argument('--learning_rate', help='learning rate', default=0.0001, type=float)
    parser.add_argument('--weight_decay', help='weight decay', default=0.01, type=float)
    parser.add_argument('--train_batch_size', help='train batch size', default=240, type=int)
    parser.add_argument('--epoch', help='epoch to train', default=3, type=int)
    parser.add_argument('--save_checkpoint_all', action="store_true")
    parser.add_argument('--checkpoint_step', type=int, default=10000000)
    parser.add_argument('--amp', action="store_true", help="automatic mixed precision training")
    parser.add_argument('--parallel', action="store_true")
    parser.add_argument('--pairwise', action="store_true", help="if loading pairwise formatted datasets")
    parser.add_argument('--random_seed', help='epoch to train', default=1996, type=int)
    parser.add_argument('--loss', help="{ms_loss|cosine_loss|circle_loss|triplet_loss}}", default="ms_loss")
    parser.add_argument('--use_miner', action="store_true")
    parser.add_argument('--miner_margin', default=0.2, type=float)
    parser.add_argument('--type_of_triplets', default="all", type=str)
    parser.add_argument('--agg_mode', default="cls", type=str, help="{cls|mean|mean_all_tok}")
    parser.add_argument('--trust_remote_code', action="store_true",
                        help="allow for custom models defined in their own modeling files")
    parser.add_argument('--distance_threshold', type=str,
                        help='distance threshold for the augmented dataset. used to clarify results file keys')

    args = parser.parse_args()
    return args


def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%I:%M:%S %p')
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


def train(model, data_loader, scaler=None, amp=False):
    train_loss = 0
    train_steps = 0
    model.cuda()
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="- Training Model"):
        model.optimizer.zero_grad()

        batch_x1, batch_x2, batch_y = data
        batch_x_cuda1, batch_x_cuda2 = {}, {}
        for k, v in batch_x1.items():
            batch_x_cuda1[k] = v.cuda()
        for k, v in batch_x2.items():
            batch_x_cuda2[k] = v.cuda()

        batch_y_cuda = batch_y.cuda()

        if amp:
            with autocast():
                loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)
        else:
            loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)
        if amp:
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            loss.backward()
            model.optimizer.step()

        train_loss += loss.item()
        train_steps += 1

    train_loss /= (train_steps + 1e-9)
    return train_loss


def main(args):
    # logging and seed
    init_logging()
    torch.manual_seed(args.random_seed)

    # load BERT tokenizer, dense_encoder
    model_wrapper = Model_Wrapper()
    encoder, tokenizer = model_wrapper.load_bert(
        path=args.model_dir,
        max_length=args.max_length,
        use_cuda=args.use_cuda,
        trust_remote_code=args.trust_remote_code,
    )

    # load SAP model
    model = Sap_Metric_Learning(
        encoder=encoder,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_cuda=args.use_cuda,
        pairwise=args.pairwise,
        loss=args.loss,
        use_miner=args.use_miner,
        miner_margin=args.miner_margin,
        type_of_triplets=args.type_of_triplets,
        agg_mode=args.agg_mode,
    )

    # parallel training
    if args.parallel:
        model.encoder = torch.nn.DataParallel(model.encoder)

    def collate_fn_batch_encoding(batch):
        query1, query2, query_id = zip(*batch)
        query_encodings1 = tokenizer.batch_encode_plus(
            list(query1),
            max_length=args.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        query_encodings2 = tokenizer.batch_encode_plus(
            list(query2),
            max_length=args.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        query_ids = torch.tensor(list(query_id))
        return query_encodings1, query_encodings2, query_ids

    # load data loader
    train_set = MetricLearningDataset_pairwise(path=args.training_file_path, tokenizer=tokenizer)
    training_data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=12,
        collate_fn=collate_fn_batch_encoding
    )

    # load dictionary and data queries
    if args.validation_file_path:
        eval_dictionary_complete = DictionaryDataset(dictionary_path=args.complete_dictionary_path).data
        eval_dictionary_only_test_codes = DictionaryDataset(dictionary_path=args.only_test_codes_dictionary_path).data
        eval_queries = QueryDataset_custom(data_dir=args.validation_file_path, filter_duplicate=False).data

    # mixed precision training
    scaler = GradScaler() if args.amp else None

    # training loop
    start = time.time()
    current_best_model = {'accuracy': -1}
    stats = {'accuracies': [], 'train_losses': [], 'val_losses': []}
    for epoch in range(1, args.epoch + 1):
        LOGGER.info("Epoch {}/{}".format(epoch, args.epoch))

        LOGGER.info("Training")
        train_loss = train(
            model=model,
            data_loader=training_data_loader,
            scaler=scaler,
            amp=True if args.amp else False
        )
        LOGGER.info(f'Training Loss: {train_loss}')
        stats['train_losses'].append(train_loss)

        if args.validation_file_path:
            LOGGER.info("Evaluating")
            _, accuracy_evalset, validation_loss_complete, validation_loss_only_test_codes = predict_and_evaluate(
                model_wrapper=model_wrapper,
                eval_dictionary_complete=eval_dictionary_complete,
                eval_dictionary_only_test_codes=eval_dictionary_only_test_codes,
                eval_queries=eval_queries,
                agg_mode=args.agg_mode
            )
            LOGGER.info(f"Accuracy: {accuracy_evalset}")
            stats['accuracies'].append(accuracy_evalset)
            LOGGER.info(f"Validation Loss Complete: {validation_loss_complete}")
            LOGGER.info(f"Validation Loss Only Test Codes: {validation_loss_only_test_codes}")
            stats['val_losses'].append((validation_loss_complete, validation_loss_only_test_codes))

            if accuracy_evalset > current_best_model['accuracy']:
                LOGGER.info("Saving Best Model")
                del current_best_model
                current_best_model = {
                    'model': copy.deepcopy(model_wrapper),
                    'accuracy': accuracy_evalset,
                    'epoch': epoch,
                }
        else:
            if epoch == args.epoch:
                current_best_model = {
                    'model': copy.deepcopy(model_wrapper),
                    'epoch': epoch,
                }

    # log training and validation time
    end = time.time()
    training_time = end - start
    training_hour = int(training_time / 60 / 60)
    training_minute = int(training_time / 60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training and validation time: {}h, {}min, {}sec".format(training_hour, training_minute, training_second))

    os.makedirs(os.path.dirname(args.results_file_path), exist_ok=True)

    # saving best model
    if args.output_dir_for_best_model:
        best_model_path = args.output_dir_for_best_model.replace("EPOCH", str(current_best_model['epoch']))
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        current_best_model['model'].save_model(best_model_path)
        # model_wrapper.save_model(best_model_path)

    # saving stats
    if args.validation_file_path:
        stats['best_model'] = {'accuracy': current_best_model['accuracy'], 'epoch': current_best_model['epoch']}
    if os.path.exists(args.results_file_path):
        with open(args.results_file_path, 'r') as file:
            data = json.load(file)
    else:
        data = {}
    if args.distance_threshold:
        data[f'{args.train_batch_size}_{args.learning_rate:.0e}_{args.distance_threshold}'] = stats
    else:
        data[f'{args.train_batch_size}_{args.learning_rate:.0e}'] = stats
    with open(args.results_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    del model
    del encoder
    del tokenizer
    del model_wrapper
    del train_set
    del training_data_loader
    torch.cuda.empty_cache()


if __name__ == '__main__':
    args = parse_args()
    main(args)
