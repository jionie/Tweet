import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import *
from sklearn.model_selection import StratifiedKFold

############################################ Define augments for test

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('-data_path', type=str,
                    default="/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/",
                    required=False, help='specify the path for data')
parser.add_argument('--n_splits', type=int, default=5, required=False, help='specify the number of folds')
parser.add_argument('--seed', type=int, default=42, required=False,
                    help='specify the random seed for splitting dataset')
parser.add_argument('--max_seq_length', type=int, default=384, required=False,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                         "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--max_query_length", default=64, type=int, required=False,
                    help="The maximum number of tokens for the question. Questions longer than this will "
                         "be truncated to this length.")
parser.add_argument('--doc_stride', type=int, default=128, required=False,
                    help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument('--threads', type=int, default=1, required=False,
                    help='multiple threads for converting example to features')
parser.add_argument('--save_path', type=str,
                    default="/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/", required=False,
                    help='specify the path for saving splitted csv')
parser.add_argument('--fold', type=int, default=0, required=False,
                    help='specify the fold for testing dataloader')
parser.add_argument('--batch_size', type=int, default=4, required=False,
                    help='specify the batch_size for testing dataloader')
parser.add_argument('--val_batch_size', type=int, default=4, required=False,
                    help='specify the val_batch_size for testing dataloader')
parser.add_argument('--num_workers', type=int, default=0, required=False,
                    help='specify the num_workers for testing dataloader')
parser.add_argument('--model_type', type=str, default="bert-base-uncased", required=False,
                    help='specify the model_type for BertTokenizer')


############################################ Define Helper functions and dataset function from transformers run_swag.py
def load_and_cache_examples(data_dir, input_file, model_type, tokenizer, max_seq_length=384, max_query_length=64,
                            doc_stride=128, threads=1, mode="train", seed=42, fold=0, output_examples=True):

    # Load data features from cache or dataset file
    if mode == "train":
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode,
                model_type,
                str(max_seq_length),
                str(seed),
                str(fold)
            ),
        )
        evaluate = False
    elif mode == "test":
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}".format(
                mode,
                model_type,
                str(max_seq_length),
            ),
        )
        evaluate = True
    else:
        raise NotImplementedError

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file):
        print("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        print("Creating features from dataset file at %s", input_file)

        processor = SquadV2Processor()

        if not evaluate:
            examples = processor.get_train_examples(data_dir, filename=input_file)
        else:
            examples = processor.get_dev_examples(data_dir, filename=input_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=threads,
        )

        print("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features

    return dataset


############################################ Define preprocessing function
# train_data = [
#     {
#         'context': "This tweet sentiment extraction challenge is great",
#         'qas': [
#             {
#                 'id': "00001",
#                 'question': "positive",
#                 'answers': [
#                     {
#                         'text': "is great",
#                         'answer_start': 43
#                     }
#                 ]
#             }
#         ]
#     }
#     ]


def train_preprocessing(data_path, save_path):

    input_file = pd.read_csv(data_path)
    input_file = np.array(input_file)

    output = {'version': 'v1.0', 'data': []}

    for line in input_file:
        paragraphs = []

        context = line[1]

        qas = []
        question = line[-1]
        qid = line[0]
        answers = []
        answer = line[2]

        if type(answer) != str or type(context) != str or type(question) != str:
            print(context, type(context))
            print(answer, type(answer))
            print(question, type(question))
            continue
        answer_starts = find_all(context, answer)

        for answer_start in answer_starts:
            answers.append({'answer_start': answer_start, 'text': answer})
        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})

        paragraphs.append({'context': context, 'qas': qas})
        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

    with open(save_path, 'w') as outfile:
        json.dump(output, outfile)

    return


def test_preprocessing(data_path, save_path):

    input_file = pd.read_csv(data_path)
    input_file = np.array(input_file)

    output = {'version': 'v1.0', 'data': []}

    for line in input_file:
        paragraphs = []

        context = line[1]

        qas = []
        question = line[-1]
        qid = line[0]
        if type(context) != str or type(question) != str:
            print(context, type(context))
            print(question, type(question))
            continue
        answers = []
        answers.append({'answer_start': 1000000, 'text': '__None__'})
        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})

        paragraphs.append({'context': context, 'qas': qas})
        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

    with open(save_path, 'w') as outfile:
        json.dump(output, outfile)

    return


def find_all(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


############################################ Define getting data split functions
def get_train_val_split(data_path="/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/",
                        save_path="/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/",
                        n_splits=5,
                        seed=42,
                        split="StratifiedKFold"):

    os.makedirs(save_path + '/split', exist_ok=True)
    df_path = os.path.join(data_path, "train.csv")
    df = pd.read_csv(df_path, encoding='utf8')

    if split == "StratifiedKFold":
        kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True).split(X=df.textID, y=df.sentiment)
    else:
        raise NotImplementedError

    for fold, (train_idx, valid_idx) in enumerate(kf):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[valid_idx]

        df_train.to_csv(save_path + '/split/train_fold_%s_seed_%s.csv' % (fold, seed), index=False)
        df_val.to_csv(save_path + '/split/val_fold_%s_seed_%s.csv' % (fold, seed), index=False)

        train_preprocessing(save_path + '/split/train_fold_%s_seed_%s.csv' % (fold, seed), save_path +
                            '/split/train_fold_%s_seed_%s.json' % (fold, seed))
        train_preprocessing(save_path + '/split/val_fold_%s_seed_%s.csv' % (fold, seed), save_path +
                            '/split/val_fold_%s_seed_%s.json' % (fold, seed))

    return


############################################ Define getting data loader functions
def get_test_loader(data_path="/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/",
                    max_seq_length=384,
                    max_query_length=64,
                    doc_stride=128,
                    threads=1,
                    model_type="bert-base-uncased",
                    batch_size=4,
                    num_workers=4):

    json_path = os.path.join(data_path, "test.json")

    if not os.path.exists(json_path):
        csv_path = os.path.join(data_path, "test.csv")
        test_preprocessing(csv_path, json_path)

    if (model_type == "bert-base-uncased") or (model_type == "bert-large-uncased"):

        tokenizer = BertTokenizer.from_pretrained(model_type, additional_special_tokens=["[UNK]", "[SEP]", "[PAD]",
                                                                                         "[CLS]", "[MASK]"])
    elif (model_type == "bert-base-cased") or (model_type == "bert-large-cased"):

        tokenizer = BertTokenizer.from_pretrained(model_type, additional_special_tokens=["[UNK]", "[SEP]", "[PAD]",
                                                                                         "[CLS]", "[MASK]"])
    elif model_type == "t5-base":

        ADD_TOKEN_LIST = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
        tokenizer = T5Tokenizer.from_pretrained(model_type, additional_special_tokens=ADD_TOKEN_LIST)
        tokenizer.cls_token = '[CLS]'
        tokenizer.sep_token = '[SEP]'

    elif model_type == "flaubert-base-uncased":

        tokenizer = FlaubertTokenizer.from_pretrained(model_type, additional_special_tokens=["[UNK]", "[SEP]", "[PAD]",
                                                                                             "[CLS]", "[MASK]"])
    elif (model_type == "flaubert-base-cased") or (model_type == "flaubert-large-cased"):

        tokenizer = FlaubertTokenizer.from_pretrained(model_type, additional_special_tokens=["[UNK]", "[SEP]", "[PAD]",
                                                                                             "[CLS]", "[MASK]"])
    elif (model_type == "xlnet-base-cased") or (model_type == "xlnet-large-cased"):

        tokenizer = XLNetTokenizer.from_pretrained(model_type, additional_special_tokens=["[UNK]", "[SEP]", "[PAD]",
                                                                                          "[CLS]", "[MASK]"])
    elif model_type == "roberta-base":

        ADD_TOKEN_LIST = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
        tokenizer = RobertaTokenizer.from_pretrained(model_type)
        tokenizer.cls_token = '[CLS]'
        tokenizer.sep_token = '[SEP]'
        num_added_tokens = tokenizer.add_tokens(ADD_TOKEN_LIST)
        print('Number of Tokens Added : ', num_added_tokens)

    elif ((model_type == "albert-base-v2") or (model_type == "albert-large-v2") or (model_type == "albert-xlarge-v2")
          or (model_type == "albert-xxlarge-v2")):

        tokenizer = AlbertTokenizer.from_pretrained(model_type, additional_special_tokens=["[UNK]", "[SEP]", "[PAD]",
                                                                                          "[CLS]", "[MASK]"])
    elif model_type == "gpt2":

        tokenizer = AutoTokenizer.from_pretrained(model_type, additional_special_tokens=["[UNK]", "[SEP]", "[PAD]",
                                                                                          "[CLS]", "[MASK]"])
    else:

        raise NotImplementedError

    ds_test = load_and_cache_examples(data_path, "test.json", model_type, tokenizer, max_seq_length, max_query_length,
                                      doc_stride, threads, mode="test", output_examples=False)
    loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    return loader


def get_train_val_loaders(data_path="/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/",
                          seed=42,
                          fold=0,
                          max_seq_length=384,
                          max_query_length=64,
                          doc_stride=128,
                          threads=1,
                          model_type="bert-base-uncased",
                          batch_size=4,
                          val_batch_size=4,
                          num_workers=2):

    train_json_path = os.path.join(data_path, 'split/train_fold_%s_seed_%s.json' % (fold, seed))
    val_json_path = os.path.join(data_path, 'split/val_fold_%s_seed_%s.json' % (fold, seed))

    if not os.path.exists(train_json_path):
        train_csv_path = os.path.join(data_path, 'split/train_fold_%s_seed_%s.csv' % (fold, seed))
        train_preprocessing(train_csv_path, train_json_path)

    if not os.path.exists(val_json_path):
        val_csv_path = os.path.join(data_path, 'split/val_fold_%s_seed_%s.csv' % (fold, seed))
        train_preprocessing(val_csv_path, val_json_path)

    if (model_type == "bert-base-uncased") or (model_type == "bert-large-uncased"):

        tokenizer = BertTokenizer.from_pretrained(model_type, additional_special_tokens=["[UNK]", "[SEP]", "[PAD]",
                                                                                         "[CLS]", "[MASK]"])
    elif (model_type == "bert-base-cased") or (model_type == "bert-large-cased"):

        tokenizer = BertTokenizer.from_pretrained(model_type, additional_special_tokens=["[UNK]", "[SEP]", "[PAD]",
                                                                                         "[CLS]", "[MASK]"])
    elif model_type == "t5-base":

        ADD_TOKEN_LIST = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
        tokenizer = T5Tokenizer.from_pretrained(model_type, additional_special_tokens=ADD_TOKEN_LIST)
        tokenizer.cls_token = '[CLS]'
        tokenizer.sep_token = '[SEP]'

    elif model_type == "flaubert-base-uncased":

        tokenizer = FlaubertTokenizer.from_pretrained(model_type, additional_special_tokens=["[UNK]", "[SEP]", "[PAD]",
                                                                                             "[CLS]", "[MASK]"])
    elif (model_type == "flaubert-base-cased") or (model_type == "flaubert-large-cased"):

        tokenizer = FlaubertTokenizer.from_pretrained(model_type, additional_special_tokens=["[UNK]", "[SEP]", "[PAD]",
                                                                                             "[CLS]", "[MASK]"])
    elif (model_type == "xlnet-base-cased") or (model_type == "xlnet-large-cased"):

        tokenizer = XLNetTokenizer.from_pretrained(model_type, additional_special_tokens=["[UNK]", "[SEP]", "[PAD]",
                                                                                          "[CLS]", "[MASK]"])
    elif model_type == "roberta-base":

        ADD_TOKEN_LIST = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
        tokenizer = RobertaTokenizer.from_pretrained(model_type)
        tokenizer.cls_token = '[CLS]'
        tokenizer.sep_token = '[SEP]'
        num_added_tokens = tokenizer.add_tokens(ADD_TOKEN_LIST)
        print('Number of Tokens Added : ', num_added_tokens)

    elif ((model_type == "albert-base-v2") or (model_type == "albert-large-v2") or (model_type == "albert-xlarge-v2")
          or (model_type == "albert-xxlarge-v2")):

        tokenizer = AlbertTokenizer.from_pretrained(model_type, additional_special_tokens=["[UNK]", "[SEP]", "[PAD]",
                                                                                          "[CLS]", "[MASK]"])
    elif model_type == "gpt2":

        tokenizer = AutoTokenizer.from_pretrained(model_type, additional_special_tokens=["[UNK]", "[SEP]", "[PAD]",
                                                                                          "[CLS]", "[MASK]"])
    else:

        raise NotImplementedError


    ds_train, examples_train, features_train = load_and_cache_examples(data_path, 'split/train_fold_%s_seed_%s.json' %
                                                                       (fold, seed), model_type, tokenizer,
                                                                       max_seq_length, max_query_length, doc_stride,
                                                                       threads, mode="train", output_examples=True)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               drop_last=True)

    ds_val, examples_val, features_val = load_and_cache_examples(data_path, 'split/val_fold_%s_seed_%s.json' %
                                                                 (fold, seed), model_type, tokenizer, max_seq_length,
                                                                 max_query_length, doc_stride, threads, mode="train",
                                                                 output_examples=True)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=val_batch_size, shuffle=True, num_workers=num_workers,
                                             drop_last=True)

    return train_loader, examples_train, features_train, val_loader, examples_val, features_val, tokenizer


############################################ Define test function
def test_train_val_split(data_path,
                         save_path,
                         n_splits,
                         seed):
    print("------------------------testing train test splitting----------------------")
    print("data_path: ", data_path)
    print("save_path: ", save_path)
    print("n_splits: ", n_splits)
    print("seed: ", seed)

    get_train_val_split(data_path, save_path, n_splits, seed)

    print("generating successfully, please check results !")

    return


def test_test_loader(data_path="/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/",
                     max_seq_length=384,
                     max_query_length=64,
                     doc_stride=128,
                     threads=1,
                     model_type="bert-base-uncased",
                     batch_size=4,
                     num_workers=4):

    loader = get_test_loader(data_path=data_path, max_seq_length=max_seq_length, max_query_length=max_query_length,
                             doc_stride=doc_stride, threads=threads, model_type=model_type, batch_size=batch_size,
                             num_workers=num_workers)

    for all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask in loader:
        print("------------------------testing test loader----------------------")
        print("all_input_ids (numpy): ", all_input_ids.numpy().shape)
        print("all_attention_masks (numpy): ", all_attention_masks.numpy().shape)
        print("all_token_type_ids (numpy): ", all_token_type_ids.numpy().shape)
        print("all_example_index (numpy): ", all_example_index.numpy().shape)
        print("all_cls_index (numpy): ", all_cls_index.numpy().shape)
        print("all_p_mask (numpy): ", all_p_mask.numpy().shape)
        print("------------------------testing test loader finished----------------------")
        break


def test_train_loader(data_path="/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/",
                      seed=42,
                      fold=0,
                      max_seq_length=384,
                      max_query_length=64,
                      doc_stride=128,
                      threads=1,
                      model_type="bert-base-uncased",
                      batch_size=4,
                      val_batch_size=4,
                      num_workers=2):

    train_loader, examples_train, features_train, val_loader, examples_val, features_val, tokenizer = \
                                            get_train_val_loaders(data_path=data_path, seed=seed, fold=fold,
                                                                max_seq_length=max_seq_length,
                                                                max_query_length=max_query_length,
                                                                doc_stride=doc_stride, threads=threads,
                                                                model_type=model_type, batch_size=batch_size,
                                                                val_batch_size=val_batch_size, num_workers=num_workers)

    for all_input_ids, all_attention_masks, all_token_type_ids, all_start_positions, all_end_positions, all_cls_index, \
        all_p_mask, all_is_impossible in train_loader:

        print("------------------------testing train loader----------------------")
        print("all_input_ids (numpy): ", all_input_ids.numpy().shape)
        print("all_attention_masks (numpy): ", all_attention_masks.numpy().shape)
        print("all_token_type_ids (numpy): ", all_token_type_ids.numpy().shape)
        print("all_start_positions (numpy): ", all_start_positions.numpy().shape)
        print("all_end_positions (numpy): ", all_end_positions.numpy().shape)
        print("all_cls_index (numpy): ", all_cls_index.numpy().shape)
        print("all_p_mask (numpy): ", all_p_mask.numpy().shape)
        print("all_is_impossible (numpy): ", all_is_impossible.numpy().shape)
        print("------------------------testing train loader finished----------------------")
        break

    for all_input_ids, all_attention_masks, all_token_type_ids, all_start_positions, all_end_positions, all_cls_index, \
        all_p_mask, all_is_impossible in val_loader:
        print("------------------------testing train loader----------------------")
        print("all_input_ids (numpy): ", all_input_ids.numpy().shape)
        print("all_attention_masks (numpy): ", all_attention_masks.numpy().shape)
        print("all_token_type_ids (numpy): ", all_token_type_ids.numpy().shape)
        print("all_start_positions (numpy): ", all_start_positions.numpy().shape)
        print("all_end_positions (numpy): ", all_end_positions.numpy().shape)
        print("all_cls_index (numpy): ", all_cls_index.numpy().shape)
        print("all_p_mask (numpy): ", all_p_mask.numpy().shape)
        print("all_is_impossible (numpy): ", all_is_impossible.numpy().shape)
        print("------------------------testing train loader finished----------------------")
        break


if __name__ == "__main__":
    args = parser.parse_args()

    # test getting train val splitting
    test_train_val_split(args.data_path,
                         args.save_path,
                         args.n_splits,
                         args.seed)

    # test test_data_loader
    test_test_loader(data_path=args.data_path,
                     max_seq_length=args.max_seq_length,
                     max_query_length=args.max_query_length,
                     doc_stride=args.doc_stride,
                     threads=args.threads,
                     model_type=args.model_type,
                     batch_size=args.val_batch_size,
                     num_workers=args.num_workers)

    # test train_data_loader
    test_train_loader(data_path=args.data_path,
                      seed=args.seed,
                      fold=args.fold,
                      max_seq_length=args.max_seq_length,
                      max_query_length=args.max_query_length,
                      doc_stride=args.doc_stride,
                      threads=args.threads,
                      model_type=args.model_type,
                      batch_size=args.batch_size,
                      val_batch_size=args.val_batch_size,
                      num_workers=args.num_workers)
