import argparse
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import *
from transformers.data.processors.squad import *
from sklearn.model_selection import StratifiedKFold, KFold

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


############################################ Modify Helper functions and dataset function from transformers run_swag.py
def squad_convert_examples_to_features_v2(
    examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, return_dataset=False, threads=1
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.
    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi
    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`
    Example::
        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)
        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
            )
        )
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

        if not is_training:
            print("test")
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask
            )
        else:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_example_index,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
            )

        return features, dataset

    return features

def load_and_cache_examples_v2(data_dir, input_file, model_type, tokenizer, max_seq_length=384, max_query_length=64,
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
    elif mode == "val":
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

    # overwrite cache for multiple tokenizers
    print("Creating features from dataset file at %s", input_file)

    processor = SquadV2Processor()

    if evaluate:
        examples = processor.get_dev_examples(data_dir, filename=input_file)
    else:
        examples = processor.get_train_examples(data_dir, filename=input_file)

    features, dataset = squad_convert_examples_to_features_v2(
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
    elif split == "KFold":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df.text)
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

    if (model_type == "bert-base-uncased"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=False,
        )
    elif (model_type == "bert-large-uncased"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=False,
        )
    elif (model_type == "bert-base-cased"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=True,
        )
    elif (model_type == "bert-large-cased"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=True,
        )
    elif model_type == "t5-base":
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=False,
        )
    elif (model_type == "xlnet-base-cased") or (model_type == "xlnet-large-cased"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=True,
        )
    elif model_type == "roberta-base":
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=True,
        )
        tokenizer.cls_token = '[CLS]'
        tokenizer.sep_token = '[SEP]'
    elif model_type == "roberta-large":
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=True,
        )
        tokenizer.cls_token = '[CLS]'
        tokenizer.sep_token = '[SEP]'
    else:

        raise NotImplementedError

    ds_test, examples_test, features_test = load_and_cache_examples_v2(data_path, "test.json", model_type, tokenizer,
                                                                    max_seq_length, max_query_length, doc_stride,
                                                                    threads, mode="test", output_examples=True)
    # print(len(ds_test.tensors))
    loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return loader, examples_test, features_test


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

    train_preprocessing(os.path.join(data_path, 'train.csv'), os.path.join(data_path, 'train.json'))
    train_json_path = os.path.join(data_path, 'split/train_fold_%s_seed_%s.json' % (fold, seed))
    val_json_path = os.path.join(data_path, 'split/val_fold_%s_seed_%s.json' % (fold, seed))

    if not os.path.exists(train_json_path):
        train_csv_path = os.path.join(data_path, 'split/train_fold_%s_seed_%s.csv' % (fold, seed))
        train_preprocessing(train_csv_path, train_json_path)

    if not os.path.exists(val_json_path):
        val_csv_path = os.path.join(data_path, 'split/val_fold_%s_seed_%s.csv' % (fold, seed))
        train_preprocessing(val_csv_path, val_json_path)

    if (model_type == "bert-base-uncased"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=False,
        )
    elif (model_type == "bert-large-uncased"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=False,
        )
    elif (model_type == "bert-base-cased"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=True,
        )
    elif (model_type == "bert-large-cased"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=True,
        )
    elif model_type == "t5-base":
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=False,
        )
    elif (model_type == "xlnet-base-cased") or (model_type == "xlnet-large-cased"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=True,
        )
    elif model_type == "roberta-base":
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=True,
        )
        tokenizer.cls_token = '[CLS]'
        tokenizer.sep_token = '[SEP]'
    elif model_type == "roberta-large":
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=True,
        )
        tokenizer.cls_token = '[CLS]'
        tokenizer.sep_token = '[SEP]'
    else:

        raise NotImplementedError


    ds_train, examples_train, features_train = load_and_cache_examples_v2(data_path, 'split/train_fold_%s_seed_%s.json' %
                                                                       (fold, seed), model_type, tokenizer,
                                                                       max_seq_length, max_query_length, doc_stride,
                                                                       threads, mode="train", seed=seed, fold=fold,
                                                                          output_examples=True)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               drop_last=True)

    ds_val, examples_val, features_val = load_and_cache_examples_v2(data_path, 'split/val_fold_%s_seed_%s.json' %
                                                                 (fold, seed), model_type, tokenizer, max_seq_length,
                                                                 max_query_length, doc_stride, threads, mode="val",
                                                                 seed=seed, fold=fold, output_examples=True)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
                                             drop_last=False)

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

    test_loader, examples_test, features_test = get_test_loader(data_path=data_path, max_seq_length=max_seq_length, max_query_length=max_query_length,
                             doc_stride=doc_stride, threads=threads, model_type=model_type, batch_size=batch_size,
                             num_workers=num_workers)

    for _, (all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask) in enumerate(test_loader):
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

    for _, (all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_start_positions, all_end_positions, all_cls_index, \
        all_p_mask, all_is_impossible) in enumerate(train_loader):

        print("------------------------testing train loader----------------------")
        print("all_input_ids (numpy): ", all_input_ids.numpy().shape)
        print("all_attention_masks (numpy): ", all_attention_masks.numpy().shape)
        print("all_token_type_ids (numpy): ", all_token_type_ids.numpy().shape)
        print("all_example_index (numpy): ", all_example_index.numpy().shape)
        print("all_start_positions (numpy): ", all_start_positions.numpy().shape)
        print("all_end_positions (numpy): ", all_end_positions.numpy().shape)
        print("all_cls_index (numpy): ", all_cls_index.numpy().shape)
        print("all_p_mask (numpy): ", all_p_mask.numpy().shape)
        print("all_is_impossible (numpy): ", all_is_impossible.numpy().shape)
        print("------------------------testing train loader finished----------------------")
        break

    for _, (all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_start_positions, all_end_positions, all_cls_index, \
        all_p_mask, all_is_impossible) in enumerate(val_loader):
        print("------------------------testing train loader----------------------")
        print("all_input_ids (numpy): ", all_input_ids.numpy().shape)
        print("all_attention_masks (numpy): ", all_attention_masks.numpy().shape)
        print("all_token_type_ids (numpy): ", all_token_type_ids.numpy().shape)
        print("all_example_index (numpy): ", all_example_index.numpy().shape)
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
