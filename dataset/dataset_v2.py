import argparse
import re
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import *
from transformers.data.processors.squad import *
import tokenizers
from sklearn.model_selection import StratifiedKFold, KFold

import nlpaug.augmenter.word as naw
import nlpaug.flow as naf

from .Datasampler import *

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
parser.add_argument('--model_type', type=str, default="roberta-base", required=False,
                    help='specify the model_type for BertTokenizer')


############################################ Define process data functions
def augmentation(text, insert=False, substitute=False, swap=True, delete=True):
    augs = []

    if insert:
        aug = naw.WordEmbsAug(
            model_type='word2vec',
            model_path='/media/jionie/my_disk/Kaggle/Tweet/model/word2vec/GoogleNews-vectors-negative300.bin',
            action="insert")
        augs.append(aug)

    if substitute:
        aug_sub = naw.SynonymAug(aug_src='wordnet')
        augs.append(aug_sub)

    if swap:
        aug_swap = naw.RandomWordAug(action="swap")
        augs.append(aug_swap)

    if delete:
        aug_del = naw.RandomWordAug()
        augs.append(aug_del)

    aug = naf.Sometimes(augs, aug_p=0.5, pipeline_p=0.5)
    # print("before aug:", text)
    text = aug.augment(text, n=1)
    # print("after aug:", text)

    return text


def process_data(tweet, selected_text, sentiment, tokenizer, model_type, max_len, augment=False):

    # lower cased here
    tweet = " ".join(str(tweet).split())
    selected_text = " ".join(str(selected_text).split())

    tweet = " " + tweet.lower()
    selected_text = " " + selected_text.lower()

    if len(tweet) == len(selected_text):
        ans_type = "long"
    elif len(selected_text) == 0:
        ans_type = "none"
    else:
        ans_type = "short"

    # remove first " "
    len_st = len(selected_text) - 1
    idx0 = None
    idx1 = None

    # get char idx
    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
        if " " + tweet[ind: ind + len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    if augment:
        # augment for non-select text
        start_str = tweet[:idx0]
        end_str = tweet[idx1 + 1:]

        if (len(start_str) > 0):
            if np.random.uniform(0, 1, 1) < 0.1:
                start_str = augmentation(start_str, insert=False, substitute=False, swap=False, delete=True)
        if (len(end_str) > 0):
            if np.random.uniform(0, 1, 1) < 0.1:
                end_str = augmentation(end_str, insert=False, substitute=False, swap=False, delete=True)

        tweet = start_str + selected_text + end_str

        # after augment we need to search again
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

    # get char mask
    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1

    # get word offsets
    tweet_offsets_word_level = []
    tweet_offsets_token_level = []
    cursor = 0
    input_ids_orig = []

    for i, word in enumerate(tweet.split()):

        sub_words = tokenizer.tokenize(" " + word)
        encoded_word = tokenizer.convert_tokens_to_ids(sub_words)
        number_of_tokens = len(encoded_word)
        input_ids_orig += encoded_word

        start_offsets = cursor
        cursor += len(" " + word)
        end_offsets = cursor

        token_level_cursor = start_offsets

        for i in range(number_of_tokens):

            tweet_offsets_word_level.append((start_offsets, end_offsets))

            if (model_type == "bert-base-uncased") or (model_type == "bert-large-uncased") \
                or (model_type == "bert-base-cased") or (model_type == "bert-large-cased"):
                # for bert tokenizer, replace "##" and add " " for first sub_word
                sub_word_len = len(sub_words[i].replace("##", ""))
                if i == 0:
                    sub_word_len += 1
            else:
                sub_word_len = len(sub_words[i])

            tweet_offsets_token_level.append((token_level_cursor, token_level_cursor + sub_word_len))
            token_level_cursor += sub_word_len

    # get word idx
    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets_token_level):

        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)

    if len(target_idx) == 0:
        print(tweet, selected_text)

    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    # print(tweet[tweet_offsets[targets_start][0] : tweet_offsets[targets_end][1]], "------------", selected_text)
    last_offset = tweet_offsets_word_level[-1][1]
    if model_type == "roberta-base" or model_type == "roberta-large" or model_type == "roberta-base-squad":

        sentiment_id = {
            'positive': 1313,
            'negative': 2430,
            'neutral': 7974
        }

        input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
        token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets_token_level = [(0, last_offset)] * 4 + tweet_offsets_token_level + [(0, last_offset)]
        tweet_offsets_word_level = [(0, last_offset)] * 4 + tweet_offsets_word_level + [(0, last_offset)]

    elif (model_type == "albert-base-v2") or (model_type == "albert-large-v2") or (model_type == "albert-xlarge-v2"):

        sentiment_id = {
            'positive': 2221,
            'negative': 3682,
            'neutral': 8387
        }

        input_ids = [2] + [sentiment_id[sentiment]] + [3] + input_ids_orig + [3]
        token_type_ids = [0, 0, 0] + [0] * (len(input_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets_token_level = [(0, last_offset)] * 3 + tweet_offsets_token_level + [(0, last_offset)]
        tweet_offsets_word_level = [(0, last_offset)] * 3 + tweet_offsets_word_level + [(0, last_offset)]

    elif (model_type == "xlnet-base-cased") or (model_type == "xlnet-large-cased"):

        sentiment_id = {
            'positive': 1654,
            'negative': 2981,
            'neutral': 9201
        }

        input_ids = [sentiment_id[sentiment]] + [4] + input_ids_orig + [3]
        token_type_ids = [0, 0] + [0] * (len(input_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets_token_level = [(0, last_offset)] * 2 + tweet_offsets_token_level + [(0, last_offset)]
        tweet_offsets_word_level = [(0, last_offset)] * 2 + tweet_offsets_word_level + [(0, last_offset)]

    elif (model_type == "bert-base-uncased") or (model_type == "bert-large-uncased"):

        sentiment_id = {
            'positive': 3893,
            'negative': 4997,
            'neutral': 8699
        }

        input_ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids_orig + [102]
        token_type_ids = [0, 0, 0] + [0] * (len(input_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets_token_level = [(0, last_offset)] * 3 + tweet_offsets_token_level + [(0, last_offset)]
        tweet_offsets_word_level = [(0, last_offset)] * 3 + tweet_offsets_word_level + [(0, last_offset)]

    elif (model_type == "bert-base-cased") or (model_type == "bert-large-cased"):

        sentiment_id = {
            'positive': 3112,
            'negative': 4366,
            'neutral': 8795
        }

        input_ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids_orig + [102]
        token_type_ids = [0, 0, 0] + [0] * (len(input_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets_token_level = [(0, last_offset)] * 3 + tweet_offsets_token_level + [(0, last_offset)]
        tweet_offsets_word_level = [(0, last_offset)] * 3 + tweet_offsets_word_level + [(0, last_offset)]

    else:
        raise NotImplementedError

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets_token_level = tweet_offsets_token_level + ([(0, last_offset)] * padding_length)
        tweet_offsets_word_level = tweet_offsets_word_level + ([(0, last_offset)] * padding_length)
    else:
        input_ids = input_ids[:max_len]
        mask = mask[:max_len]
        token_type_ids = token_type_ids[:max_len]
        tweet_offsets_token_level = tweet_offsets_token_level[:max_len]
        tweet_offsets_word_level = tweet_offsets_word_level[:max_len]

    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'ans_type': ans_type,
        'offsets_token_level': tweet_offsets_token_level,
        'offsets_word_level': tweet_offsets_word_level
    }


############################################ Define Tweet Dataset class
class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text, tokenizer, model_type, max_len, augment=False):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):

        data = process_data(
            self.tweet[item],
            self.selected_text[item],
            self.sentiment[item],
            self.tokenizer,
            self.model_type,
            self.max_len,
            self.augment
        )

        onthot_ans_type = {
            'long': torch.tensor([1, 0, 0], dtype=torch.float),
            'none': torch.tensor([0, 1, 0], dtype=torch.float),
            'short': torch.tensor([0, 0, 1], dtype=torch.float),
        }

        return torch.tensor(data["ids"], dtype=torch.long), \
               torch.tensor(data["mask"], dtype=torch.long), \
               torch.tensor(data["token_type_ids"], dtype=torch.long), \
               torch.tensor(data["targets_start"], dtype=torch.long),\
               torch.tensor(data["targets_end"], dtype=torch.long),\
               onthot_ans_type[data["ans_type"]], \
               data["orig_tweet"], \
               data["orig_selected"], \
               data["ans_type"], \
               data["sentiment"], \
               torch.tensor(data["offsets_token_level"], dtype=torch.long), \
               torch.tensor(data["offsets_word_level"], dtype=torch.long)



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
        kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True).split(X=df.text, y=df.sentiment)
    elif split == "KFold":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df.text)
    else:
        raise NotImplementedError

    for fold, (train_idx, valid_idx) in enumerate(kf):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[valid_idx]

        df_train.to_csv(save_path + '/split/train_fold_%s_seed_%s.csv' % (fold, seed), index=False)
        df_val.to_csv(save_path + '/split/val_fold_%s_seed_%s.csv' % (fold, seed), index=False)

    return


############################################ Define getting data loader functions
def get_test_loader(data_path="/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/",
                    max_seq_length=384,
                    model_type="bert-base-uncased",
                    batch_size=4,
                    num_workers=4):

    CURR_PATH = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(data_path, "test.csv")
    df_test = pd.read_csv(csv_path)
    df_test.loc[:, "selected_text"] = df_test.text.values

    if (model_type == "bert-base-uncased"):
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers_vocab/{}-vocab.txt".format(model_type)),
            lowercase=True,
        )
    elif (model_type == "bert-large-uncased"):
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers_vocab/{}-vocab.txt".format(model_type)),
            lowercase=True,
        )
    elif (model_type == "bert-base-cased"):
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers_vocab/{}-vocab.txt".format(model_type)),
            lowercase=True,
        )
    elif (model_type == "bert-large-cased"):
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers_vocab/{}-vocab.txt".format(model_type)),
            lowercase=True,
        )
    elif (model_type == "xlnet-base-cased") or (model_type == "xlnet-large-cased"):
        tokenizer = XLNetTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers_vocab/{}-spiece.model".format(model_type)),
            lowercase=True,
        )
    elif (model_type == "albert-base-v2") or (model_type == "albert-large-v2") or (model_type == "albert-xlarge-v2"):
        tokenizer = AlbertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers_vocab/{}-spiece.model".format(model_type)),
            lowercase=True,
        )
    elif model_type == "roberta-base":
        tokenizer = RobertaTokenizer(
            vocab_file=os.path.join(CURR_PATH, "transformers_vocab/{}-vocab.json".format(model_type)),
            merges_file=os.path.join(CURR_PATH, "transformers_vocab/{}-merges.txt".format(model_type)),
            lowercase=True,
        )
    elif model_type == "roberta-base-squad":
        tokenizer = RobertaTokenizer(
            vocab_file=os.path.join(CURR_PATH, "transformers_vocab/{}-vocab.json".format(model_type)),
            merges_file=os.path.join(CURR_PATH, "transformers_vocab/{}-merges.txt".format(model_type)),
            lowercase=True,
        )
    elif model_type == "roberta-large":
        tokenizer = RobertaTokenizer(
            vocab_file=os.path.join(CURR_PATH, "transformers_vocab/{}-vocab.json".format(model_type)),
            merges_file=os.path.join(CURR_PATH, "transformers_vocab/{}-merges.txt".format(model_type)),
            lowercase=True,
        )
    else:

        raise NotImplementedError

    ds_test = TweetDataset(
        tweet=df_test.text.values,
        sentiment=df_test.sentiment.values,
        selected_text=df_test.selected_text.values,
        tokenizer=tokenizer,
        model_type=model_type,
        max_len=max_seq_length
    )
    # print(len(ds_test.tensors))
    loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return loader, tokenizer


def get_train_val_loaders(data_path="/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/",
                          seed=42,
                          fold=0,
                          max_seq_length=384,
                          model_type="bert-base-uncased",
                          batch_size=4,
                          val_batch_size=4,
                          num_workers=2,
                          Datasampler="ImbalancedDatasetSampler"):

    CURR_PATH = os.path.dirname(os.path.realpath(__file__))
    train_csv_path = os.path.join(data_path, 'split/train_fold_%s_seed_%s.csv' % (fold, seed))
    val_csv_path = os.path.join(data_path, 'split/val_fold_%s_seed_%s.csv' % (fold, seed))

    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)

    if (model_type == "bert-base-uncased"):
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers_vocab/{}-vocab.txt".format(model_type)),
            lowercase=True,
        )
    elif (model_type == "bert-large-uncased"):
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers_vocab/{}-vocab.txt".format(model_type)),
            lowercase=True,
        )
    elif (model_type == "bert-base-cased"):
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers_vocab/{}-vocab.txt".format(model_type)),
            lowercase=True,
        )
    elif (model_type == "bert-large-cased"):
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers_vocab/{}-vocab.txt".format(model_type)),
            lowercase=True,
        )
    elif (model_type == "xlnet-base-cased") or (model_type == "xlnet-large-cased"):
        tokenizer = XLNetTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH,
                                                       "transformers_vocab/{}-spiece.model".format(model_type)),
            lowercase=True,
        )
    elif (model_type == "albert-base-v2") or (model_type == "albert-large-v2") or (model_type == "albert-xlarge-v2"):
        tokenizer = AlbertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH,
                                                       "transformers_vocab/{}-spiece.model".format(model_type)),
            lowercase=True,
        )
    elif model_type == "roberta-base":
        tokenizer = RobertaTokenizer(
            vocab_file=os.path.join(CURR_PATH, "transformers_vocab/{}-vocab.json".format(model_type)),
            merges_file=os.path.join(CURR_PATH, "transformers_vocab/{}-merges.txt".format(model_type)),
            lowercase=True,
        )
    elif model_type == "roberta-base-squad":
        tokenizer = RobertaTokenizer(
            vocab_file=os.path.join(CURR_PATH, "transformers_vocab/{}-vocab.json".format(model_type)),
            merges_file=os.path.join(CURR_PATH, "transformers_vocab/{}-merges.txt".format(model_type)),
            lowercase=True,
        )
    elif model_type == "roberta-large":
        tokenizer = RobertaTokenizer(
            vocab_file=os.path.join(CURR_PATH, "transformers_vocab/{}-vocab.json".format(model_type)),
            merges_file=os.path.join(CURR_PATH, "transformers_vocab/{}-merges.txt".format(model_type)),
            lowercase=True,
        )
    else:

        raise NotImplementedError

    ds_train = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values,
        tokenizer=tokenizer,
        model_type=model_type,
        max_len=max_seq_length,
        augment=False
    )

    if Datasampler == "None":
        train_loader = torch.utils.data.DataLoader(ds_train,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   drop_last=True)
    elif Datasampler == "ImbalancedDatasetSampler":
        train_loader = torch.utils.data.DataLoader(ds_train,
                                                   batch_size=batch_size,
                                                   sampler=ImbalancedDatasetSampler(ds_train),
                                                   num_workers=num_workers,
                                                   drop_last=True)
    else:
        raise NotImplementedError

    ds_val = TweetDataset(
        tweet=df_val.text.values,
        sentiment=df_val.sentiment.values,
        selected_text=df_val.selected_text.values,
        tokenizer=tokenizer,
        model_type=model_type,
        max_len=max_seq_length
    )
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
                                             drop_last=False)

    return train_loader, val_loader, tokenizer


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
                     model_type="bert-base-uncased",
                     batch_size=4,
                     num_workers=4):

    test_loader, _ = get_test_loader(data_path=data_path, max_seq_length=max_seq_length, model_type=model_type,
                                  batch_size=batch_size, num_workers=num_workers)

    for _, (all_input_ids, all_attention_masks, all_token_type_ids, all_start_positions,
            all_end_positions, all_onthot_ans_type, all_orig_tweet, all_orig_selected, all_tweet, all_sentiment,
            all_offsets_token_level, all_offsets_word_level) in enumerate(test_loader):

        print("------------------------testing test loader----------------------")
        print("all_input_ids (numpy): ", all_input_ids.numpy().shape)
        print("all_attention_masks (numpy): ", all_attention_masks.numpy().shape)
        print("all_token_type_ids (numpy): ", all_token_type_ids.numpy().shape)
        print("all_start_positions (numpy): ", all_start_positions.numpy().shape)
        print("all_end_positions (numpy): ", all_end_positions.numpy().shape)
        print("all_orig_tweet (string): ", all_orig_tweet)
        print("all_orig_selected (string): ", all_orig_selected)
        print("all_tweet (string after processing): ", all_tweet)
        print("all_sentiment (string): ", all_sentiment)
        print("all_offsets (numpy): ", all_offsets_token_level.numpy().shape)
        print("------------------------testing test loader finished----------------------")
        break


def test_train_loader(data_path="/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/",
                      seed=42,
                      fold=0,
                      max_seq_length=384,
                      model_type="bert-base-uncased",
                      batch_size=4,
                      val_batch_size=4,
                      num_workers=2):

    train_loader, val_loader, _ = get_train_val_loaders(data_path=data_path, seed=seed, fold=fold,
                                                     max_seq_length=max_seq_length, model_type=model_type,
                                                     batch_size=batch_size, val_batch_size=val_batch_size,
                                                     num_workers=num_workers)

    for _, (all_input_ids, all_attention_masks, all_token_type_ids, all_start_positions, all_end_positions,
                    all_onthot_ans_type, all_orig_tweet, all_orig_selected, all_tweet, all_sentiment,
            all_offsets_token_level, all_offsets_word_level) in enumerate(train_loader):

        print("------------------------testing train loader----------------------")
        print("all_input_ids (numpy): ", all_input_ids.numpy().shape)
        print("all_attention_masks (numpy): ", all_attention_masks.numpy().shape)
        print("all_token_type_ids (numpy): ", all_token_type_ids.numpy().shape)
        print("all_start_positions (numpy): ", all_start_positions.numpy().shape)
        print("all_end_positions (numpy): ", all_end_positions.numpy().shape)
        print("all_orig_tweet (string): ", all_orig_tweet)
        print("all_orig_selected (string): ", all_orig_selected)
        print("all_tweet (string after processing): ", all_tweet)
        print("all_sentiment (string): ", all_sentiment)
        print("all_offsets (numpy): ", all_offsets_token_level.numpy().shape)
        print("------------------------testing train loader finished----------------------")
        break

    for _, (all_input_ids, all_attention_masks, all_token_type_ids, all_start_positions, all_end_positions,
                    all_onthot_ans_type, all_orig_tweet, all_orig_selected, all_tweet, all_sentiment,
            all_offsets_token_level, all_offsets_word_level) in enumerate(val_loader):

        print("------------------------testing val loader----------------------")
        print("all_input_ids (numpy): ", all_input_ids.numpy().shape)
        print("all_attention_masks (numpy): ", all_attention_masks.numpy().shape)
        print("all_token_type_ids (numpy): ", all_token_type_ids.numpy().shape)
        print("all_start_positions (numpy): ", all_start_positions.numpy().shape)
        print("all_end_positions (numpy): ", all_end_positions.numpy().shape)
        print("all_orig_tweet (string): ", all_orig_tweet)
        print("all_orig_selected (string): ", all_orig_selected)
        print("all_tweet (string after processing): ", all_tweet)
        print("all_sentiment (string): ", all_sentiment)
        print("all_offsets (numpy): ", all_offsets_token_level.numpy().shape)
        print("------------------------testing val loader finished----------------------")
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
                     model_type=args.model_type,
                     batch_size=args.val_batch_size,
                     num_workers=args.num_workers)

    # test train_data_loader
    test_train_loader(data_path=args.data_path,
                      seed=args.seed,
                      fold=args.fold,
                      max_seq_length=args.max_seq_length,
                      model_type=args.model_type,
                      batch_size=args.batch_size,
                      val_batch_size=args.val_batch_size,
                      num_workers=args.num_workers)
