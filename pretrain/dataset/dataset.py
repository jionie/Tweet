import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from transformers import *
from transformers.data.processors.squad import *
import tokenizers
from sklearn.model_selection import StratifiedKFold

import nlpaug.augmenter.word as naw
import nlpaug.flow as naf


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


def process_data(tweet, sentiment, tokenizer, max_len, augment):

    if augment:
        tweet = augmentation(tweet, insert=False, substitute=True, swap=False, delete=True)

    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet

    input_ids = [0] + input_ids_orig + [2]
    token_type_ids = [0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'sentiment': sentiment,
    }


############################################ Define Tweet Dataset class
class TweetDataset:
    def __init__(self, tweet, sentiment, tokenizer, max_len, augment=False):
        self.tweet = tweet
        self.sentiment = sentiment
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data(
            self.tweet[item],
            self.sentiment[item],
            self.tokenizer,
            self.max_len,
            self.augment
        )

        onthot_sentiments_type = {
            'neutral': torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float),
            'worry': torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float),
            'happiness': torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float),
            'sadness': torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float),
            'love': torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float),
            'surprise': torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float),
            'fun': torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float),
            'relief': torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float),
            'hate': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float),
            'empty': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float),
            'enthusiasm': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float),
            'boredom': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float),
            'anger': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float),
        }

        return torch.tensor(data["ids"], dtype=torch.long), \
               torch.tensor(data["mask"], dtype=torch.long), \
               torch.tensor(data["token_type_ids"], dtype=torch.long), \
               onthot_sentiments_type[data["sentiment"]]


def get_train_val_loaders(
                          seed=42,
                          max_seq_length=192,
                          model_type="roberta-base",
                          batch_size=4,
                          val_batch_size=4,
                          num_workers=2):

    CURR_PATH = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv("/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/tweet_dataset.csv")
    df = df.dropna()

    kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True).split(X=df.text, y=df.sentiment)

    df_train = None
    df_val = None
    for fold, (train_idx, valid_idx) in enumerate(kf):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[valid_idx]
        break

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
    elif model_type == "roberta-large":
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case=True,
        )
    else:

        raise NotImplementedError

    ds_train = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        tokenizer=tokenizer,
        max_len=max_seq_length,
        augment=False
    )


    train_loader = torch.utils.data.DataLoader(ds_train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               drop_last=True)

    ds_val = TweetDataset(
        tweet=df_val.text.values,
        sentiment=df_val.sentiment.values,
        tokenizer=tokenizer,
        max_len=max_seq_length
    )
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
                                             drop_last=False)

    return train_loader, val_loader