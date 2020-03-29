import argparse
import os
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, models, transforms
from transformers import *
from sklearn.utils import shuffle
import random
import html
from math import floor, ceil
from sklearn.model_selection import GroupKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.char as nac
import nlpaug.flow as naf
from sklearn.preprocessing import LabelBinarizer


############################################ Define augments for test

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('-data_path', type=str, default="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/train_augment_final_with_clean.csv", \
    required=False, help='specify the path for train.csv')
parser.add_argument('-test_data_path', type=str, default="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/test.csv", \
    required=False, help='specify the path for test.csv')
parser.add_argument('-content', type=str, default="Question", \
    required=False, help='specify the content for token')
parser.add_argument('--n_splits', type=int, default=5, \
    required=False, help='specify the number of folds')
parser.add_argument('--seed', type=int, default=42, \
    required=False, help='specify the random seed for splitting dataset')
parser.add_argument('--save_path', type=str, default="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/", \
    required=False, help='specify the path for saving splitted csv')
parser.add_argument('--test_fold', type=int, default=0, \
    required=False, help='specify the test fold for testing dataloader')
parser.add_argument('--batch_size', type=int, default=4, \
    required=False, help='specify the batch_size for testing dataloader')
parser.add_argument('--val_batch_size', type=int, default=4, \
    required=False, help='specify the val_batch_size for testing dataloader')
parser.add_argument('--num_workers', type=int, default=0, \
    required=False, help='specify the num_workers for testing dataloader')
parser.add_argument('--model_type', type=str, default="bert-base-uncased", \
    required=False, help='specify the model_type for BertTokenizer')


############################################ Define Dataset Contants
SEP_TOKEN_ID = 102

QUESTION_TARGET_COLUMNS = ['question_asker_intent_understanding',
                'question_body_critical',
                'question_conversational',
                'question_expect_short_answer',
                'question_fact_seeking',
                'question_has_commonly_accepted_answer',
                'question_interestingness_others',
                'question_interestingness_self',
                'question_multi_intent',
                'question_not_really_a_question',
                'question_opinion_seeking',
                'question_type_choice',
                'question_type_compare',
                'question_type_consequence',
                'question_type_definition',
                'question_type_entity',
                'question_type_instructions',
                'question_type_procedure',
                'question_type_reason_explanation',
                'question_type_spelling',
                'question_well_written',
                ]

ANSWER_TARGET_COLUMNS = [
                'answer_helpful',
                'answer_level_of_information',
                'answer_plausible',
                'answer_relevance',
                'answer_satisfaction',
                'answer_type_instructions',
                'answer_type_procedure',
                'answer_type_reason_explanation',
                'answer_well_written']

TARGET_COLUMNS = QUESTION_TARGET_COLUMNS + ANSWER_TARGET_COLUMNS


############################################ Define Dataset 

class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, df, host_encoder=None, category_encoder=None, max_len=512, \
                model_type="xlnet-base-uncased", \
                content="Question", \
                train_mode=True, labeled=True, \
                augment=True, extra_token=False):
        self.df = df
        self.max_len = max_len
        self.model_type = model_type
        self.train_mode = train_mode
        self.labeled = labeled
        self.content = content
        
        # print(self.model_type)
        
        if ((self.model_type == "bert-base-uncased") or (self.model_type == "bert-large-uncased")):
            
            self.tokenizer = BertTokenizer.from_pretrained(model_type,\
                additional_special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
            
        elif((self.model_type == "bert-base-cased") or (self.model_type == "bert-large-cased")):
            
            self.tokenizer = BertTokenizer.from_pretrained(model_type,\
                additional_special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
            
        elif((self.model_type == "t5-base")):
            
            ADD_TOKEN_LIST = ['[TITLE]', '[BODY]','[CATEGORY]', '[DOMAIN]', '[HOST]']
            self.tokenizer = T5Tokenizer.from_pretrained(model_type, additional_special_tokens=ADD_TOKEN_LIST)
            self.tokenizer.cls_token = '[CLS]'
            self.tokenizer.sep_token = '[SEP]'
            
        elif((self.model_type == "flaubert-base-uncased")):
            
            self.tokenizer = FlaubertTokenizer.from_pretrained(model_type,\
                additional_special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
            
        elif((self.model_type == "flaubert-base-cased") or (self.model_type == "flaubert-large-cased")):
            
            self.tokenizer = FlaubertTokenizer.from_pretrained(model_type,\
                additional_special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
            
        elif ((self.model_type == "xlnet-base-cased") \
            or (self.model_type == "xlnet-large-cased")):
            
            self.tokenizer = XLNetTokenizer.from_pretrained(model_type, \
                additional_special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
            
        elif((self.model_type == "roberta-base")):
            
            ADD_TOKEN_LIST = ['[TITLE]', '[BODY]', '[ANSWER]', '[CATEGORY]', '[DOMAIN]', '[HOST]', \
                '[category:LIFE_ARTS]', 
                '[category:CULTURE]', 
                '[category:SCIENCE]', 
                '[category:STACKOVERFLOW]', 
                '[category:TECHNOLOGY]', 
                '[domain:stackexchange]',
                '[domain:stackoverflow]',
                '[domain:askubuntu]',
                '[domain:serverfault]',
                '[domain:superuser]',
                '[domain:mathoverflow]'] + list(df.host.unique())
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_type)
            self.tokenizer.cls_token = '[CLS]'
            self.tokenizer.sep_token = '[SEP]'
            num_added_tokens = self.tokenizer.add_tokens(ADD_TOKEN_LIST)
            print('Number of Tokens Added : ', num_added_tokens)

            
        elif((self.model_type == "albert-base-v2") or (self.model_type == "albert-large-v2") \
            or (self.model_type == "albert-xlarge-v2") \
            or (self.model_type == "albert-xxlarge-v2")):
            
            self.tokenizer = AlbertTokenizer.from_pretrained(model_type, \
                additional_special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
            
        elif((self.model_type == "gpt2")):
            
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", \
                additional_special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
            
        else:
            
            raise NotImplementedError
            
        self.augment = augment
        self.extra_token = extra_token
        self.translation_title_rate = 0.5
        self.translation_body_rate = 0.5
        self.translation_answer_rate = 0.5
        self.translation_single_language = 0.25
        self.random_select_date = 0.1
        
        # one-hot encode category and host columns
        if host_encoder is not None:
            transformed = host_encoder.transform(self.df["host"])
            self.df["host"] = transformed.tolist()
            self.df["host"] = self.df["host"].apply(lambda x: np.array(x))
        
        if category_encoder is not None:
            transformed = category_encoder.transform(self.df["category"])
            self.df["category"] = transformed.tolist()
            self.df["category"] = self.df["category"].apply(lambda x: np.array(x))

    def __getitem__(self, index):
        row = self.df.iloc[index]
        token_ids, seg_ids = self.get_token_ids(row, index)
        if self.labeled:
            labels = self.get_label(row)
            if self.extra_token:
                category_label = torch.tensor(row.category)
                host_label = torch.tensor(row.host)
                return token_ids, seg_ids, labels, category_label, host_label
            else:
                return token_ids, seg_ids, labels
        else:
            return token_ids, seg_ids

    def __len__(self):
        return len(self.df)

    def augmentation(self, text, insert=False, substitute=False, swap=True, delete=True):
        
        augs = []
        
        if insert:
            # aug = naw.ContextualWordEmbsAug(
            #     model_path=self.model_type, action="insert", device='cuda')
            aug = naw.WordEmbsAug(
                model_type='word2vec', model_path='/media/jionie/my_disk/Kaggle/Google_Quest_Answer/model/word2vec/GoogleNews-vectors-negative300.bin',
                action="insert")
            augs.append(aug)
        
        if substitute:
            # aug = naw.ContextualWordEmbsAug(
            #     model_path=self.model_type, action="substitute", device='cuda')
            # aug = naw.WordEmbsAug(
            #     model_type='word2vec', model_path='/media/jionie/my_disk/Kaggle/Google_Quest_Answer/model/word2vec/GoogleNews-vectors-negative300.bin',
            #     action="substitute")
            aug_sub = naw.SynonymAug(aug_src='wordnet')
            augs.append(aug_sub)
            # text = aug.augment(text)

        if swap:
            aug_swap = naw.RandomWordAug(action="swap")
            augs.append(aug_swap)
            # text = aug.augment(text)

        if delete:
            aug_del = naw.RandomWordAug()
            augs.append(aug_del)
            # text = aug.augment(text)
            
        aug = naf.Sometimes(augs, aug_p=0.5, pipeline_p=0.5)
        # print("before aug:", text)
        text = aug.augment(text, n=1)
        # print("after aug:", text)

        return text

    def select_tokens(self, tokens, max_num):
        if len(tokens) <= max_num:
            return tokens
        if self.train_mode:
            num_remove = len(tokens) - max_num
            remove_start = random.randint(0, len(tokens)-num_remove-1)
            return tokens[:remove_start] + tokens[remove_start + num_remove:]
        else:
            return tokens[:max_num//2] + tokens[-(max_num - max_num//2):]
        
    def trim_input_single_content(self, title, content, max_sequence_length=512, 
                t_max_len=30, c_max_len=512-30-4, num_token=3):

        if self.augment:
            title = self.augmentation(title, insert=False, substitute=True, swap=False, delete=True)
            content = self.augmentation(content, insert=False, substitute=True, swap=False, delete=True)

        t = self.tokenizer.tokenize(title)
        c = self.tokenizer.tokenize(content)

        t_len = len(t)
        c_len = len(c)

        if (t_len+c_len+num_token) > max_sequence_length:

            if t_max_len > t_len:
                t_new_len = t_len
                c_max_len = c_max_len + floor((t_max_len - t_len)/2)
            else:
                t_new_len = t_max_len

            if c_max_len > c_len:
                c_new_len = c_len 
            else:
                c_new_len = c_max_len


            if t_new_len+c_new_len+num_token > max_sequence_length:
                raise ValueError("New sequence length should be less or equal than %d, but is %d" 
                                 % (max_sequence_length, (t_new_len+c_new_len+num_token)))

            if self.augment:
                # random select
                if random.random() < self.random_select_date:
                    if len(t) - t_new_len > 0:
                        t_start = np.random.randint(0, len(t) - t_new_len)
                    else:
                        t_start = 0

                    if len(c) - c_new_len > 0:
                        c_start = np.random.randint(0, len(c) - c_new_len)
                    else:
                        c_start = 0

                    t = t[t_start : (t_start + t_new_len)]
                    c = c[c_start : (c_start + c_new_len)]
                    
                else:
                    # truncate
                    if len(t) - t_new_len > 0:
                        t = t[:t_new_len//4] + t[len(t)-t_new_len+t_new_len//4:]
                    else:
                        t = t[:t_new_len]

                    if len(c) - c_new_len > 0:
                        c = c[:c_new_len//4] + c[len(c)-c_new_len+c_new_len//4:]
                    else:
                        c = c[:c_new_len]

            else:
    
                # truncate
                if len(t) - t_new_len > 0:
                    t = t[:t_new_len//4] + t[len(t)-t_new_len+t_new_len//4:]
                else:
                    t = t[:t_new_len]

                if len(c) - c_new_len > 0:
                    c = c[:c_new_len//4] + c[len(c)-c_new_len+c_new_len//4:]
                else:
                    c = c[:c_new_len]

        # some bad cases
        if (len(t) + len(c) + num_token > max_sequence_length):
            more_token = len(t) + len(c) + num_token - max_sequence_length
            c = c[:(len(c)-more_token)]
        
        return t, c

    def trim_input(self, title, question, answer, max_sequence_length=512, 
                t_max_len=30, q_max_len=int((512-30-4)/2), a_max_len=(512-30-4 - int((512-30-4)/2)), num_token=4):

        question = html.unescape(question)
        answer = html.unescape(answer)
        title = html.unescape(title)
        
        if self.augment:
            # print("title: ", title)
            title = self.augmentation(title, insert=False, substitute=True, swap=False, delete=True)
            # print("question: ", question)
            question = self.augmentation(question, insert=False, substitute=True, swap=False, delete=True)
            # print("answer: ", answer)
            answer = self.augmentation(answer, insert=False, substitute=True, swap=False, delete=True)

        t = self.tokenizer.tokenize(title)
        q = self.tokenizer.tokenize(question)
        a = self.tokenizer.tokenize(answer)

        t_len = len(t)
        q_len = len(q)
        a_len = len(a)

        if (t_len+q_len+a_len+num_token) > max_sequence_length:

            if t_max_len > t_len:
                t_new_len = t_len
                a_max_len = a_max_len + floor((t_max_len - t_len)/2)
                q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
            else:
                t_new_len = t_max_len

            if a_max_len > a_len:
                a_new_len = a_len 
                q_new_len = q_max_len + (a_max_len - a_len)
            elif q_max_len > q_len:
                a_new_len = a_max_len + (q_max_len - q_len)
                q_new_len = q_len
            else:
                a_new_len = a_max_len
                q_new_len = q_max_len


            if t_new_len+a_new_len+q_new_len+num_token > max_sequence_length:
                raise ValueError("New sequence length should be less or equal than %d, but is %d" 
                                 % (max_sequence_length, (t_new_len+a_new_len+q_new_len+num_token)))

            if self.augment:
                # random select
                if random.random() < self.random_select_date:
                    if len(t) - t_new_len > 0:
                        t_start = np.random.randint(0, len(t) - t_new_len)
                    else:
                        t_start = 0

                    if len(q) - q_new_len > 0:
                        q_start = np.random.randint(0, len(q) - q_new_len)
                    else:
                        q_start = 0

                    if len(a) - a_new_len > 0:
                        a_start = np.random.randint(0, len(a) - a_new_len)
                    else:
                        a_start = 0

                    t = t[t_start : (t_start + t_new_len)]
                    q = q[q_start : (q_start + q_new_len)]
                    a = a[a_start : (a_start + a_new_len)]
                    
                else:
                    # truncate
                    if len(t) - t_new_len > 0:
                        t = t[:t_new_len//4] + t[len(t)-t_new_len+t_new_len//4:]
                    else:
                        t = t[:t_new_len]

                    if len(q) - q_new_len > 0:
                        q = q[:q_new_len//4] + q[len(q)-q_new_len+q_new_len//4:]
                    else:
                        q = q[:q_new_len]

                    if len(a) - a_new_len > 0:
                        a = a[:a_new_len//4] + a[len(a)-a_new_len+a_new_len//4:]
                    else:
                        a = a[:a_new_len]
                        
            else:
    
                if len(t) - t_new_len > 0:
                    t = t[:t_new_len//4] + t[len(t)-t_new_len+t_new_len//4:]
                else:
                    t = t[:t_new_len]

                if len(q) - q_new_len > 0:
                    q = q[:q_new_len//4] + q[len(q)-q_new_len+q_new_len//4:]
                else:
                    q = q[:q_new_len]

                if len(a) - a_new_len > 0:
                    a = a[:a_new_len//4] + a[len(a)-a_new_len+a_new_len//4:]
                else:
                    a = a[:a_new_len]

        # some bad cases
        if (len(a) + len(t) + len(q) + num_token > max_sequence_length):
            more_token = len(a) + len(t) + len(q) + num_token - max_sequence_length
            a = a[:(len(a)-more_token)]
        
        return t, q, a
        
    def get_token_ids(self, row, index):
        
        if self.extra_token:
            num_token = 6
        else:
            # if self.model_type == "roberta-base":
            #     num_token = 11
            # else:
            #     num_token = 4
            num_token = 4
        
        if self.content == "Question":
            num_token -= 1
        elif self.content == "Answer":
            num_token -= 1
        
        if self.content == "Question_Answer":   
            t_max_len=30
            q_max_len=int((self.max_len-t_max_len-num_token)/2)
            a_max_len=(self.max_len-t_max_len - num_token - int((self.max_len-t_max_len-num_token)/2))
        elif self.content == "Question":
            t_max_len=30
            q_max_len=self.max_len-t_max_len-num_token
            a_max_len=0
        elif self.content == "Answer":
            t_max_len=30
            q_max_len=0
            a_max_len=self.max_len-t_max_len-num_token  
        else:
            raise NotImplementedError

        
        if self.augment:
            
            if random.random() < self.translation_title_rate:
                if random.random() < self.translation_single_language:
                    title = row.t_aug
                else:
                    if random.random() < 1/3:
                        title = row.t_chinese
                    elif random.random() < 2/3:
                        title = row.t_french
                    else:
                        title = row.t_german
            else:
                title = row.question_title
            
            if not isinstance(title, str):
                if np.isnan(title):
                    title = row.question_title
                
            if random.random() < self.translation_body_rate:
                if random.random() < self.translation_single_language:
                    question = row.q_aug
                else:
                    if random.random() < 1/3:
                        question = row.b_chinese
                    elif random.random() < 2/3:
                        question = row.b_french
                    else:
                        question = row.b_german
            else:
                question = row.question_body
            
            if not isinstance(question, str):    
                if np.isnan(question):
                    question = row.question_body
                
            if random.random() < self.translation_answer_rate:
                if random.random() < self.translation_single_language:
                    answer = row.a_aug
                else:
                    if random.random() < 1/3:
                        answer = row.a_chinese
                    elif random.random() < 2/3:
                        answer = row.a_french
                    else:
                        answer = row.a_german
            else:
                answer = row.answer
            
            if not isinstance(answer, str):
                if np.isnan(answer):
                    answer = row.answer

            if self.content == "Question_Answer":
                t_tokens, q_tokens, a_tokens = self.trim_input(title, question, answer, max_sequence_length=self.max_len, \
                    t_max_len=t_max_len, q_max_len=q_max_len, a_max_len=a_max_len, num_token=num_token)
            elif self.content == "Question":
                t_tokens, c_tokens = self.trim_input_single_content(title, question, max_sequence_length=self.max_len, \
                    t_max_len=t_max_len, c_max_len=q_max_len, num_token=num_token)
            elif self.content == "Answer":
                t_tokens, c_tokens = self.trim_input_single_content(title, answer, max_sequence_length=self.max_len, \
                    t_max_len=t_max_len, c_max_len=a_max_len, num_token=num_token)
            else:
                raise NotImplementedError
            
        else:
            
            if self.content == "Question_Answer":
                t_tokens, q_tokens, a_tokens = self.trim_input(row.question_title, row.question_body, row.answer, \
                    max_sequence_length=self.max_len, \
                    t_max_len=t_max_len, q_max_len=q_max_len, a_max_len=a_max_len, num_token=num_token)
            elif self.content == "Question":
                t_tokens, c_tokens = self.trim_input_single_content(row.question_title, row.question_body, \
                    max_sequence_length=self.max_len, \
                    t_max_len=t_max_len, c_max_len=q_max_len, num_token=num_token)
            elif self.content == "Answer":
                t_tokens, c_tokens = self.trim_input_single_content(row.question_title, row.answer, \
                    
                    max_sequence_length=self.max_len, \
                    t_max_len=t_max_len, c_max_len=a_max_len, num_token=num_token)
            else:
                raise NotImplementedError

        if ((self.model_type == "bert-base-uncased") \
            or (self.model_type == "bert-base-cased") \
            or (self.model_type == "bert-large-uncased") \
            or (self.model_type == "bert-large-cased")):

            if self.content == "Question_Answer":
                if self.extra_token:
                    tokens = ['[CLS]'] + ['[CLS]'] + ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
                else:
                    tokens = ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
            elif ((self.content == "Question") or (self.content == "Answer")):
                if self.extra_token:
                    tokens = ['[CLS]'] + ['[CLS]'] + ['[CLS]'] + t_tokens + ['[SEP]'] + c_tokens + ['[SEP]']
                else:
                    tokens = ['[CLS]'] + t_tokens + ['[SEP]'] + c_tokens + ['[SEP]']
            else:
                raise NotImplementedError
            
        elif ((self.model_type == "flaubert-base-uncased") \
            or (self.model_type == "flaubert-base-cased") \
            or (self.model_type == "flaubert-large-cased")):

            if self.content == "Question_Answer":
                if self.extra_token:
                    tokens = ['[CLS]'] + ['[CLS]'] + ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
                else:
                    tokens = ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
            elif ((self.content == "Question") or (self.content == "Answer")):
                if self.extra_token:
                    tokens = ['[CLS]'] + ['[CLS]'] + ['[CLS]'] + t_tokens + ['[SEP]'] + c_tokens + ['[SEP]']
                else:
                    tokens = ['[CLS]'] + t_tokens + ['[SEP]'] + c_tokens + ['[SEP]']
            else:
                raise NotImplementedError
                
            
        elif ((self.model_type == "xlnet-base-cased") \
            or (self.model_type == "xlnet-large-cased")):
            
            if self.content == "Question_Answer":
                if self.extra_token:
                    tokens = ['[CLS]'] + ['[CLS]'] + ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
                else:
                    tokens = ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
            elif ((self.content == "Question") or (self.content == "Answer")):
                if self.extra_token:
                    tokens = ['[CLS]'] + ['[CLS]'] + ['[CLS]'] + t_tokens + ['[SEP]'] + c_tokens + ['[SEP]']
                else:
                    tokens = ['[CLS]'] + t_tokens + ['[SEP]'] + c_tokens + ['[SEP]']
            else:
                raise NotImplementedError
            
        elif (self.model_type == "roberta-base"):
            
            if self.content == "Question_Answer":
                if self.extra_token:
                    tokens = ['[CLS]'] + ['[CLS]'] + ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
                else:
                    tokens = ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
                    # tokens =  ['[CLS]']  + ['[CATEGORY]'] + ['[category:{}]'.format(row['category'])] + \
                    #                 ['[DOMAIN]'] + ['[domain:{}]'.format(row['host'].split('.')[-2])] + \
                    #                 ['[HOST]'] + [row['host']] + ['[TITLE]'] + t_tokens + ['[BODY]'] + \
                    #                 q_tokens + ['[ANSWER]'] + a_tokens + ['[SEP]']
            elif ((self.content == "Question") or (self.content == "Answer")):
                if self.extra_token:
                    tokens = ['[CLS]'] + ['[CLS]'] + ['[CLS]'] + t_tokens + ['[SEP]'] + c_tokens + ['[SEP]']
                else:
                    tokens = ['[CLS]'] + t_tokens + ['[SEP]'] + c_tokens + ['[SEP]']
            else:
                raise NotImplementedError
            
        elif ((self.model_type == "t5-base")):
            
            if self.content == "Question_Answer":
                if self.extra_token:
                    tokens = ['[CLS]'] + ['[CLS]'] + ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
                else:
                    tokens = ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
            elif ((self.content == "Question") or (self.content == "Answer")):
                if self.extra_token:
                    tokens = ['[CLS]'] + ['[CLS]'] + ['[CLS]'] + t_tokens + ['[SEP]'] + c_tokens + ['[SEP]']
                else:
                    tokens = ['[CLS]'] + t_tokens + ['[SEP]'] + c_tokens + ['[SEP]']
            else:
                raise NotImplementedError
            
        elif ((self.model_type == "albert-base-v2") \
            or (self.model_type == "albert-large-v2") \
            or (self.model_type == "albert-xlarge-v2") \
            or (self.model_type == "albert-xxlarge-v2")):
            
            if self.content == "Question_Answer":
                if self.extra_token:
                    tokens = ['[CLS]'] + ['[CLS]'] + ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
                else:
                    tokens = ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
            elif ((self.content == "Question") or (self.content == "Answer")):
                if self.extra_token:
                    tokens = ['[CLS]'] + ['[CLS]'] + ['[CLS]'] + t_tokens + ['[SEP]'] + c_tokens + ['[SEP]']
                else:
                    tokens = ['[CLS]'] + t_tokens + ['[SEP]'] + c_tokens + ['[SEP]']
            else:
                raise NotImplementedError
            
        else:
            
            raise NotImplementedError
        
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) < self.max_len:
            token_ids += [0] * (self.max_len - len(token_ids))
        ids = torch.tensor(token_ids)
        if self.model_type == "t5-base":
            seg_ids = ids
        else:
            seg_ids = self.get_seg_ids(ids)
        return ids, seg_ids
    
    def get_seg_ids(self, ids):
        seg_ids = torch.zeros_like(ids)
        seg_idx = 0
        first_sep = True
        for i, e in enumerate(ids):
            seg_ids[i] = seg_idx
            # if e == SEP_TOKEN_ID:
            if e == self.tokenizer.sep_token_id:
                if first_sep:
                    first_sep = False
                else:
                    seg_idx = 1
        pad_idx = torch.nonzero(ids == 0)
        seg_ids[pad_idx] = 0

        return seg_ids

    def get_label(self, row):
        #print(row[TARGET_COLUMNS].values)
        if self.content == "Question_Answer":
            return torch.tensor(row[TARGET_COLUMNS].values.astype(np.float32))
        elif self.content == "Question":
            return torch.tensor(row[QUESTION_TARGET_COLUMNS].values.astype(np.float32))
        elif self.content == "Answer":
            return torch.tensor(row[ANSWER_TARGET_COLUMNS].values.astype(np.float32))
        else:
            raise NotImplementedError

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        seg_ids = torch.stack([x[1] for x in batch])
    
        if self.labeled:
            labels = torch.stack([x[2] for x in batch])
            if self.extra_token:
                category_labels = torch.stack([x[3] for x in batch])
                host_labels = torch.stack([x[4] for x in batch])
                return token_ids, seg_ids, labels, category_labels, host_labels
            else:
                return token_ids, seg_ids, labels
        else:
            return token_ids, seg_ids


def get_test_loader(data_path="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/test.csv", \
    max_len=512, \
    model_type="bert-base-uncased", \
    content="Question", \
    batch_size=4, \
    extra_token=True):
    
    ds_test = QuestDataset(test_df, None, None, max_len, \
        model_type, content=content, train_mode=False, labeled=False, augment=False, extra_token=extra_token)
    loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=ds_test.collate_fn, drop_last=False)
    loader.num = len(test_df)
    
    return loader

def get_train_val_split(data_path="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/train_augment.csv", \
                        save_path="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/", \
                        n_splits=5, \
                        seed=42, \
                        split="GroupKfold"):

    os.makedirs(save_path + '/split', exist_ok=True)
    df = pd.read_csv(data_path, encoding='utf8')
    
    if split == "GroupKfold":
        # shuffle df by seed
        df = shuffle(df, random_state=seed)
        kf = GroupKFold(n_splits=n_splits).split(X=df.question_body, groups=df.question_body)
    elif split == "MultiStratifiedKfold":
        kf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True).split(df.question_body, \
            df[TARGET_COLUMNS].values)
    else:
        raise NotImplementedError
        
    for fold, (train_idx, valid_idx) in enumerate(kf):
        
        df_train = df.iloc[train_idx]
        df_val = df.iloc[valid_idx]

        df_train.to_csv(save_path + '/split/train_fold_%s_seed_%s.csv'%(fold, seed))
        df_val.to_csv(save_path + '/split/val_fold_%s_seed_%s.csv'%(fold, seed))

    return 
    

def get_train_val_loaders(train_data_path="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/split/train_fold_0_seed_42.csv", \
                        val_data_path="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/split/train_fold_0_seed_42.csv", \
                        host_encoder=None, \
                        category_encoder=None, \
                        max_len=512, \
                        model_type="bert-base-uncased", \
                        content="Question", \
                        batch_size=4, \
                        val_batch_size=4, \
                        num_workers=2, \
                        augment=True, \
                        extra_token=True):

    
    df_train = pd.read_csv(train_data_path, encoding='utf8')
    df_val = pd.read_csv(val_data_path, encoding='utf8')

    print(df_train.shape)
    print(df_val.shape)

    ds_train = QuestDataset(df_train, host_encoder, category_encoder, max_len, \
        model_type, content=content, train_mode=True, labeled=True, augment=augment, extra_token=extra_token)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=ds_train.collate_fn, drop_last=True)
    train_loader.num = len(df_train)

    ds_val = QuestDataset(df_val, host_encoder, category_encoder, max_len, \
        model_type, content=content, train_mode=False, labeled=True, augment=False, extra_token=extra_token)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, collate_fn=ds_val.collate_fn, drop_last=False)
    val_loader.num = len(df_val)
    val_loader.df = df_val

    return train_loader, val_loader, ds_train.tokenizer

############################################ Define test function

def test_train_val_split(data_path, \
                         save_path, \
                         n_splits, \
                         seed):
    
    print("------------------------testing train test splitting----------------------")
    print("data_path: ", data_path)
    print("save_path: ", save_path)
    print("n_splits: ", n_splits)
    print("seed: ", seed)

    get_train_val_split(data_path, save_path, n_splits, seed)

    print("generating successfully, please check results !")

    return

def test_train_loader(train_data_path="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/split/train_fold_0_seed_42.csv", \
                     val_data_path="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/split/train_fold_0_seed_42.csv", \
                     host_encoder=None, \
                     category_encoder=None, \
                     model_type="bert-base-uncased", \
                     content="Question", \
                     batch_size=4, \
                     val_batch_size=4, \
                     num_workers=2, \
                     extra_token=True):
    
    train_loader, val_loader = get_train_val_loaders(train_data_path=train_data_path, \
                     val_data_path=val_data_path, \
                     host_encoder=host_encoder, \
                     category_encoder=category_encoder, \
                     model_type=model_type, \
                     content=content, \
                     batch_size=batch_size, \
                     val_batch_size=val_batch_size, \
                     num_workers=num_workers, \
                     extra_token=extra_token)
        
    if extra_token:
        for ids, seg_ids, labels, category_labels, host_labels in train_loader:
            print("------------------------testing train loader with extra_token----------------------")
            print("ids:", ids)
            print("seg_ids (numpy): ", seg_ids.numpy())
            print("labels: ", labels)
            print("category_labels shape: ", category_labels.shape)
            print("host_labels shape: ", host_labels.shape)
            print("category_labels: ", category_labels)
            print("host_labels: ", host_labels)
            print("------------------------testing train loader finished----------------------")
            break

        for ids, seg_ids, labels, category_labels, host_labels in val_loader:
            print("------------------------testing val loader with extra_token----------------------")
            print("ids:", ids)
            print("seg_ids (numpy): ", seg_ids.numpy())
            print("category_labels shape: ", category_labels.shape)
            print("host_labels shape: ", host_labels.shape)
            print("labels: ", labels)
            print("category_labels: ", category_labels)
            print("host_labels: ", host_labels)
            print("------------------------testing val loader finished----------------------")
            break
        
    else:
        for ids, seg_ids, labels in train_loader:
            print("------------------------testing train loader without extra_token----------------------")
            print("ids:", ids)
            print("seg_ids (numpy): ", seg_ids.numpy())
            print("labels: ", labels)
            print("------------------------testing train loader finished----------------------")
            break

        for ids, seg_ids, labels in val_loader:
            print("------------------------testing val loader without extra_token----------------------")
            print("ids:", ids)
            print("seg_ids (numpy): ", seg_ids.numpy())
            print("labels: ", labels)
            print("------------------------testing val loader finished----------------------")
            break


def test_test_loader(data_path="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/test.csv", \
                     model_type="bert-base-uncased", \
                     content="Question", \
                     batch_size=4, \
                     extra_token=True):

    loader = get_test_loader(data_path=data_path, \
                            model_type=model_type, \
                            content=content, \
                            batch_size=batch_size, \
                            extra_token=extra_token)

    for ids, seg_ids in loader:
        print("------------------------testing test loader----------------------")
        print("ids: ", ids)
        print("seg_ids (numpy): ", seg_ids.numpy())
        print("------------------------testing test loader finished----------------------")
        break


if __name__ == "__main__":

    args = parser.parse_args()
    
    # one hot columns
    train_df = pd.read_csv(args.data_path)
    test_df = pd.read_csv(args.test_data_path)
    
    train_host_list = train_df['host'].unique().tolist()
    test_host_list = test_df['host'].unique().tolist()
    host_encoder = LabelBinarizer()
    host_encoder.fit(list(set(train_host_list + test_host_list)))
    
    train_category_list = train_df['category'].unique().tolist()
    test_category_list = test_df['category'].unique().tolist()
    category_encoder = LabelBinarizer()
    category_encoder.fit(list(set(train_category_list + test_category_list)))

    # test getting train val splitting
    # test_train_val_split(args.data_path, \
    #                      args.save_path, \
    #                      args.n_splits, \
    #                      args.seed)

    # test train_data_loader
     
    train_data_path = args.save_path + '/split/train_fold_%s_seed_%s.csv'%(args.test_fold, args.seed)
    val_data_path= args.save_path + '/split/val_fold_%s_seed_%s.csv'%(args.test_fold, args.seed)

    test_train_loader(train_data_path=train_data_path, \
                      val_data_path=val_data_path, \
                      host_encoder=host_encoder, \
                      category_encoder=category_encoder, \
                      model_type=args.model_type, \
                      content=args.content, \
                      batch_size=args.batch_size, \
                      val_batch_size=args.val_batch_size, \
                      num_workers=args.num_workers)

    # test test_data_loader
    test_test_loader(data_path=args.test_data_path, \
                     model_type=args.model_type, \
                     content=args.content, \
                     batch_size=args.val_batch_size)
    
    
    test_train_loader(train_data_path=train_data_path, \
                      val_data_path=val_data_path, \
                      host_encoder=host_encoder, \
                      category_encoder=category_encoder, \
                      model_type=args.model_type, \
                      content=args.content, \
                      batch_size=args.batch_size, \
                      val_batch_size=args.val_batch_size, \
                      num_workers=args.num_workers, \
                      extra_token=False)

    # test test_data_loader
    test_test_loader(data_path=args.test_data_path, \
                     model_type=args.model_type, \
                     content=args.content, \
                     batch_size=args.val_batch_size, \
                      extra_token=False)
