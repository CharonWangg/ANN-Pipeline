import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import ast
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import load_dataset
import yaml

Config = yaml.safe_load(open("./config.yaml"))

def load_cmu_scifi():
    dataset = load_dataset("lara-martin/Scifi_TV_Shows")
    return dataset


def str2list(string):
    l = ast.literal_eval(string)
    return l


def parse_event(tokenizer, event):
    args = str2list(event)
    args = [tokenize(tokenizer, arg) for arg in args]

    info = {}
    info["input_ids"] = [arg["input_ids"] for arg in args]
    info["attention_mask"] = [arg["attention_mask"] for arg in args]
    return info


def numpy_split_df(df, split_num):
    lst_index = list(map(lambda a: a.tolist(), np.array_split([i for i in range(len(df))], split_num)))
    chunk_list = []
    for idx in lst_index:
        df_split = df.iloc[idx[0]: idx[-1] + 1]
        chunk_list.append(df_split)
    return chunk_list


def apply_parallel(df, func, num_cpu, tokenizer, valid=False):
    # divide events into chunks
    chunk_list = numpy_split_df(df, num_cpu*10)
    examples_list = Parallel(n_jobs=num_cpu, backend='multiprocessing')(delayed(func)(split_df, df, tokenizer, valid) for split_df in
                                             tqdm(chunk_list, desc="Parallel", total=len(chunk_list)))
    return examples_list


def super_make_examples(df, valid=False):
    num_cpu = cpu_count()
    examples_list = apply_parallel(df, make_examples, num_cpu, valid)
    examples = sum(examples_list, [])
    return examples

def super_make_sent_examples(df, tokenizer_name, valid=False):
    num_cpu = cpu_count()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    examples_list = apply_parallel(df, make_sent_examples, num_cpu, tokenizer, valid)
    examples = sum(examples_list, [])
    return examples


def df_split(df):
    '''
    x: time series array ()
    :return:
    '''
    train, valid = train_test_split(df, train_size=Config["DATA"]["TRAIN_SIZE"], shuffle=False)
    return train, valid

# TODO Sample a following event from same story
def hard_positive_sample(df, idx):
    idx = df[df["idx"] == idx].index.tolist()[0]
    story_num = df.loc[idx]["story_num"]
    # exclude input event and other story
    df = df[df["story_num"] == story_num]
    try:
        sample = df.loc[idx + 1]
    except:
        sample = df.loc[idx]
    return sample


# TODO Randomly sample a event from same story
def soft_positive_sample(df, idx):
    idx = df[df["idx"] == idx].index.tolist()[0]
    story_num = df.loc[idx]["story_num"]
    # exclude input event and other story
    df = df.drop(index=idx)
    df = df[df["story_num"] == story_num]
    sample = df.sample(n=1).values[0]
    return sample


# TODO Randomly sample a event from other story
def hard_negative_sample(df, idx):
    idx = df[df["idx"] == idx].index.tolist()[0]
    story_num = df.loc[idx]["story_num"]
    # exclude the same story
    df = df[df["story_num"] != story_num]
    sample = df.sample(n=1).values[0]
    return sample


# TODO Randomly sample a event from same story
def soft_negative_sample(df, idx):
    idx = df[df["idx"] == idx].index.tolist()[0]
    story_num = df.loc[idx]["story_num"]
    # exclude input event and other story
    df = df.drop(index=idx)
    df = df[df["story_num"] == story_num]
    sample = df.sample(n=1).values[0]
    return sample


def tokenize(tokenizer, sentence, config=Config):
    tokens_info = tokenizer(sentence,
                            padding='max_length',
                            truncation=True,
                            max_length=config["TOKENIZE"]["MAX_SEQ_LEN"],
                            )
    return tokens_info

def make_sent_examples(group, df, tokenizer=None, valid=False):
    examples = []
    # for training --》 No Negative Sample
    # if not valid:
    for idx, pivot in group.iterrows():
        ## Only use Duplicate Samples
        example = {"anchor": {}, "positive": {}, "negative": {}}
        # anchor
        anchor = str2list(pivot["anchor_event"])
        anchor = "[SEP]".join(anchor)
        tokens_info = tokenize(tokenizer, anchor)
        example["anchor"]["input_ids"] = tokens_info["input_ids"]
        example["anchor"]["attention_mask"] = tokens_info["attention_mask"]
        # positive
        positive = str2list(pivot["positive_event"])
        positive = "[SEP]".join(positive)
        tokens_info = tokenize(tokenizer, positive)
        example["positive"]["input_ids"] = tokens_info["input_ids"]
        example["positive"]["attention_mask"] = tokens_info["attention_mask"]
        # hard_negative
        negative = str2list(pivot["negative_event"])
        negative = "[SEP]".join(negative)
        tokens_info = tokenize(tokenizer, negative)
        example["negative"]["input_ids"] = tokens_info["input_ids"]
        example["negative"]["attention_mask"] = tokens_info["attention_mask"]
        # label - story_num
        example["label"] = pivot["story_num"]
        examples.append(example)

    return examples
