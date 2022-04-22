import seaborn as sns
from sklearn.linear_model import LogisticRegression
import pandas as pd
from utils.data import *
from utils.util import fix_all_seeds
from utils.trainer import Trainer
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import yaml

def super_make_sent_examples(df, valid=False):
    num_cpu = cpu_count()
    examples_list = apply_parallel(df, loop, num_cpu, valid)
    examples = sum(examples_list, [])
    return examples

def loop(group, df, valid=False):
    examples = []
    df["event"] = df["event"].str.lstrip()
    # for training --》 No Negative Sample
    # if not valid:
    for idx, anchor in group.iterrows():    # a
        example = {}
        # label - story_num
        example["story_num"] = anchor["story_num"]
        example["story_line"] = anchor["story_line"]
        #anchor
        example["anchor_event"] = anchor["event"]
        example["anchor_gen_event"] = anchor["gen_event"]
        example["anchor_sent"] = anchor["sent"]
        # positive
        row = hard_positive_sample(df, anchor["idx"])
        example["positive_event"] = row[3]
        example["positive_gen_event"] = row[5]
        example["positive_sent"] = row[4]
        # hard_negative
        row = hard_negative_sample(df, anchor["idx"])
        example["negative_event"] = row[3]
        example["negative_gen_event"] = row[5]
        example["negative_sent"] = row[4]

        examples.append(example)

    return examples


if __name__ == "__main__":
    data = pd.read_csv("./data/cmu_scifi/train_flatten.csv", index_col=0).reset_index(drop=True)
    data["idx"] = range(len(data))
    # rename columns
    #data.rename(columns={"event": "gen_event0", "gen_event": "event"}, inplace=True)

    fix_all_seeds(42)  # fix all seeds for reproducibility of results
    data = super_make_sent_examples(data)
    data = pd.DataFrame(data)
    data.to_csv("./data/cmu_scifi/full.csv")
    train, valid = train_test_split(data, train_size=0.8, random_state=42)
    train.to_csv("./data/cmu_scifi/triplet_train_42.csv")
    valid.to_csv("./data/cmu_scifi/triplet_valid_42.csv")


