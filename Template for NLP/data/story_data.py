import os
import pandas as pd
import torch
import torch.utils.data as data
from utils import *


class StoryData(data.Dataset):
    def __init__(self, ds_type="train", aug=False,
                 train_path=None, valid_path=None, test_path=None,
                 sample_size=256, tokenizer_name=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.aug = self.ds_type=="train" and self.aug

        self.check_files()

    def check_files(self):
        if self.ds_type == "train":
            assert os.path.exists(self.train_path), "Train file not found"
            path = self.train_path
        elif self.ds_type == "valid":
            assert os.path.exists(self.valid_path), "Val file not found"
            path = self.valid_path
        elif self.ds_type == "test":
            assert os.path.exists(self.test_path), "Test file not found"
            path = self.test_path
        else:
            raise Exception("Invalid dataset type")

        # Load data
        data = pd.read_csv(path).sample(n=self.sample_size) if self.sample_size != -1 else pd.read_csv(path)
        self.pairs = super_make_sent_examples(data, self.tokenizer_name, self.ds_type!="train")

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        anchor = {
            'input_ids': torch.tensor(pair["anchor"]["input_ids"], dtype=torch.long),
            'attention_mask': torch.tensor(pair["anchor"]["attention_mask"], dtype=torch.long)
        }
        positive = {
            'input_ids': torch.tensor(pair["positive"]["input_ids"], dtype=torch.long),
            'attention_mask': torch.tensor(pair["positive"]["attention_mask"], dtype=torch.long)
        }
        negative = {
            'input_ids': torch.tensor(pair["negative"]["input_ids"], dtype=torch.long),
            'attention_mask': torch.tensor(pair["negative"]["attention_mask"], dtype=torch.long)
        }

        item = {"anchor": anchor, "positive": positive, "negative": negative}
        label = torch.tensor(pair["label"], dtype=torch.long)

        return item, label

    def __len__(self):
        return len(self.pairs)

