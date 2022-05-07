import numpy as np
import pandas as pd
import os
import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info

"""screen hparams to avoid unnecessary information"""

screened_hparams = ["uid", "log_dir", "exp_name", "run", "dataset", "data_dir", "model_name", "model_path", "seed",
                    "train_batch_size", "dropout", "depth", "width_multiplier", "class_num", "loss",
                    "aug", "aug_prob",
                    "lr", "lr_scheduler", "lr_warmup_epochs", "lr_decay_steps", "lr_decay_rate", "lr_decay_min_lr",
                    "max_epochs", "cur_epoch", "optimizer", "weight_decay", "momentum", "l1", "l2", "patience",
                    "accumulate_grad_batches",
                    "limit_train_batches", "limit_val_batches", "limit_test_batches", "limit_predict_batches",
                    "accelerator", "gpus", "strategy", "sync_batchnorm", "precision",
                    "img_mean", "img_std",
                    "train_loss", "train_acc", "val_loss", "val_acc"]

dynamic_hparams = ["log_dir", "model_path", "cur_epoch", "train_loss", "train_acc", "val_loss", "val_acc", "strategy",
                   "gpus"]
static_hparams = [p for p in screened_hparams if p not in dynamic_hparams]


def generate_uid(df):
    """Generates a unique id."""
    if df.empty:
        next_id = 0
    else:
        next_id = max(df.uid) + 1
    return int(next_id)


def match_hparams(hparams, df):
    """check if hparmas values are in the dataframe"""
    found = False
    # convert None to 'None'
    if "strategy" in hparams:
        hparams["strategy"] = str(hparams["strategy"])

    for idx, row in df.iterrows():
        # check if all hparams are in the existed row
        # to prevent the same hparams with different gpus to be added as different experiments
        # exclude gpus from the comparison during hparams matching
        match = [1 for k, v in hparams.items() if row[k] == v or str(row[k]) == str(v)]
        if sum(match) == len(hparams):
            # if all hparams are matched, modify the metric
            cur_uid = row['uid']
            found = True
            break

    if not found:
        # if not, append a new row
        hparams["uid"] = generate_uid(df)
        cur_uid = hparams["uid"]
        for param in dynamic_hparams:
            hparams[param] = None
        df = df.append(hparams, ignore_index=True)

    return df, cur_uid


def hparams2csv(hparams, csv_path):
    """Writes hparams to a csv file."""
    # convert None to 'None'
    if "strategy" in hparams:
        hparams["strategy"] = str(hparams["strategy"])
    # if csvs exist, open in append mode
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df, cur_uid = match_hparams(hparams, df)
    else:
        for param in dynamic_hparams:
            hparams[param] = None
        df = pd.DataFrame(columns=hparams.keys())
        hparams["uid"] = generate_uid(df)
        df = df.append(hparams, ignore_index=True)
        # reorder the columns
        df = df[screened_hparams]
        cur_uid = hparams["uid"]

    df.to_csv(csv_path, index=False)
    return cur_uid


class CSVModelCheckpoint(ModelCheckpoint):
    def __init__(self,
                 hparams=None,
                 dirpath=None,
                 filename=None,
                 monitor=None,
                 verbose=False,
                 save_last=None,
                 save_top_k=1,
                 save_weights_only=False,
                 mode="min",
                 auto_insert_metric_name=True,
                 every_n_train_steps=None,
                 train_time_interval=None,
                 every_n_epochs=None,
                 save_on_train_epoch_end=None, ):

        super().__init__()
        self.__dict__.update(locals())
        self.csv_index = None
        self.csv_path = hparams["save_dir"] + "/models_log.csv"
        self.log_dir = hparams["log_dir"] + hparams["exp_name"]
        self.strategy = hparams["strategy"]
        self.gpus = hparams["gpus"]
        if os.path.exists(self.csv_path):
            hparams = {k: hparams[k] for k in static_hparams if k in hparams}
        else:
            hparams = {k: hparams[k] for k in screened_hparams if k in hparams}
        self.cur_uid = int(hparams2csv(hparams, self.csv_path))

    # rewrite the save function to save the hparams into csv
    def _update_best_and_save(self, current, trainer, monitor_candidates):
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"), device=current.device)

        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
        self.best_model_score = self.best_k_models[self.best_model_path]

        # add the uid to the filepath and del_filepath
        filepath = filepath.replace("uid", f"uid={self.cur_uid}")
        del_filepath = del_filepath.replace("uid", f"uid={self.cur_uid}") if del_filepath is not None else None

        if self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f}"
                f" (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
            )
        self._save_checkpoint(trainer, filepath)
        #################################################
        # (extra insertion) save the hparams into csv
        df = pd.read_csv(self.csv_path)
        df["uid"] = df["uid"].astype(int)
        df.loc[df["uid"] == self.cur_uid, "train_loss"] = monitor_candidates["train_loss_epoch"].detach().cpu().numpy()
        df.loc[df["uid"] == self.cur_uid, "train_acc"] = monitor_candidates["train_acc"].detach().cpu().numpy()
        df.loc[df["uid"] == self.cur_uid, "val_loss"] = monitor_candidates["val_loss"].detach().cpu().numpy()
        df.loc[df["uid"] == self.cur_uid, "val_acc"] = monitor_candidates["val_acc"].detach().cpu().numpy()
        df.loc[df["uid"] == self.cur_uid, "cur_epoch"] = int(monitor_candidates["epoch"])

        if pd.isna(df.loc[df["uid"] == self.cur_uid]["log_dir"]).any() or \
                f"version_{trainer.logger.version}" not in df.loc[df["uid"] == self.cur_uid]["log_dir"].values[0]:
            df.loc[df["uid"] == self.cur_uid, "log_dir"] = self.log_dir + "/" + f"version_{trainer.logger.version}"
            df.loc[df["uid"] == self.cur_uid, "strategy"] = str(self.strategy)
            df.loc[df["uid"] == self.cur_uid, "gpus"] = str(self.gpus)

        df.loc[df["uid"] == self.cur_uid, "model_path"] = filepath
        df.to_csv(self.csv_path, index=False)
        ##############################################

        if del_filepath is not None and filepath != del_filepath:
            trainer.strategy.remove_checkpoint(del_filepath)
