import time

import numpy as np
import pandas as pd
import os
import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info

"""screen hparams to avoid unnecessary information"""
all_hparams = ["cfg", "seed", "train_batch_size", "valid_batch_size", "test_batch_size", "num_classes", "num_workers",
               "limit_train_batches", "limit_val_batches", "limit_test_batches", "limit_predict_batches",
               "train_size", "img_mean", "img_std", "aug", "aug_prob",
               "lr", "lr_scheduler", "lr_warmup_epochs", "lr_decay_steps", "lr_decay_rate",
               "lr_decay_min_lr", "dataset", "data_dir", "model_name", "loss"
                                                                       "momentum", "weight_decay", "max_epochs", "gpus",
               "optimizer", "l1", "l2", "patience",
               "depth", "width_multiplier", "input_size", "output_size", "num_hidden_layers", "dropout", "activation",
               "max_epochs", "accumulate_grad_batches", "gpus", "strategy", "precision",
               "val_check_interval", "log_dir", "exp_name", "run", "save_dir"]

screened_hparams = ["uid", "log_dir", "exp_name", "run", "dataset", "data_dir", "model_name", "model_path", "seed",
                    "train_batch_size", "dropout", "depth", "width_multiplier", "num_classes", "loss",
                    "aug", "aug_prob",
                    "lr", "lr_scheduler", "lr_warmup_epochs", "lr_decay_steps", "lr_decay_rate", "lr_decay_min_lr",
                    "max_epochs", "cur_epoch", "cur_step", "optimizer", "weight_decay", "momentum", "l1", "l2",
                    "patience",
                    "accumulate_grad_batches",
                    "limit_train_batches", "limit_val_batches", "limit_test_batches", "limit_predict_batches",
                    "accelerator", "gpus", "strategy", "sync_batchnorm", "precision",
                    "img_mean", "img_std", "val_check_interval",
                    "train_loss", "train_acc", "val_loss", "val_acc"]

dynamic_hparams = ["log_dir", "model_path", "cur_epoch", "cur_step", "train_loss", "train_acc", "val_loss", "val_acc",
                   "strategy",
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
        self.csv_path = hparams["save_dir"] + hparams["exp_name"] + "/models_log.csv"
        self.log_dir = hparams["log_dir"] + hparams["exp_name"]
        self.strategy = hparams["strategy"]
        self.gpus = hparams["gpus"]
        if os.path.exists(self.csv_path):
            hparams = {k: hparams[k] for k in static_hparams if k in hparams}
        else:
            if not os.path.exists(hparams["save_dir"] + hparams["exp_name"]):
                os.makedirs(hparams["save_dir"] + hparams["exp_name"])
            hparams = {k: hparams[k] for k in screened_hparams if k in hparams}
        self.cur_uid = int(hparams2csv(hparams, self.csv_path))

    # used to generate the untrained checkpoint
    def init_checkpoint(self, trainer, metrics):
        loss, acc = metrics["test_loss"], metrics["test_acc"]
        path = self.dirpath + f"uid={self.cur_uid}-model_name={self.hparams['model_name']}-dataset={self.hparams['dataset']}" \
                                                           f"-epoch=0-step=0-val_acc={acc:.3f}.ckpt"
        trainer.save_checkpoint(path)
        print(f"Saved first checkpoint to: {path}!")
        df = pd.read_csv(self.csv_path)
        df.loc[df["uid"] == self.cur_uid, "log_dir"] = self.log_dir + "/" + f"version_{trainer.logger.version}"
        df.loc[df["uid"] == self.cur_uid, "strategy"] = str(self.strategy)
        df.loc[df["uid"] == self.cur_uid, "gpus"] = str(self.gpus)
        df.loc[df["uid"] == self.cur_uid, "cur_epoch"] = 0
        df.loc[df["uid"] == self.cur_uid, "model_path"] = path
        df.loc[df["uid"] == self.cur_uid, "cur_step"] = 0
        df.loc[df["uid"] == self.cur_uid, "val_acc"] = acc
        df.loc[df["uid"] == self.cur_uid, "val_loss"] = loss
        df.to_csv(self.csv_path, index=False)

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
        row = {}
        row["uid"] = int(self.cur_uid)
        row["val_loss"] = monitor_candidates["val_loss_epoch"].detach().cpu().numpy()
        row["val_acc"] = monitor_candidates["val_acc_epoch"].detach().cpu().numpy()
        row["cur_epoch"] = int(monitor_candidates["epoch"])
        row["cur_step"] = int(monitor_candidates["step"])

        if pd.isna(df.loc[df["uid"] == self.cur_uid]["log_dir"]).any() or \
                f"version_{trainer.logger.version}" not in df.loc[df["uid"] == self.cur_uid]["log_dir"].values[0]:
            row["log_dir"] = self.log_dir + "/" + f"version_{trainer.logger.version}"
            row["strategy"] = str(self.strategy)
            row["gpus"] = str(self.gpus)
            row["cur_epoch"] = int(monitor_candidates["epoch"])

        row["model_path"] = filepath
        if self.every_n_train_steps is not None and self.every_n_train_steps > 0:
            if "train_loss" in monitor_candidates:
                row["train_loss"] = monitor_candidates["train_loss"].detach().cpu().numpy()
                row["train_acc"] = monitor_candidates["train_acc"].detach().cpu().numpy()
            # add a row for the current step
            old_row = df.loc[df["uid"] == self.cur_uid].iloc[0].to_dict()
            old_row.update(row)
            df = df.append(old_row, ignore_index=True)
        else:
            if "train_loss_epoch" in monitor_candidates:
                row["train_loss"] = monitor_candidates["train_loss_epoch"].detach().cpu().numpy()
                row["train_acc"] = monitor_candidates["train_acc_epoch"].detach().cpu().numpy()
            # modify the row for the current step
            old_row = df.loc[df["uid"] == self.cur_uid].iloc[0].to_dict()
            old_row.update(row)
            df.loc[df["uid"] == self.cur_uid] = old_row
        df.to_csv(self.csv_path, index=False)
        ##############################################

        if del_filepath is not None and filepath != del_filepath:
            trainer.strategy.remove_checkpoint(del_filepath)
