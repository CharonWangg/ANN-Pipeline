from copy import copy

import yaml
import pandas as pd
from .log_util import match_hparams, all_hparams, static_hparams
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import argparse
import os
import ast
from pathlib2 import Path


# flatten the .yaml hierarchy dict
def flatten_dict(d, key=''):
    items = []
    for k, v in d.items():
        new_key = key + k + '_'
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            new_key = new_key.rstrip('_')
            items.append((new_key.lower(), v))
    return dict(items)


def yaml_to_kwargs(cfg_path):
    yaml_dict = yaml.safe_load(open(cfg_path))
    kwargs = flatten_dict(yaml_dict)
    new_kwargs = {}
    for long_k in kwargs:
        for short_k in all_hparams:
            if short_k in long_k:
                new_kwargs[short_k] = kwargs[long_k]
                break
    print(len(kwargs))
    print(len(new_kwargs))
    return new_kwargs

# load model in uid format by searching its hparams in a csv file
def load_model_path_by_csv(root, hparams=None, mode='train'):
    if hparams is None:
        return None
    elif isinstance(hparams, dict):
        pass
    elif isinstance(hparams, object):
        hparams = vars(hparams)

    if mode == "train":
        # concat the root and .csv
        csv_path = root + "models_log.csv"
        hparams = {k: v for k, v in hparams.items() if k in static_hparams}
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df, cur_uid = match_hparams(hparams, df)
            if (pd.isna(df.loc[df["uid"]==cur_uid, "model_path"])).any():
                return None
            else:
                # return the best val_acc model path
                load_path = df.groupby(by="uid").get_group(cur_uid)
                load_path = load_path.iloc[0]["model_path"] if load_path.shape[0] == 1 else load_path.sorted_values(by="val_acc", ascending=False)["model_path"].values[0]
                print('Loading the best model of uid {} from {}'.format(cur_uid, load_path))
                return load_path if not pd.isna(load_path) else None
        else:
            print("The csv file does not exist, will create one during training.")
            return None
    elif mode == "inference":
        # concat the root and .csv
        csv_path = root + "models_log.csv"
        df = pd.read_csv(csv_path)
        # exclude the inference hparams that could be different from training hparams
        hparams.pop("inference_seed")
        hparams.pop("gpus")
        try:
            for k, v in hparams.items():
                df = df.groupby(by=k).get_group(v)
            # return the best val_acc model path
            load_path = df.iloc[0]["model_path"] if df.shape[0] == 1 else df.sorted_values(by="val_acc", ascending=False)["model_path"].values[0]

            return load_path
        except KeyError:
            print("The hparams are not in the csv file.")
            return None





def load_model_path_by_hparams(root, hparams=None):
    """ When best = True, return the best model's path in a directory
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the
        first three args.
    Args:
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """
    if hparams is None:
        return None
    elif isinstance(hparams, object):
        hparams = hparams.__dict__
    # concat the root and dataset name
    root = os.path.join(root, hparams["exp_name"])
    if not os.path.exists(root):
        return None

    if Path(root).is_file():
        return root


    # match the hparams to the file name
    files = [str(i) for i in list(Path(root).iterdir())
             if f'model_name={hparams["model_name"]}'.lower() in str(i) and
             f'dataset={hparams["dataset"]}'.lower() in str(i) and
             f'depth={hparams["depth"]}'.lower() in str(i) and
             f'width_multiplier={hparams["width_multiplier"]}'.lower() in str(i) and
             f'dropout={hparams["dropout"]}'.lower() in str(i) and
             f'lr={hparams["lr"]}'.lower() in str(i) and
             f'weight_decay={hparams["weight_decay"]}'.lower() in str(i) and
             f'max_epochs={hparams["max_epochs"]}'.lower() in str(i) and
             f'batch_size={hparams["train_batch_size"]}'.lower() in str(i) and
             f'l1={hparams["l1"]}'.lower() in str(i) and
             f'l2={hparams["l2"]}'.lower() in str(i) and
             f'optimizer={hparams["optimizer"]}'.lower() in str(i) and
             f'lr_scheduler={hparams["lr_scheduler"]}'.lower() in str(i) and
             f'criterion={hparams["loss"]}'.lower() in str(i) and
             f'precision={hparams["precision"]}'.lower() in str(i) and
             f'seed={hparams["seed"]}'.lower() in str(i)
             in str(i).lower()]

    if not files:
        return None
    else:
        print('Loading model from {}'.format(files[-1]))
        return files[-1]


def configure_hidden_size(hidden_size, num_hidden_layers):
    # check if hidden size is a list
    if not isinstance(hidden_size, list):
        hidden_size = [hidden_size] * num_hidden_layers
    elif len(hidden_size) == 1:
        hidden_size = [hidden_size] * num_hidden_layers

    return hidden_size


def args_setup(cfg_path='config.yaml'):
    cfg = yaml.safe_load(open(cfg_path))
    parser = ArgumentParser()
    # init
    parser.add_argument('--cfg', type=str, default=cfg_path, help='config file path')
    parser.add_argument('--seed', default=cfg["SEED"], type=int)
    parser.add_argument('--inference_seed', default=cfg["SEED"], type=int)
    # data
    parser.add_argument('--train_batch_size', default=cfg["DATA"]["TRAIN_BATCH_SIZE"], type=int)
    parser.add_argument('--valid_batch_size', default=cfg["DATA"]["VALID_BATCH_SIZE"], type=int)
    parser.add_argument('--test_batch_size', default=cfg["DATA"]["TEST_BATCH_SIZE"], type=int)
    parser.add_argument('--train_size', default=cfg["DATA"]["TRAIN_SIZE"], type=int)
    parser.add_argument('--num_classes', default=cfg["DATA"]["NUM_CLASSES"], type=int)
    parser.add_argument('--num_workers', default=cfg["DATA"]["NUM_WORKERS"], type=int)
    # data augmentation
    parser.add_argument('--aug', default=cfg["DATA"]["AUG"], type=bool)
    parser.add_argument('--aug_prob', default=cfg["DATA"]["AUG_PROB"], type=float)
    # train
    parser.add_argument('--lr', default=cfg["OPTIMIZATION"]["LR"], type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', default=cfg["OPTIMIZATION"]["LR_SCHEDULER"],
                        choices=['step', 'cosine', 'constant', 'cyclic', 'plateau', 'multistep', 'cifar', 'one_cycle'], type=str)
    parser.add_argument('--lr_warmup_epochs', default=cfg["OPTIMIZATION"]["LR_WARMUP_EPOCHS"], type=int)
    parser.add_argument('--lr_decay_steps', default=cfg["OPTIMIZATION"]["LR_DECAY_STEPS"], type=int)
    parser.add_argument('--lr_decay_rate', default=cfg["OPTIMIZATION"]["LR_DECAY_RATE"], type=float)
    parser.add_argument('--lr_decay_min_lr', default=cfg["OPTIMIZATION"]["LR_DECAY_MIN_LR"], type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default=cfg["DATA"]["DATASET"], type=str)
    parser.add_argument('--data_dir', default=cfg["DATA"]["DATA_PATH"], type=str)
    parser.add_argument('--model_name', default=cfg["MODEL"]["NAME"], type=str)
    parser.add_argument('--loss', default=cfg["OPTIMIZATION"]["LOSS"],
                        choices=["cross_entropy", "binary_cross_entropy", "l1", "l2"], type=str)
    parser.add_argument('--weight_decay', default=cfg["OPTIMIZATION"]["WEIGHT_DECAY"], type=float)
    parser.add_argument('--momentum', default=cfg["OPTIMIZATION"]["MOMENTUM"], type=float)
    parser.add_argument('--optimizer', default=cfg["OPTIMIZATION"]["OPTIMIZER"], choices=["Adam", "SGD", "RMSprop"],
                        type=str)
    parser.add_argument('--l1', default=cfg["MODEL"]["L1"], type=float)
    parser.add_argument('--l2', default=cfg["MODEL"]["L2"], type=float)
    parser.add_argument('--patience', default=cfg["OPTIMIZATION"]["PATIENCE"], type=int)
    parser.add_argument('--log_dir', default=cfg["LOG"]["PATH"], type=str)
    parser.add_argument('--exp_name', default=cfg["LOG"]["NAME"], type=str)
    parser.add_argument('--run', default=0, type=int)
    parser.add_argument('--save_dir', default=cfg["MODEL"]["SAVE_DIR"], type=str)

    # Model Hyperparameters
    parser.add_argument('--input_size', default=cfg["MODEL"]["INPUT_SIZE"], type=int)
    parser.add_argument('--output_size', default=cfg["MODEL"]["OUTPUT_SIZE"], type=int)
    parser.add_argument('--num_hidden_layers', default=cfg["MODEL"]["NUM_HIDDEN_LAYERS"], type=int)
    parser.add_argument('--dropout', default=cfg["MODEL"]["DROPOUT"], type=float)
    parser.add_argument('--activation', default=cfg["MODEL"]["ACTIVATION"],
                        choices=["relu", "tanh", "sigmoid", "leaky_relu", "prelu"], type=str)

    # Specific Model Hyperparameters for ResNet
    parser.add_argument('--depth', default=cfg["MODEL"]["RESNET"]["DEPTH"], type=int)
    parser.add_argument('--width_multiplier', default=cfg["MODEL"]["RESNET"]["WIDTH_MULTIPLIER"], type=int)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=cfg["OPTIMIZATION"]["MAX_EPOCHS"])
    parser.set_defaults(accumulate_grad_batches=cfg["OPTIMIZATION"]["ACC_GRADIENT_STEPS"])
    parser.set_defaults(accelerator="auto")
    parser.set_defaults(gpus=cfg["GPUS"])
    parser.set_defaults(strategy=cfg["STRATEGY"])
    parser.set_defaults(precision=cfg["PRECISION"])
    parser.set_defaults(limit_train_batches=cfg["DATA"]["NUM_TRAIN"], type=float)
    parser.set_defaults(limit_val_batches=cfg["DATA"]["NUM_VAL"], type=float)
    parser.set_defaults(limit_test_batches=cfg["DATA"]["NUM_TEST"], type=float)
    parser.set_defaults(limit_predict_batches=cfg["DATA"]["NUM_PREDICT"], type=float)
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    # List Arguments
    args.img_mean = cfg["DATA"]["IMG_MEAN"]
    args.img_std = cfg["DATA"]["IMG_STD"]
    # parse the strategy
    if str(args.strategy).lower() == "none":
        args.strategy = None
    # parse the continuous devices argument if it is a string
    if isinstance(args.gpus, str):
        if "[" in args.gpus:
            args.gpus = ast.literal_eval(args.gpus)
        else:
            args.gpus = [int(g) for g in args.gpus]
        args.devices = len(args.gpus)
    elif isinstance(args.gpus, int):
        args.gpus = [args.gpus]
        args.devices = 1
    elif isinstance(args.gpus, list):
        args.devices = len(args.gpus)

    # sync batch size with devices
    if args.devices > 1:
        args.sync_batchnorm = True
    args.auto_select_gpus = False

    return args


def kwargs_setup(cfg_path='./config.yaml'):
    kwargs = {}
    cfg = yaml.safe_load(open(cfg_path))
    # init
    kwargs['cfg'] = cfg_path
    kwargs['seed'] = cfg["SEED"]
    kwargs['inference_seed'] = cfg["SEED"]

    # data
    kwargs['dataset'] = cfg["DATA"]["DATASET"]
    kwargs['data_dir'] = cfg["DATA"]["DATA_PATH"]
    kwargs['train_batch_size'] = cfg["DATA"]["TRAIN_BATCH_SIZE"]
    kwargs['valid_batch_size'] = cfg["DATA"]["VALID_BATCH_SIZE"]
    kwargs['test_batch_size'] = cfg["DATA"]["TEST_BATCH_SIZE"]
    kwargs['limit_train_batches'] = cfg["DATA"]["NUM_TRAIN"]
    kwargs['limit_val_batches'] = cfg["DATA"]["NUM_VAL"]
    kwargs['limit_test_batches'] = cfg["DATA"]["NUM_TEST"]
    kwargs['limit_predict_batches'] = cfg["DATA"]["NUM_PREDICT"]
    kwargs['train_size'] = cfg["DATA"]["TRAIN_SIZE"]
    kwargs['num_classes'] = cfg["DATA"]["NUM_CLASSES"]
    kwargs['num_workers'] = cfg["DATA"]["NUM_WORKERS"]
    kwargs['aug'] = cfg["DATA"]["AUG"]
    kwargs['aug_prob'] = cfg["DATA"]["AUG_PROB"]

    # model
    kwargs['model_name'] = cfg["MODEL"]["NAME"]
    kwargs['input_size'] = cfg["MODEL"]["INPUT_SIZE"]
    kwargs['output_size'] = cfg["MODEL"]["OUTPUT_SIZE"]
    kwargs['num_hidden_layers'] = cfg["MODEL"]["NUM_HIDDEN_LAYERS"]
    kwargs['dropout'] = cfg["MODEL"]["DROPOUT"]
    kwargs['activation'] = cfg["MODEL"]["ACTIVATION"]
    kwargs['l1'] = cfg["MODEL"]["L1"]
    kwargs['l2'] = cfg["MODEL"]["L2"]
    kwargs['save_dir'] = cfg["MODEL"]["SAVE_DIR"]
    # Specific Model Hyperparameters for ResNet
    kwargs['depth'] = cfg["MODEL"]["RESNET"]["DEPTH"]
    kwargs['width_multiplier'] = cfg["MODEL"]["RESNET"]["WIDTH_MULTIPLIER"]

    # optimization
    kwargs['loss'] = cfg["OPTIMIZATION"]["LOSS"]
    kwargs['lr'] = cfg["OPTIMIZATION"]["LR"]
    kwargs['max_epochs'] = cfg["OPTIMIZATION"]["MAX_EPOCHS"]
    kwargs['momentum'] = cfg["OPTIMIZATION"]["MOMENTUM"]
    kwargs['weight_decay'] = cfg["OPTIMIZATION"]["WEIGHT_DECAY"]
    kwargs['lr_scheduler'] = cfg["OPTIMIZATION"]["LR_SCHEDULER"]
    kwargs['lr_warmup_epochs'] = cfg["OPTIMIZATION"]["LR_WARMUP_EPOCHS"]
    kwargs['lr_decay_steps'] = cfg["OPTIMIZATION"]["LR_DECAY_STEPS"]
    kwargs['lr_decay_rate'] = cfg["OPTIMIZATION"]["LR_DECAY_RATE"]
    kwargs['lr_decay_min_lr'] = cfg["OPTIMIZATION"]["LR_DECAY_MIN_LR"]
    kwargs['patience'] = cfg["OPTIMIZATION"]["PATIENCE"]
    kwargs['accumulate_grad_batches'] = cfg["OPTIMIZATION"]["ACC_GRADIENT_STEPS"]
    kwargs['optimizer'] = cfg["OPTIMIZATION"]["OPTIMIZER"]

    # log
    kwargs['log_dir'] = cfg["LOG"]["PATH"]
    kwargs['exp_name'] = cfg["LOG"]["NAME"]
    kwargs['run'] = cfg["LOG"]["RUN"]

    # device
    kwargs['accelerator'] = "auto"
    kwargs['gpus'] = cfg["GPUS"]
    kwargs['precision'] = cfg["PRECISION"]
    kwargs['strategy'] = cfg["STRATEGY"]
    kwargs['precision'] = cfg["PRECISION"]
    kwargs['deterministic'] = True
    if str(kwargs['strategy']).lower() == "none":
        kwargs['strategy'] = None

    return kwargs


def configure_args(cfg_path, hparams):
    args = args_setup(cfg_path)
    # align other args with hparams args
    args = vars(args)
    args.update(hparams)
    args = argparse.Namespace(**args)

    return args
