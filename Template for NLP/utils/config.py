import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
import argparse
import os
from pathlib2 import Path

# TODO: load model in uid format by searching its hparams in a csv file
def load_model_path_by_csv(root, hparams=None):
    pass


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

    if Path(root).is_file():
        return root

    # in case epoch is not set
    if "epoch" not in hparams:
        hparams["epoch"] = "*"
    # match the hparams to the file name
    files = [str(i) for i in list(Path(root).iterdir())
             if f'model_name={hparams["model_name"]}'.lower() in str(i) and
             f'dataset={hparams["dataset"]}'.lower() in str(i) and
             f'epoch={hparams["epoch"]}'.lower() in str(i) and
             f'lr={hparams["lr"]}'.lower() in str(i) and
             f'weight_decay={hparams["weight_decay"]}'.lower() in str(i) and
             f'train_batch_size={hparams["train_batch_size"]}'.lower() in str(i) and
             f'l1={hparams["l1"]}'.lower() in str(i) and
             f'l2={hparams["l2"]}'.lower() in str(i) and
             f'dropout={hparams["dropout"]}'.lower() in str(i) and
             f'optimizer={hparams["optimizer"]}'.lower() in str(i) and
             f'loss={hparams["loss"]}'.lower() in str(i) and
             f'seed={hparams["seed"]}'.lower() in str(i)
             in str(i).lower()]

    if not files:
        return None
    else:
        print('Loading model from {}'.format(files[-1]))
        return files[-1]

def args_setup(cfg_path='./config.yaml'):
    cfg = yaml.safe_load(open(cfg_path))
    parser = ArgumentParser()
    # init
    parser.add_argument('--cfg', type=str, default=cfg_path, help='config file path')
    parser.add_argument('--seed', default=cfg["SEED"], type=int)

    # data
    parser.add_argument('--dataset', default=cfg["DATA"]["DATASET"], type=str)
    parser.add_argument('--train_path', default=cfg["DATA"]["TRAIN_PATH"], type=str)
    parser.add_argument('--valid_path', default=cfg["DATA"]["VALID_PATH"], type=str)
    parser.add_argument('--test_path', default=cfg["DATA"]["TEST_PATH"], type=str)
    parser.add_argument('--num_train', default=cfg["DATA"]["NUM_TRAIN"], type=int)
    parser.add_argument('--num_test', default=cfg["DATA"]["NUM_TEST"], type=int)
    parser.add_argument('--num_valid', default=cfg["DATA"]["NUM_VALID"], type=int)
    parser.add_argument('--train_batch_size', default=cfg["DATA"]["TRAIN_BATCH_SIZE"], type=int)
    parser.add_argument('--valid_batch_size', default=cfg["DATA"]["VALID_BATCH_SIZE"], type=int)
    parser.add_argument('--test_batch_size', default=cfg["DATA"]["TEST_BATCH_SIZE"], type=int)
    parser.add_argument('--train_size', default=cfg["DATA"]["TRAIN_SIZE"], type=int)
    parser.add_argument('--class_num', default=cfg["DATA"]["CLASS_NUM"], type=int)
    parser.add_argument('--num_workers', default=cfg["DATA"]["NUM_WORKERS"], type=int)

    # data augmentation
    parser.add_argument('--aug', default=cfg["DATA"]["AUG"], type=bool)
    parser.add_argument('--aug_prob', default=cfg["DATA"]["AUG_PROB"], type=float)

    # tokenizer
    parser.add_argument('--tokenizer_name', default=cfg["TOKENIZE"]["NAME"], type=str)
    parser.add_argument('--max_seq_len', default=cfg["TOKENIZE"]["MAX_SEQ_LEN"], type=int)
    parser.add_argument('--doc_stride', default=cfg["TOKENIZE"]["DOC_STRIDE"], type=int)

    # train
    parser.add_argument('--lr', default=cfg["OPTIMIZATION"]["LR"], type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', default=cfg["OPTIMIZATION"]["LR_SCHEDULER"],
                        choices=['step', 'cosine', 'constant', 'cyclic', 'plateau'], type=str)
    parser.add_argument('--lr_decay_steps', default=cfg["OPTIMIZATION"]["LR_DECAY_STEPS"], type=int)
    parser.add_argument('--lr_decay_rate', default=cfg["OPTIMIZATION"]["LR_DECAY_RATE"], type=float)
    parser.add_argument('--lr_decay_min_lr', default=cfg["OPTIMIZATION"]["LR_DECAY_MIN_LR"], type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--model_name', default=cfg["MODEL"]["NAME"], type=str)
    parser.add_argument('--loss', default=cfg["OPTIMIZATION"]["LOSS"],
                        choices=["cross_entropy", "binary_cross_entropy", "l1", "l2", "diy"], type=str)
    parser.add_argument('--margin', default=cfg["OPTIMIZATION"]["MARGIN"], type=float)
    parser.add_argument('--weight_decay', default=cfg["OPTIMIZATION"]["WEIGHT_DECAY"], type=float)
    parser.add_argument('--momentum', default=cfg["OPTIMIZATION"]["MOMENTUM"], type=float)
    parser.add_argument('--optimizer', default=cfg["OPTIMIZATION"]["OPTIMIZER"], choices=["Adam", "SGD", "RMSprop", "DIY"],
                        type=str)
    parser.add_argument('--l1', default=cfg["MODEL"]["L1"], type=float)
    parser.add_argument('--l2', default=cfg["MODEL"]["L2"], type=float)
    parser.add_argument('--patience', default=cfg["OPTIMIZATION"]["PATIENCE"], type=int)
    parser.add_argument('--log_dir', default=cfg["LOG"]["PATH"], type=str)
    parser.add_argument('--save_dir', default=cfg["MODEL"]["SAVE_DIR"], type=str)

    # Model Hyperparameters
    parser.add_argument('--encoder_name', default=cfg["MODEL"]["ENCODER"]["NAME"], type=str)
    parser.add_argument('--encoder_hidden_size', default=cfg["MODEL"]["ENCODER"]["HIDDEN_SIZE"], type=int)
    parser.add_argument('--arg_comp_hidden_size', default=cfg["MODEL"]["ARG_COMP"]["HIDDEN_SIZE"], type=int)
    parser.add_argument('--arg_comp_output_size', default=cfg["MODEL"]["ARG_COMP"]["OUTPUT_SIZE"], type=int)
    parser.add_argument('--event_comp_hidden_size', default=cfg["MODEL"]["EVENT_COMP"]["HIDDEN_SIZE"], type=int)
    parser.add_argument('--event_comp_output_size', default=cfg["MODEL"]["EVENT_COMP"]["OUTPUT_SIZE"], type=int)
    parser.add_argument('--dropout', default=cfg["MODEL"]["DROPOUT"], type=float)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=cfg["OPTIMIZATION"]["MAX_EPOCHS"])
    parser.set_defaults(accumulate_grad_batches=cfg["OPTIMIZATION"]["ACC_GRADIENT_STEPS"])
    parser.set_defaults(accelerator="auto")
    parser.set_defaults(devices=cfg["DEVICES"])
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    # List Arguments
    args.mean_sen = cfg["DATA"]["IMG_MEAN"]
    args.std_sen = cfg["DATA"]["IMG_STD"]

    return args

def configure_args(cfg_path, hparams):
    args = args_setup(cfg_path)
    # align other args with hparams args
    args = vars(args)
    args.update(hparams)
    args = argparse.Namespace(**args)

    return args
