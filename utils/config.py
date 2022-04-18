import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
import argparse
import os
from pathlib2 import Path


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
    # concat the root and dataset name
    root = os.path.join(root, hparams["dataset"])
    # match the hparams to the file name
    files = [str(i) for i in list(Path(root).iterdir())
             if "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(
            hparams["model_name"],
            hparams["dataset"],
            hparams["hidden_size"][0],
            hparams["num_hidden_layers"],
            hparams["activation"],
            hparams["dropout"],
            hparams["lr"],
            hparams["train_batch_size"],
            hparams["regularization"],
            hparams["regularization_weight"],
            hparams["max_epochs"],
            hparams["optimizer"],
            hparams["loss"],
            hparams["seed"],
        ).lower() in str(i).lower()]

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


def args_setup(cfg_path='./config.yaml'):
    cfg = yaml.safe_load(open(cfg_path))
    parser = ArgumentParser()
    # init
    parser.add_argument('--cfg', type=str, default=cfg_path, help='config file path')
    # parser.add_argument('--stage', type=str, default="fit", choices=["fit", "test"])
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--seed', default=cfg["SEED"], type=int)
    # data
    parser.add_argument('--train_batch_size', default=cfg["DATA"]["TRAIN_BATCH_SIZE"], type=int)
    parser.add_argument('--valid_batch_size', default=cfg["DATA"]["VALID_BATCH_SIZE"], type=int)
    parser.add_argument('--test_batch_size', default=cfg["DATA"]["TEST_BATCH_SIZE"], type=int)
    parser.add_argument('--train_size', default=cfg["DATA"]["TRAIN_SIZE"], type=int)
    parser.add_argument('--class_num', default=cfg["DATA"]["CLASS_NUM"], type=int)
    parser.add_argument('--num_workers', default=cfg["DATA"]["NUM_WORKERS"], type=int)
    # data augmentation
    parser.add_argument('--aug', default=cfg["DATA"]["AUG"], type=bool)
    parser.add_argument('--aug_prob', default=cfg["DATA"]["AUG_PROB"], type=float)
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
    parser.add_argument('--dataset', default=cfg["DATA"]["DATASET"], type=str)
    parser.add_argument('--data_dir', default=cfg["DATA"]["DATA_PATH"], type=str)
    parser.add_argument('--model_name', default=cfg["MODEL"]["NAME"], type=str)
    parser.add_argument('--loss', default=cfg["OPTIMIZATION"]["LOSS"],
                        choices=["cross_entropy", "binary_cross_entropy", "l1", "l2"], type=str)
    parser.add_argument('--weight_decay', default=cfg["OPTIMIZATION"]["WEIGHT_DECAY"], type=float)
    parser.add_argument('--momentum', default=cfg["OPTIMIZATION"]["MOMENTUM"], type=float)
    parser.add_argument('--optimizer', default=cfg["OPTIMIZATION"]["OPTIMIZER"], choices=["Adam", "SGD", "RMSprop"],
                        type=str)
    parser.add_argument('--regularization', default=cfg["MODEL"]["REGULARIZATION"], type=str)
    parser.add_argument('--regularization_weight', default=cfg["MODEL"]["REGULARIZATION_WEIGHT"], type=float)
    parser.add_argument('--patience', default=cfg["OPTIMIZATION"]["PATIENCE"], type=int)
    parser.add_argument('--log_dir', default=cfg["LOG"]["PATH"], type=str)
    parser.add_argument('--save_dir', default=cfg["MODEL"]["SAVE_DIR"], type=str)

    # Model Hyperparameters
    parser.add_argument('--input_size', default=cfg["MODEL"]["INPUT_SIZE"], type=int)
    parser.add_argument('--output_size', default=cfg["MODEL"]["OUTPUT_SIZE"], type=int)
    parser.add_argument('--num_hidden_layers', default=cfg["MODEL"]["NUM_HIDDEN_LAYERS"], type=int)
    parser.add_argument('--dropout', default=cfg["MODEL"]["DROPOUT"], type=float)
    parser.add_argument('--activation', default=cfg["MODEL"]["ACTIVATION"],
                        choices=["relu", "tanh", "sigmoid", "leaky_relu", "prelu"], type=str)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=cfg["OPTIMIZATION"]["MAX_EPOCHS"])
    parser.set_defaults(accumulate_grad_batches=cfg["OPTIMIZATION"]["ACC_GRADIENT_STEPS"])
    parser.set_defaults(accelerator="auto")
    parser.set_defaults(devices=[0])
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    # List Arguments
    args.mean_sen = cfg["DATA"]["IMG_MEAN"]
    args.std_sen = cfg["DATA"]["IMG_STD"]
    args.hidden_size = configure_hidden_size(cfg["MODEL"]["HIDDEN_SIZE"], cfg["MODEL"]["NUM_HIDDEN_LAYERS"])
    args.projection_size = configure_hidden_size(cfg["MODEL"]["PROJECTION_SIZE"], cfg["MODEL"]["NUM_HIDDEN_LAYERS"])
    print(args)  # Print args
    return args


def configure_analysis_args(cfg_path, hparams):
    args = args_setup(cfg_path)
    # Set the hidden size to the same as the training
    hparams["hidden_size"] = configure_hidden_size(hparams["hidden_size"], hparams["num_hidden_layers"])
    # align other args with hparams args
    args = vars(args)
    args.update(hparams)
    args = argparse.Namespace(**args)
    return args
