"""
This main training entrance of the whole project.
"""
import torch
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning.callbacks as plc
from model import ModelInterface
from data import DataInterface
from utils.config import args_setup, load_model_path_by_hparams, configure_args


def load_callbacks(args):
    callbacks = []
    # used to control early stopping
    callbacks.append(plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=args.patience,
        min_delta=0.001,
        verbose=True
    ))
    # used to save the best model
    callbacks.append(plc.ModelCheckpoint(
        monitor='val_acc',
        dirpath=args.save_dir, # + '/' + f'{args.dataset}' + '/',
        filename=f'{args.model_name}-' + \
                 f'{args.dataset}-' + \
                 f'{args.depth}-' + \
                 f'{args.width_multiplier}-' + \
                 f'{args.activation}-' + \
                 f'{args.dropout}-' + \
                 f'{args.lr}-' + \
                 f'{args.weight_decay}-' + \
                 f'{args.train_batch_size}-' + \
                 f'{args.l1}-' + \
                 f'{args.l2}-' + \
                 f'{args.optimizer}-' + \
                 f'{args.loss}-' + \
                 f'{args.seed}-' + \
                 '{val_acc:.3f}',
        save_top_k=1,
        mode='max',
        verbose=True,
        save_last=False
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))

    # Disable ProgressBar
    # callbacks.append(plc.progress.TQDMProgressBar(
    #     refresh_rate=0,
    # ))
    return callbacks


def main(args):
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.benchmark = True
    load_path = load_model_path_by_hparams(args.save_dir, args)
    print(f'load_path: {load_path}')
    data_module = DataInterface(vars(args))

    if load_path is None:
        model = ModelInterface(vars(args))
    else:
        model = ModelInterface(vars(args))
        # print('Found checkpoint, stop training.')
        # return 0

    # If you want to change the logger's saving folder
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.exp_name)
    args.logger = logger

    callbacks = load_callbacks(args)
    args.callbacks = callbacks
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module, ckpt_path=load_path)


if __name__ == '__main__':
    cfg_path = './config.yaml'
    # hparams used for training
    hparams = {"dataset": "cifar10", "model_name": "resnet_off",
               "depth": 20, "width_multiplier": 1.0,
               "activation": "relu", "dropout": 0.0,
               "lr": 0.01, "optimizer": "sgd", "momentum": 0.9,
               "lr_scheduler": "cosine", "lr_decay_steps": 1,
               "lr_warmup_epochs": 5,
               "train_batch_size": 128, "weight_decay": 5e-4,
               "l1": 0.0, "l2": 0.0, "max_epochs": 200, "epoch": 0,
               "aug": True, "aug_prob": 0.5,
               "seed": 42, "devices": [2], "precision": 16}
    # import modified args
    args = configure_args(cfg_path, hparams)
    # train
    print(f'hparams: {args}')
    main(args)
