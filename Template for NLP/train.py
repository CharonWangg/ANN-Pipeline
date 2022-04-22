"""
This main training entrance of the whole project.
"""
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from model import ModelInterface
from data import DataInterface
from utils.config import args_setup, load_model_path_by_hparams, configure_args


def load_callbacks(args):
    callbacks = []
    # used to control early stopping
    callbacks.append(plc.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=args.patience,
        min_delta=0.001,
        verbose=True
    ))
    # used to save the best model
    callbacks.append(plc.ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.save_dir,
        filename=f'model_name={args.model_name}-' + \
                 f'dataset={args.dataset}-' + \
                 f'lr={args.lr}-' + \
                 f'wd={args.weight_decay}-' + \
                 f'batch={args.train_batch_size}-' + \
                 f'l1={args.l1}-' + \
                 f'l2={args.l2}-' + \
                 f'dropout={args.dropout}-' + \
                 f'opt={args.optimizer}-' + \
                 f'loss={args.loss}-' + \
                 f'seed={args.seed}-' + \
                 '{epoch}-' + \
                 '{val_loss}-',
        save_top_k=1,
        mode='min',
        verbose=True,
        save_last=False
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))

    # callbacks.append(plc.progress.TQDMProgressBar(
    #     refresh_rate=0,
    # ))
    return callbacks


def main(args):
    pl.seed_everything(args.seed, workers=True)
    load_path = load_model_path_by_hparams(args.save_dir, args)
    print(f'load_path: {load_path}')
    data_module = DataInterface(vars(args))

    if load_path is None:
        model = ModelInterface(vars(args))
    else:
        model = ModelInterface(vars(args))
        args.resume_from_checkpoint = load_path
        print('Found checkpoint, stop training.')

    # # If you want to change the logger's saving folder
    # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)

    callbacks = load_callbacks(args)
    args.callbacks = callbacks
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)


if __name__ == '__main__':

    cfg_path = './config.yaml'
    # hparams used for training
    hparams = {}
    # import modified args
    args = configure_args(cfg_path, hparams)
    # train
    main(args)
