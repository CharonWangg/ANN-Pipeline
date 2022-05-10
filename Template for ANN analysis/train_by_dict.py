"""
This main training entrance of the whole project.
"""
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pipeline.src.utils.log_util import CSVModelCheckpoint
import pytorch_lightning.callbacks as plc
from pipeline.src.model import ModelInterface
from pipeline.src.data import DataInterface
from pipeline.src.utils.config_util import load_model_path_by_csv, configure_args


def load_callbacks(args):
    callbacks = []
    # used to control early stopping
    callbacks.append(plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        strict=False,
        patience=args.patience,
        min_delta=0.001,
        verbose=True
    ))
    # used to save the best model
    callbacks.append(CSVModelCheckpoint(
        hparams=vars(args),
        monitor='val_acc',
        dirpath=args.save_dir + f'{args.exp_name}' + '/',
        filename='uid-' + \
                 f'model_name={args.model_name}-' + \
                 f'dataset={args.dataset}-' + \
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
    pl.seed_everything(args.seed)
    # torch.backends.cudnn.benchmark = True
    load_path = load_model_path_by_csv(args.save_dir, args)
    # print(f'load_path: {load_path}')
    data_module = DataInterface(**vars(args))

    if load_path is None:
        model = ModelInterface(**vars(args))
        args.ckpt_path = None
    else:
        model = ModelInterface(**vars(args)).load_from_checkpoint(load_path)
        args.ckpt_path = load_path
        # print('Found checkpoint, stop training.')
        # return 0

    # If you want to change the logger's saving folder
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.exp_name)
    args.logger = logger

    callbacks = load_callbacks(args)
    args.callbacks = callbacks
    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    trainer.test(model, data_module)


if __name__ == '__main__':
    cfg_path = 'configs/config.yaml'
    # hparams used for training
    hparams = {"dataset": "cifar10", "model_name": "resnet_he",
               "depth": 20, "width_multiplier": 1.0,
               "activation": "relu", "dropout": 0.0,
               "lr": 0.12, "optimizer": "sgd", "momentum": 0.9,
               "lr_scheduler": "cifar", "lr_decay_rate": 0.1,
               "lr_decay_steps": 1, "lr_warmup_epochs": 5,
               "train_batch_size": 128, "weight_decay": 5e-4,
               "l1": 0.0, "l2": 0.0, "max_epochs": 100, "epoch": 0,
               "aug": True, "aug_prob": 0.5,
               "seed": 7, "gpus": "0", "strategy": None,
               "precision": 32}
    # import modified args
    args = configure_args(cfg_path, hparams)
    # train
    # print(f'hparams: {args}')
    main(args)
