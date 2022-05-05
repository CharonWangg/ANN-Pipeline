"""
This main training entrance of the whole project.
"""
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning.callbacks as plc
from utils.log_util import CSVModelCheckpoint
from model import ModelInterface
from data import DataInterface
from utils.config_util import args_setup, load_model_path_by_csv, configure_args


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
    data_module = DataInterface(vars(args))

    if load_path is None:
        model = ModelInterface(vars(args))
        args.ckpt_path = None
    else:
        model = ModelInterface(vars(args))
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
    trainer.test(model, data_module, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    cfg_path = './config.yaml'
    args = args_setup(cfg_path)
    main(args)
