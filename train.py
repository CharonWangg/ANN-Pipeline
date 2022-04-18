""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and
    DInterface can be seen as transparent to all your args.
"""
import pytorch_lightning as pl

from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from model import ModelInterface
from data import DataInterface
from pipeline.train.utils.config import args_setup, load_model_path_by_hparams


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
        dirpath=args.save_dir + '/' + f'{args.dataset}' + '/',
        filename=f'{args.model_name}-' + \
                 f'{args.dataset}-' + \
                 f'{args.hidden_size[0]}-' + \
                 f'{args.num_hidden_layers}-' + \
                 f'{args.activation}-' + \
                 f'{args.dropout}-' + \
                 f'{args.lr}-' + \
                 f'{args.train_batch_size}-' + \
                 f'{args.regularization}-' + \
                 f'{args.regularization_weight}-' + \
                 f'{args.max_epochs}-' + \
                 f'{args.optimizer}-' + \
                 f'{args.loss}-' + \
                 f'{args.seed}-' + \
                 '{val_acc:.3f}',
        save_top_k=1,
        mode='max',
        verbose=True,
        save_last=True
    ))

    # if args.lr_scheduler:
    #     callbacks.append(plc.LearningRateMonitor(
    #         logging_interval='epoch'))

    callbacks.append(plc.progress.TQDMProgressBar(
        refresh_rate=0,
    ))
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
        return 0

    # # If you want to change the logger's saving folder
    # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)

    callbacks = load_callbacks(args)
    args.callbacks = callbacks
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    cfg_path = './config.yaml'
    args = args_setup(cfg_path)
    main(args)
