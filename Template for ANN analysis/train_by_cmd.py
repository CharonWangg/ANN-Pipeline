"""
This main training entrance of the whole project.
"""
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
import pytorch_lightning.callbacks as plc
from src.utils import CSVModelCheckpoint
from src.model import ModelInterface
from src.data import DataInterface
from src.utils import args_setup, load_model_path_by_csv


def load_callbacks(args):
    callbacks = []
    if not isinstance(args.val_check_interval, int):
        # used to control early stopping
        callbacks.append(plc.EarlyStopping(
            monitor='val_acc',
            mode='max',
            strict=False,
            patience=args.patience,
            min_delta=0.001,
            check_finite=True,
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
    else:
        # used to save the model every epoch
        callbacks.append(CSVModelCheckpoint(
            hparams=vars(args),
            monitor='step',
            dirpath=args.save_dir + f'{args.exp_name}' + '/',
            filename='uid-' + \
                     f'model_name={args.model_name}-' + \
                     f'dataset={args.dataset}-' + \
                     '{epoch:d}-' + \
                     '{step:d}-' + \
                     '{val_acc:.3f}',
            save_top_k=-1,
            every_n_train_steps=args.val_check_interval,
            every_n_epochs=None,
            mode='max',
            verbose=True,
            save_last=True,
            save_on_train_epoch_end=False
         ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='step'))

    # Disable ProgressBar
    callbacks.append(plc.progress.TQDMProgressBar(
        refresh_rate=0,
    ))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    # torch.backends.cudnn.benchmark = True
    load_path = load_model_path_by_csv(args.save_dir + args.exp_name + "/", args)
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

    callbacks = load_callbacks(args)
    args.callbacks = callbacks

    # Tensorboard Logger
    # tb_logger = TensorBoardLogger(save_dir=args.log_dir, name=args.exp_name)
    comet_logger = CometLogger(api_key=args.comet_api_key,
                               save_dir=args.log_dir + args.exp_name + '/',
                               project_name="RepPaths",
                               rest_api_key=os.environ.get("COMET_REST_API_KEY"),
                               experiment_key=os.environ.get("COMET_EXPERIMENT_KEY"),
                               experiment_name=args.exp_name,
                               display_summary_level=0)
    args.logger = comet_logger
    trainer = Trainer.from_argparse_args(args)

    if args.val_check_interval != 1.0 and isinstance(args.val_check_interval, int):
        # uid = callbacks[0].cur_uid
        test_metrics = trainer.test(model, data_module, ckpt_path=args.ckpt_path)[0]
        # path = args.save_dir + args.exp_name + '/' + f"uid={uid}-model_name={args.model_name}-dataset={args.dataset}" \
        #                                              f"-epoch=0-step=0-val_acc={test_acc:.3f}.ckpt"
        # trainer.save_checkpoint(path)
        # print(f"Saved first checkpoint to: {path}!")
        callbacks[0].init_checkpoint(trainer, test_metrics)

    trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    trainer.test(model, data_module)


if __name__ == '__main__':
    cfg_path = 'configs/config.yaml'
    args = args_setup(cfg_path)
    main(args)
