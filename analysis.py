import os
import pytorch_lightning as pl
from argparse import ArgumentParser

import yaml
from progressbar import ProgressBar
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from model import ModelInterface
from data import DataInterface
from utils import load_model_path_by_hparams
from utils import args_setup, configure_analysis_args
from utils import approx_dataloader_dimensionality

def main(args):
    pl.seed_everything(args.seed, workers=True)
    load_path = load_model_path_by_hparams(args.save_dir, args)
    print(f'load_path: {load_path}')
    data_module = DataInterface(vars(args))

    if load_path is None:
        model = ModelInterface(vars(args))
        print('Can\'t Found checkpoint, using un-trained model instead..')
    else:
        model = ModelInterface(vars(args))
        args.ckpt_path = load_path
        print('Found checkpoint, start analyzing..')

    trainer = Trainer.from_argparse_args(args)
    # check the approximate dimension of pixels
    dim = approx_dataloader_dimensionality(data_module)
    # inference
    layers_output = trainer.predict(model, data_module)  # {f"h_{i}": torch.Tensor}
    #


if __name__ == '__main__':
    cfg_path = './config.yaml'
    hparams = {"dataset": "cifar10", "model_name": "resnet", "hidden_size": 64, "num_hidden_layers": 2,
               "activation": "relu", "dropout": 0.3, "lr": 0.001, "optimizer": "adam", "batch_size": 50,
               "regularization": "none", "regularization_weight": 0.001, "max_epochs": 10, "seed": 42}
    args = configure_analysis_args(cfg_path, hparams)

    # analysis
    main(args)
