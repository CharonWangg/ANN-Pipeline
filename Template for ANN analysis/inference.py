import pytorch_lightning as pl
from pytorch_lightning import Trainer

from model import ModelInterface
from data import DataInterface
from utils import load_model_path_by_hparams
from utils import args_setup, configure_args, load_model_path_by_csv
from utils import approx_dataloader_dimensionality


def main(args, hparams):
    pl.seed_everything(args.inference_seed)
    load_path = load_model_path_by_csv(args.save_dir, hparams, mode='inference')
    print(f'load_path: {load_path}')
    data_module = DataInterface(vars(args))

    if load_path is None:
        model = ModelInterface(vars(args))
        print('Can\'t Found checkpoint, using un-trained model instead..')
    else:
        model = ModelInterface(vars(args))
        # args.ckpt_path = load_path
        print('Found checkpoint, start analyzing..')

    trainer = Trainer.from_argparse_args(args)
    # check the approximate dimension of pixels
    dim = approx_dataloader_dimensionality(data_module)
    # inference
    layers_output = trainer.predict(model, data_module, ckpt_path=load_path)  # {f"h_{i}": torch.Tensor}
    return layers_output


if __name__ == '__main__':
    cfg_path = 'config.yaml'
    hparams = {"dataset": "cifar10", "model_name": "resnet_he",
               "depth": 20, "width_multiplier": 1.0,
               "run": 0,
               "inference_seed": 42, "devices": [2]}

    args = configure_args(cfg_path, hparams)
    # inference
    '''
    {"pixels"(raw images), "h_{i}"(hidden_state),
     "h_o"(output_state), "h_p"(h_o after softmax)
     "labels"(raw label) } Unranked
    '''
    layers_output = main(args, hparams)
    print(layers_output)
