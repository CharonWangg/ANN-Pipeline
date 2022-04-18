import torch
from repsim.kernels import Kernel, center, SquaredExponential, Laplace, Linear
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from pathlib import Path
from itertools import product
from tqdm.auto import tqdm, trange


def approx_dimensionality(x):
    """Compute effective dimensionality of size (samples, d) data 'x' using Participation Ratio.
    Answer will be less than or equal to d.
    """
    x = x.view(x.size(0), -1)
    mu = x.mean(dim=0)
    cov = torch.einsum('ia,ib->ab', x - mu, x - mu) / (x.size(0) - 1)
    _, s, _ = torch.svd(cov)
    return ((s.sum() * s.sum()) / (s * s).sum()).item()


def approx_dataloader_dimensionality(data_module):
    """Compute effective dimensionality of a dataloader using Participation Ratio.
    """
    data_module.setup(stage='predict')
    data_loader = data_module.predict_dataloader()
    with torch.no_grad():
        dim = approx_dimensionality(next(iter(data_loader))[0])
        print(f"Effective dimensionality of {data_module.dataset} pixels = {dim}")
    return dim


# TODO: support different kernel for path calculation
def configure_kernel(kernel_type, scale):
    """
    Configure kernel for path calculation.
    """
    if kernel_type == 'se':
        kernel = SquaredExponential(scale)
    elif kernel_type == 'laplace':
        kernel = Laplace(scale)
    elif kernel_type == 'linear':
        kernel = Linear()
    elif kernel_type == 'kernel_base':
        kernel = Kernel()
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    return kernel

def configure_
