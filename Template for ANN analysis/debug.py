import pandas as pd

from pipeline.src.utils import yaml_to_kwargs


def get_model_size(model):
    para_num = sum([p.numel() for p in model.parameters()])
    # para_size: 参数个数 * 每个4字节(float32) / 1024 / 1024，单位为 MB
    para_size = para_num * 4 / 1024 / 1024
    return para_size


if __name__ == '__main__':
    df = pd.read_csv("/data2/charon/reppaths/models/resnet_he_cifar10_width_repro/models_log.csv")
    df["val_check_interval"] = 1.0
    df.to_csv("/data2/charon/reppaths/models/resnet_he_cifar10_width_repro/models_log.csv")
    df = pd.read_csv("/data2/charon/reppaths/models/resnet_he_cifar10_width_repro/models_log.csv")
    print(df["val_check_interval"])


