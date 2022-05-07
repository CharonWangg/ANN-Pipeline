import json

import pandas as pd

from model.common import resnet_cifar


def get_model_size(model):
    para_num = sum([p.numel() for p in model.parameters()])
    # para_size: 参数个数 * 每个4字节(float32) / 1024 / 1024，单位为 MB
    para_size = para_num * 4 / 1024 / 1024
    return para_size


if __name__ == '__main__':
    df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
    print(df)
    df = df.append({"A": 3}, ignore_index=True)
    print(df)
