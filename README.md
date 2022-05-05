# ANN Pipeline
Trying to be a comprehensive ANN training and analysis pipeline based on Pytorch Lightning
## Codebase Setting
- config.yaml: default config file and allowed be change by passing hparams(dict) into train or analysis
- train.sh: shell script used to call train_by_cmd.py
- train_by_dict.py: .py script used to train by passing hparams dict
- analysis.py: inference demo
- data: used to control different dataset class
    - cifar10
    - mnist
- model: used to control different model class
    - resnet_he: repulicated resnet from [Deeper and wider network](https://arxiv.org/pdf/2010.15327.pdf)
    - resnet_off: official torchvision resnet18 for cifar
    - others currently are not adopted
## Folder Setting
For ANN Analysis

    representation_paths
    ├── Template for ANN analysis        # this repo
    ├── data                             # folder for all data
    ├── models                           # folder for all trained models and models_log.csv used to manage all the info about models
    ├── results                          # folder for all results

For NLP:

    NLP tasks
    ├── Template for NLP                 # this repo
    ├── data                             # folder for all data
    ├── models                           # folder for all trained models

## TODO
For ANN Analysis:
- complete analysis pipeline
