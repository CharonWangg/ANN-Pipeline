# ANN Pipeline
Trying to be a comprehensive ANN training and analysis pipeline based on Pytorch Lightning
## Codebase Setting
- config.yaml: default config file and allowed be change by passing hparams(dict) into train or analysis
- train.py: train demo
- analysis.py: inference demo
- data: used to control different dataset class
- model: used to control different model class
## Folder Setting
- For ANN Analysis:
    representation_paths
    ├── Template for ANN analysis        # this repo
    ├── data                             # folder for all data
    ├── models                           # folder for all trained models
    ├── results                          # folder for all results

- For NLP:
    NLP tasks
    ├── Template for NLP                 # this repo
    ├── data                             # folder for all data
    ├── models                           # folder for all trained models
## TODO
For ANN Analysis:
- add more functions to train pipeline
- complete analysis pipeline
- formalize the interface between modules (model store format (csv?)/model load format(csv search?)/inference store format)
