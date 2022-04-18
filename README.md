# ANN Pipeline
Trying to be a comprehensive ANN training and analysis pipeline based on Pytorch Lightning
## Codebase Setting
- config.yaml: default config file and allowed be change by passing hparams(dict) into train or analysis
- train.py: train demo
- analysis.py: inference demo
## Folder Setting
    representation_paths
    ├── ANN-Pipeline            # this repo
    ├── data                    # folder for all data
    ├── models                  # folder for all trained models
## TODO
- add more functions to train pipeline
- complete analysis pipeline
- formalize the interface between modules (model store format (csv?)/model load format(csv search?)/inference store format)
- 
