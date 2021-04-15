# fsdl-active-learning

Comparing different active learning strategies for image classification (FSDL course 2021 capstone project)

## Introduction

This repository builds upon the template of **lab 08** of the [Full Stack Deep Learning Spring 2021 labs](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs) and extends it with a new dataset, model and active learning strategies.

## Relevant Changes Compared to Lab Template

[text_recognizer/data/droughtwatch.py](./text_recognizer/data/droughtwatch.py): Downloads data from the [W&B Drought Prediction Benchmark](https://github.com/wandb/droughtwatch) and converts it to HDF5 format which can be used by PyTorch for training and inference.

[text_recognizer/models/resnet_classifier.py](./text_recognizer/models/resnet_classifier.py): Implements a PyTorch ResNet model for image classification, with adapted preprocessing steps (image resizing) and class outputs (4 instead of 1000). The model can be used for transfer learning on the drought prediction data.

## Quickstart

### Local

```bash
git pull [repo-url] # clone from git
cd [folder]

make conda-update # creates a conda env with the base packages
conda activate fsdl-active-learning-2021 # activates the conda env
make pip-tools # installs required pip packages inside the conda env

python training/run_experiment.py --max_epochs=1 --num_workers=4 --data_class=DroughtWatch --model_class=ResnetClassifier # start training
```

### Google Colab

Refer to the example notebook under [notebooks/clone_repo_and_train.ipynb](./notebooks/clone_repo_and_train.ipynb).
