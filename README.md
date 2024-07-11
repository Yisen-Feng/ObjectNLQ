# ObjectNLQ @ Ego4D Episodic Memory Challenge 2024

[Techical report](https://arxiv.org/abs/2406.15778)

This repo supports training and evaluation of the Ego4D-NLQ dataset. 

## Table of Contents

* [Preparation](#Preparation)
    * [Install dependencies](#Install-dependencies)
    * [Prepare offline data (e.g., features and files)](#Prepare-offline-data)
* [Code Overview](#Code-Overview)
* [Experiments](#Experiments)
  * [From-scratch](#Training-From-Scratch)
  * [Finetune](#Training-Finetune)
  * [Inference](#Inference)
  * [Ensemble](#Ensemble)
* [Acknowledgement](#Acknowledgements)


##  Preparation

### Install-dependencies 
* Follow [INSTALL.nd](install/INSTALL.md) for installing necessary dependencies and compiling the code.Torch version recommand >=1.8.0

### Prepare-offline-data
* GroundNLQ leverage the extracted egocentric InterVideo and EgoVLP features and CLIP textual token features, please refer to [GroundNLQ](https://github.com/houzhijian/GroundNLQ).
* Download the object data, including files. Will release this data soon.

## Code-Overview

* ./libs/core: Parameter default configuration module.
  * ./configs: Parameter file.
* ./ego4d_data: the annotation data.
* ./tools: Scripts for running,train_ego4d_finetune_head_*.sh are for finetune,while train_ego4d_* are for scratch.
* ./libs/datasets: Data loader and IO module.
* ./libs/modeling: Our main model with all its building blocks.
* ./libs/utils: Utility functions for training, inference, and postprocessing.

##  Experiments
We adopt distributed data parallel [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) and 
fault-tolerant distributed training with [torchrun](https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html).

### Training-From-Scratch
Training can be launched by running the following command:
```
bash <tools/*,scripts> CONFIG_FILE EXP_ID CUDA_DEVICE_ID
```
where `CONFIG_FILE` is the config file for model/dataset hyperparameter initialization,
`EXP_ID` is the model output directory name defined by yourself, `CUDA_DEVICE_ID` is cuda device id.
The checkpoints and other experiment log files will be written into `<output_folder>/OUTPUT_PATH`, output_folder is defined in the config file. 
- ObjectNLQ：ego4d_nlq_v2_object_encoderca_1e-4.yaml

### Training-Finetune
Training can be launched by running the following command:
```
bash tools/train ego4d_finetune_head_onegpu.sh CONFIG_FILE RESUME_PATH OUTPUT_PATH CUDA_DEVICE_ID
```
where `RESUME_PATH` is the path of the pretrained model weights.

The config file is the same as scratch.

### Inference
Once the model is trained, you can use the following commands for inference:
```
python eval_nlq.py CONFIG_FILE CHECKPOINT_PATH -gpu CUDA_DEVICE_ID <--save>
```
where `CHECKPOINT_PATH` is the path to the saved checkpoint,`save` is for controling the output . 


* The results (Recall@K at IoU = 0.3 or 0.5) on the val. set should be similar to the performance of the below table reported in the main report.

|  Method   |    Dataset   | R@1 IoU=0.3 | R@1 IoU=0.5 | R@5 IoU=0.3 | R@5 IoU=0.5   |
|--------------------------|-------------|-------------|-------------|---------------|
| ObjectNLQ |     NLQ      | 28.43       | 19.95       | 56.06       | 42.09         | 
| ObjectNLQ |  GoalStep    | 28.34       | 24.08       | 57.03       | 50.39         | 


### Ensemble
We conduct post-model prediction ensemble to enhance performance for leaderboard submission.
The actual command used in the experiments is
```
python ensemble.py
```
or 
```
python ensemble_more.py
```

## Acknowledgements
This code is inspired by [GroundNLQ](https://github.com/houzhijian/GroundNLQ). 
We use the same video and text feature as GroundNLQ. 
We thank the authors for their awesome open-source contributions. 