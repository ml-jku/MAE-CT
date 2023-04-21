# MAE-CT: Masked Autoencoder Contrastive Tuning

Pytorch implementation of **M**asked **A**uto**E**ncoder **C**ontrastive **T**uning (MAE-CT) 
from our paper [Contrastive Tuning: A Little Help to Make Masked Autoencoders Forget](https://arxiv.org/abs/2304.10520).


# Pretrained Checkpoints

## MAE reimplementation


|Encoder|Pretrain|Probing|k-NN|
|:---:|:---:|:---:|:---:|
|ViT-B/16|hp|66.7|51.1|
|ViT-L/16|hp|75.9|60.6|
|ViT-H/16|hp|78.0|61.1|


## MAE-CT

|Encoder|Pretrain|Probing|k-NN|
|:---:|:---:|:---:|:---:|
|ViT-B/16|hp|73.5|64.1|
|ViT-L/16|hp|80.2|78.0|
|ViT-H/16|hp|81.5|79.4|

## MAE-CT<sub>*aug*</sub>

|Encoder|Pretrain|Probing|k-NN|
|:---:|:---:|:---:|:---:|
|ViT-B/16|hp|73.5|64.1|
|ViT-L/16|hp|80.2|78.0|
|ViT-H/16|hp|81.5|79.4|




# Setup
Setup a conda environment: `conda env create --file environment_linux.yml --name maect`

We use [FlashAttention](https://github.com/HazyResearch/flash-attention)
([paper](https://arxiv.org/abs/2205.14135)) to greatly accelerate computations. 
We recommend to install it, but this repo can be used (without modification)
without FlashAttention.

## Configuration of dataset paths and environment specific things
- `cp template_static_config.yaml static_config.yaml`
- adjust values to your setup

For low-shot evaluations, we use the official splits from
[SimCLRv2](https://github.com/google-research/simclr/tree/master/imagenet_subsets)
and [MSN](https://github.com/facebookresearch/msn).

## [Optional] configure weights and biases
This repo uses [Weights & Biases](https://wandb.ai) for experiment tracking, but offers an alternative
in case you do not want to use it. By default W&B logging is disabled via the `default_wandb_mode: disabled`
configuration in the `static_config.yaml`. You can enable it via `static_config.yaml` 
or via the CLI `--wandb_mode online`. 

If you enabled W&B logging, the W&B entity and project will (by default) be fetched from the `wandb_config.yaml`.
You can create this via `cp template_wandb_config.yaml wandb_config.yaml` and adjust the values to your setup.

# Run

- `--hp <YAML>` e.g. `--hp hyperparams.yaml` define what to run
- `--devices <DEVICES>` e.g. `--devices 0` to run on GPU0 or `--devices 0,1,2,3` to run on 4 GPUs

