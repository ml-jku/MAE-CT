# MAE-CT: Masked Autoencoder Contrastive Tuning

Pytorch implementation of **M**asked **A**uto**E**ncoder **C**ontrastive **T**uning (MAE-CT)
from our paper <br/>
[Contrastive Tuning: A Little Help to Make Masked Autoencoders Forget](https://arxiv.org/abs/2304.10520).

<p align="center">
<img width="31.5%" alt="maect_schematic" src="https://github.com/ml-jku/MAE-CT/blob/137d969be4c78d156465bb18d09f52d3c762114f/.github/schematic_contrastive_tuning.svg">
<img width="67%" alt="lowshot_vitl" src="https://github.com/ml-jku/MAE-CT/blob/2ff19e68df9c3a1a7cb17a1846ac0d937359392c/.github/lowshot_aug_L_white.svg">
</p>

This repository provides:

- Pretrained checkpoints for 
  [MAE](https://github.com/ml-jku/MAE-CT#mae-reimplementation), 
  [MAE-CT](https://github.com/ml-jku/MAE-CT#mae-ct) and 
  [MAE-CT<sub>*aug*</sub>](https://github.com/ml-jku/MAE-CT#mae-ctaug)
  - [Hyperparameters](https://github.com/ml-jku/yamls) for all checkpoints
  - [Linear probes](https://github.com/ml-jku/probes) trained on the respective checkpoints + 
    [logs](https://github.com/ml-jku/logs/probe)
- Instructions to generate low-shot datasets for evaluation
- Instructions on how to use our models as backbone

# Pretrained Checkpoints


## MAE reimplementation

|Weights|Pretrain|Probe|Probe|k-NN|
|:---:|:---:|:---:|:---:|:---:|
|[ViT-B/16](https://ml.jku.at/research/maect/download/mae_reimpl_base16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/06326017fa605a9b650da36b1f63dd0376e4bd28/yamls/mae/base16.yaml)|[66.7](https://github.com/ml-jku/MAE-CT/blob/0f2fa9373b30ed4d29cf6093ddd2821d179f771b/probes/mae_reimpl_base16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/1330ad99aab7cfe1e7031db33711d4011bc2c1b8/logs/probe/mae_base16.th)|51.1|
|[ViT-L/16](https://ml.jku.at/research/maect/download/mae_reimpl_large16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/06326017fa605a9b650da36b1f63dd0376e4bd28/yamls/mae/large16.yaml)|[75.9](https://github.com/ml-jku/MAE-CT/blob/0f2fa9373b30ed4d29cf6093ddd2821d179f771b/probes/mae_reimpl_large16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/1330ad99aab7cfe1e7031db33711d4011bc2c1b8/logs/probe/mae_large16.th)|60.6|
|[ViT-H/16](https://ml.jku.at/research/maect/download/mae_reimpl_huge16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/06326017fa605a9b650da36b1f63dd0376e4bd28/yamls/mae/huge16.yaml)|[78.0](https://github.com/ml-jku/MAE-CT/blob/0f2fa9373b30ed4d29cf6093ddd2821d179f771b/probes/mae_reimpl_huge16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/1330ad99aab7cfe1e7031db33711d4011bc2c1b8/logs/probe/mae_huge16.th)|61.1|
|ViT-H/14|[original](https://github.com/facebookresearch/mae)|[79.1](https://github.com/ml-jku/MAE-CT/blob/0f2fa9373b30ed4d29cf6093ddd2821d179f771b/probes/mae_huge14.th)|[log](https://github.com/ml-jku/MAE-CT/blob/1330ad99aab7cfe1e7031db33711d4011bc2c1b8/logs/probe/mae_huge14.th)|58.9|

## MAE-CT

|Encoder|Pretrain|Probe|Probe|k-NN|
|:---:|:---:|:---:|:---:|:---:|
|[ViT-B/16](https://ml.jku.at/research/maect/download/maect_base16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/06326017fa605a9b650da36b1f63dd0376e4bd28/yamls/maect/base16.yaml)|[73.5](https://github.com/ml-jku/MAE-CT/blob/0f2fa9373b30ed4d29cf6093ddd2821d179f771b/probes/maect_base16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/1330ad99aab7cfe1e7031db33711d4011bc2c1b8/logs/probe/maect_base16.th)|64.1|
|[ViT-L/16](https://ml.jku.at/research/maect/download/maect_large16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/06326017fa605a9b650da36b1f63dd0376e4bd28/yamls/maect/large16.yaml)|[80.2](https://github.com/ml-jku/MAE-CT/blob/0f2fa9373b30ed4d29cf6093ddd2821d179f771b/probes/maect_large16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/1330ad99aab7cfe1e7031db33711d4011bc2c1b8/logs/probe/maect_large16.th)|78.0|
|[ViT-H/16](https://ml.jku.at/research/maect/download/maect_huge16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/06326017fa605a9b650da36b1f63dd0376e4bd28/yamls/maect/huge16.yaml)|[81.5](https://github.com/ml-jku/MAE-CT/blob/0f2fa9373b30ed4d29cf6093ddd2821d179f771b/probes/maect_huge16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/1330ad99aab7cfe1e7031db33711d4011bc2c1b8/logs/probe/maect_huge16.th)|79.4|
|[ViT-H/14](https://ml.jku.at/research/maect/download/maect_huge14.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/06326017fa605a9b650da36b1f63dd0376e4bd28/yamls/maect/huge14.yaml)|[81.3](https://github.com/ml-jku/MAE-CT/blob/0f2fa9373b30ed4d29cf6093ddd2821d179f771b/probes/maect_huge14.th)|[log](https://github.com/ml-jku/MAE-CT/blob/1330ad99aab7cfe1e7031db33711d4011bc2c1b8/logs/probe/maect_huge14.th)|79.1|
## MAE-CT<sub>*aug*</sub>

|Encoder|Pretrain|Probe|Probe|k-NN|
|:---:|:---:|:---:|:---:|:---:|
|[ViT-B/16](https://ml.jku.at/research/maect/download/maect_base16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/06326017fa605a9b650da36b1f63dd0376e4bd28/yamls/maect/base16.yaml)|[76.9](https://github.com/ml-jku/MAE-CT/blob/0f2fa9373b30ed4d29cf6093ddd2821d179f771b/probes/maectaug_base16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/1330ad99aab7cfe1e7031db33711d4011bc2c1b8/logs/probe/maectaug_base16.th)|73.4|
|[ViT-L/16](https://ml.jku.at/research/maect/download/maect_large16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/06326017fa605a9b650da36b1f63dd0376e4bd28/yamls/maect/large16.yaml)|[81.5](https://github.com/ml-jku/MAE-CT/blob/0f2fa9373b30ed4d29cf6093ddd2821d179f771b/probes/maectaug_large16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/1330ad99aab7cfe1e7031db33711d4011bc2c1b8/logs/probe/maectaug_large16.th)|79.1|
|[ViT-H/16](https://ml.jku.at/research/maect/download/maect_huge16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/06326017fa605a9b650da36b1f63dd0376e4bd28/yamls/maect/huge16.yaml)|[82.2](https://github.com/ml-jku/MAE-CT/blob/0f2fa9373b30ed4d29cf6093ddd2821d179f771b/probes/maectaug_huge16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/1330ad99aab7cfe1e7031db33711d4011bc2c1b8/logs/probe/maectaug_huge16.th)|79.8|
|[ViT-H/14](https://ml.jku.at/research/maect/download/maect_huge14.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/06326017fa605a9b650da36b1f63dd0376e4bd28/yamls/maect/huge14.yaml)|[82.0](https://github.com/ml-jku/MAE-CT/blob/0f2fa9373b30ed4d29cf6093ddd2821d179f771b/probes/maectaug_huge14.th)|[log](https://github.com/ml-jku/MAE-CT/blob/1330ad99aab7cfe1e7031db33711d4011bc2c1b8/logs/probe/maectaug_huge14.th)|78.9|

# Reproducability

- Models can be trained using the hyperparameters provided [here](https://github.com/ml-jku/MAE-CT/yamls). 
  Examples to start training can be found [here](https://github.com/ml-jku/MAE-CT#train-models).
- We provide instructions for reproducing our probing result in 
  [PROBING.md](https://github.com/ml-jku/MAE-CT/blob/e5bc604d1d9003da823e6d832a5ad6762f6897c1/PROBING.md).

# Use checkpoints as backbone for other tasks

The script `eval_probe.py` demonstrates how one can load our models from a checkpoint and use it for a downstream task.
The script extracts the features of the encoder and feeds it to a linear probe as task, but the code can be adjusted
for other downstream tasks as well.


# Setup

Setup a conda environment: `conda env create --file environment_linux.yml --name maect`

We use [FlashAttention](https://github.com/HazyResearch/flash-attention)
([paper](https://arxiv.org/abs/2205.14135)) to greatly accelerate computations. We recommend to install it, but this
repo can be also used without FlashAttention (without modification).

## Configuration of dataset paths and environment specific things

- `cp template_static_config.yaml static_config.yaml`
- edit values in `static_config.yaml` to your setup

For low-shot evaluations, we use the official splits from
[SimCLRv2](https://github.com/google-research/simclr/tree/master/imagenet_subsets)
and [MSN](https://github.com/facebookresearch/msn).

To generate these ImageNet subsets we use the
[ImageNetSubsetGenerator](https://github.com/BenediktAlkin/ImageNetSubsetGenerator) repository.

## [Optional] Configure Weights & Biases

This repo uses [Weights & Biases](https://wandb.ai) for experiment tracking, but offers an alternative in case you do
not want to use it. By default W&B logging is disabled via the `default_wandb_mode: disabled`
configuration in the `static_config.yaml`. You can enable it via `static_config.yaml`
or via the CLI `--wandb_mode online`.

If you enabled W&B logging, the W&B entity and project will (by default) be fetched from the `wandb_config.yaml`. You
can create this via `cp template_wandb_config.yaml wandb_config.yaml` and adjust the values to your setup.

# Run

To run your own experiments or reproduce our results you have to specify the desired hyperparameters via a yaml file.
Afterwards start the training/evaluation run by specifying the following CLI arguments for `main_train.py`

- `--hp <YAML>` (e.g. `--hp yamls/mae/base16.yaml`)
- `--devices <DEVICES>` (e.g. `--devices 0` to run on GPU0 or `--devices 0,1,2,3` to run on 4 GPUs)

## Output

Each yaml file will create a folder in your output directory (defined via `output_path` in `static_config.yaml`). The
output directory is structured into subdirectories with the `stage_name` and the `stage_id`. Example:
`~/output_path/pretrain/9j3kl092`

The output directory of each run is organized as follows:

- `checkpoints`: Model weights will be stored here (choose interval by adjusting the values of the `checkpoint_logger`
  in the yaml file of a run)
- `primitive`: All metrics that are written to Weights & Biases are also stored locally here. If you don't want to use
  W&B you can parse metrics from the files within this directory.
- `log.txt`: logfile
- `hp_resolved.yaml`: a copy of the yaml file that was specified in the `--hp` CLI arg

The yamls used for our paper can be found [here](https://github.com/ml-jku/MAE-CT/tree/main/yamls). Each step of MAE-CT
requires its own yaml file where the later steps require a reference to a checkpoint of a previous step. This can be
defined by changing the `stage_id` of the `initializer` objects within the yaml.

## Examples

### Train models

- Pretrain a MAE on 8 GPUs: <br/>
  `python main_train.py --hp yamls/mae/base16.yaml --devices 0,1,2,3,4,5,6,7`
- An example to train a NNCLR head on frozen encoder features will be up soon.
- Apply contrastive tuning on 8 GPUs (in the yaml file, change the `stage_id` of the `initializer` the encoder and the nnclr head
  to the `stage_id` of your previous step): <br/>
  `python main_train.py --hp yamls/maect/base16.yaml --devices 0,1,2,3,4,5,6,7`

### Evaluate pretrained models

- Adapt the `initializer` of `yamls_probe.yaml` to the model you want to evaluate
- `python main_train.py --hp yamls/probe.yaml --devices 0,1,2,3`

# Citation

If you find this repository useful, please consider giving it a star :star: and cite us

```
@article{lehner2023maect,
      title={Contrastive Tuning: A Little Help to Make Masked Autoencoders Forget}, 
      author={Johannes Lehner and Benedikt Alkin and Andreas Fürst and Elisabeth Rumetshofer and Lukas Miklautz and Sepp Hochreiter},
      journal={arXiv preprint arXiv:2304.10520},
      year={2023}
}
```
