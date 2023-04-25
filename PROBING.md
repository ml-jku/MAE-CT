# Probing

## Reproduce paper results

The hyperparameters to reproduce our probing results can be found 
[here](https://github.com/ml-jku/MAE-CT/blob/e5bc604d1d9003da823e6d832a5ad6762f6897c1/yamls/probe.yaml).
- change the `initializer` field to load the model you want to evaluate
- run with `python main_train.py --hp yamls/probe.yaml --devices 0`


## Evaluate models

Once a probe is trained, it can be evaluated via the `eval_probe.py` script. We use
[FlashAttention](https://github.com/HazyResearch/flash-attention) in combination with bfloat16 precision.
Results might differ slightly if no [FlashAttention](https://github.com/HazyResearch/flash-attention) or other 
precisions are used.

The weight files can be downloaded from the [main page](https://github.com/ml-jku/MAE-CT).

```
# B/16
# accuracy: 0.6673
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/mae_reimpl_base16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/mae_reimpl_base16.th" --device <DEVICE>
# accuracy: 0.7350
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maect_base16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maect_base16.th" --pooling mean_patch --device <DEVICE>
# accuracy: 0.7692
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maectaug_base16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maectaug_base16.th" --pooling mean_patch --device <DEVICE>

# L/16
# accuracy: 0.7592
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/mae_reimpl_large16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/mae_reimpl_large16.th" --device <DEVICE>
# accuracy: 0.8017
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maect_large16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maect_large16.th" --device <DEVICE>
# accuracy: 0.8146
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maectaug_large16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maectaug_large16.th" --device <DEVICE>

# H/16
# accuracy: 0.7796
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/mae_reimpl_huge16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/mae_reimpl_huge16.th" --device <DEVICE>
# accuracy: 0.8152
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maect_huge16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maect_huge16.th" --device <DEVICE>
# accuracy: 0.8217
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maectaug_huge16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maectaug_huge16.th" --device <DEVICE>

# H/14
# accuracy: 0.7723
# python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/mae_pretrain_vit_huge.pth" --head "/system/user/publicwork/ssl/github_checkpoints/probes/mae_huge14.th" --device <DEVICE>
# accuracy: 0.8124
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maect_huge14.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maect_huge14.th" --device <DEVICE>
# accuracy: 0.8195
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maectaug_huge14.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maectaug_huge14.th" --device <DEVICE>
```