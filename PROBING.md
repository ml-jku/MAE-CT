# Probing

## Reproduce paper results

The hyperparameters to reproduce our probing results can be found [here](https://github.com/ml-jku/MAE-CT/yamls/probe).


## Evaluate models

Once a probe is trained, it can be evaluated via the `eval_probe.py` script. We use
[FlashAttention](https://github.com/HazyResearch/flash-attention) in combination with bfloat16 precision.
Results might differ slightly if no [FlashAttention](https://github.com/HazyResearch/flash-attention) or other 
precisions are used.

```
# B/16
# TODO
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/mae_base16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/mae_base16.th" --device <DEVICE>
# accuracy: 0.7350
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maect_base16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maect_base16.th" --pooling mean_patch --device <DEVICE>
# accuracy: 0.7692
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maectaug_base16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maectaug_base16.th" --pooling mean_patch --device <DEVICE>

# L/16
# TODO
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/mae_large16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/mae_large16.th" --device <DEVICE>
# accuracy: 0.8017
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maect_large16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maect_large16.th" --device <DEVICE>
# accuracy: 0.8146
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maectaug_large16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maectaug_large16.th" --device <DEVICE>

# H/16
# TODO
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/mae_huge16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/mae_huge16.th" --device <DEVICE>
# accuracy: 0.8152
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maect_huge16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maect_huge16.th" --device <DEVICE>
# accuracy: 0.8217
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maectaug_huge16.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maectaug_huge16.th" --device <DEVICE>

# H/14
# TODO
# python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/mae_pretrain_vit_huge.pth" --head "/system/user/publicwork/ssl/github_checkpoints/probes/mae_huge14.th" --device <DEVICE>
# accuracy: 0.8124
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maect_huge14.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maect_huge14.th" --device <DEVICE>
# accuracy: 0.8195
python eval_probe.py --encoder "/system/user/publicwork/ssl/github_checkpoints/maectaug_huge14.th" --head "/system/user/publicwork/ssl/github_checkpoints/probes/maectaug_huge14.th" --device <DEVICE>
```