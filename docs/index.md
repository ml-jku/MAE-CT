[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mim-refiner-a-contrastive-learning-boost-from/self-supervised-image-classification-on)](https://paperswithcode.com/sota/self-supervised-image-classification-on?p=mim-refiner-a-contrastive-learning-boost-from) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mim-refiner-a-contrastive-learning-boost-from/image-clustering-on-imagenet)](https://paperswithcode.com/sota/image-clustering-on-imagenet?p=mim-refiner-a-contrastive-learning-boost-from)



[[Code](https://github.com/ml-jku/MAE-CT)] [[Paper + Appendix](https://arxiv.org/abs/2304.10520)] [[Models](https://github.com/ml-jku/MAE-CT#pretrained-checkpoints)] [[Logs](https://github.com/ml-jku/MIM-Refiner#logs)]  [[BibTeX](https://github.com/ml-jku/MAE-CT#citation)]

<p align="center">
<img width="31.5%" alt="maect_schematic" src="https://github.com/ml-jku/MAE-CT/blob/137d969be4c78d156465bb18d09f52d3c762114f/.github/schematic_contrastive_tuning.svg">
<img width="67%" alt="lowshot_vitl" src="https://github.com/ml-jku/MAE-CT/blob/2ff19e68df9c3a1a7cb17a1846ac0d937359392c/.github/lowshot_aug_L_white.svg">
</p>


MIM-Refiner refines the representation of pre-trained **M**asked **I**mage **M**odels (MIM) by attaching 
**I**nstance **D**iscrimination (ID) heads to multiple intermediate heads. This setup is then trained for a few epochs with
with our **N**earest **N**eighbor **A**lignment (NNA) objective.


<p align="center">
<img width="100%" alt="mimrefiner_schematic" src="https://raw.githubusercontent.com/ml-jku/MIM-Refiner/main/docs/imgs/schematic.svg">
</p>


MIM-Refiner drastically advances state-of-the-art in ImageNet-1K linear probing.
It achieves an improvement of +2.5% over the previous state-of-the-art.
In comparison, over the last 4 years, state-of-the-art improved by +2.6%.


<p align="center">
<img width="80%" alt="mimrefiner_timeline" src="https://raw.githubusercontent.com/ml-jku/MIM-Refiner/main/docs/imgs/timeline.svg">
</p>

