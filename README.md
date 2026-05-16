# Position-Aware Contextual Feature Fields for Few-Shot Neural Radiance Field Fine-Tuning
---
This repository contains the implementation of the method described in the paper "Position-Aware Contextual Feature Fields for Few-Shot Neural Radiance Field Fine-Tuning". Submitted to The Visual Computer (Springer Journal).
## Installation
---
#### Tested on Ubuntu 20.04 + Pytorch 1.11.0 + CUDA 11.3
Install environment:
```
python=3.8
pip install torch torchvision 
pip install einops imageio-ffmpeg opencv-python scikit-image scipy tensorboard tensorflow tqdm
```
## Datasets
---
### 1.Training datasets
The organization of the datasets should be consistent with the structure below.
```
|--data/
    |--spaces_dataset/
    |--ibrnet_collected_1/
    |--ibrnet_collected_2/
    |--google_scanned_objects/
```
- [Spaces](https://github.com/augmentedperception/spaces_dataset)
- [ibrnet_collected](https://drive.google.com/file/d/1dZZChihfSt9iIzcQICojLziPvX1vejkp/view)
- [Google Scanned Objects](https://drive.google.com/file/d/1tKHhH-L1viCvTuBO1xg--B_ioK7JUrrE/view)
### 2.Evaluation Datasets
```
|--data/
    |--nerf_synthetic/
    |--nerf_llff_data/
```
- [LLFF](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4)
- [NeRF_Synthetic](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4)
## Training
---
The training script is in train.py, running:
```
python train.py --config configs/pretrain.txt
```
## Finetuning
---
To use CFF finetune on a specific scene, for example, fern, using the pretrained model, run:
```
python train.py --config configs/finetune_llff.txt
```
## Evaluation
---
You can use eval/eval.py to evaluate the pretrained/finetuned models to obtain the PSNR, SSIM and LPIPS on different scenes. For example, on the fern scene in the LLFF dataset, you can specify your paths in configs/eval_llff.txt and then run:
```
cd eval/
python eval.py --config ../configs/eval_llff.txt
```
