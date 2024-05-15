<base target="_blank"/>

# Official codes for High-Resolution Image Harmonization with a Simple Hybrid CNN-Transformer Network 


## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Install basicsr
```bash
python setup_basicsr.py develop --no_cuda_ext
```

## Train/Test
- Download [iHarmony4](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4) dataset.
We have partitioned a validation set from the original iHarmony4 dataset. 
Detailed partitioning text files can be found at [Google Drive](https://drive.google.com/drive/folders/1HckIxN0HR68DRAiYJO_RqGKYk_m1fxN0?usp=drive_link)
- Train our model:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model SPANET --name experimental_name --dataset_root /***/iHarmony4/HAdobe5k/ --dataset_name HAdobe5k --batch_size 4 --init_port 55554 --local_rank 4 --crop_size 1024 --load_size 1024 --netG SPANET
```
- Test our model:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model SPANET --name experimental_name --dataset_root /***/iHarmony4/HAdobe5k/ --dataset_name HAdobe5k --batch_size 4 --init_port 55554 --local_rank 4 --crop_size 1024 --load_size 1024 --netG SPANET
```

## Apply a pre-trained model
- Download pre-trained models from

## Evaluation
We provide the code in `ih_evaluation.py`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation.py --dataroot <dataset_dir> --result_root  /**/results/experiment_name/test_latest/images/ --evaluation_type our --dataset_name HAdobe5k  --image_size 1024
```


# Acknowledgement
For some of the data modules and model functions used in this source code, we need to acknowledge the repositories of [DoveNet](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4/tree/master/DoveNet), [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [
SpiralNet](https://github.com/zhenglab/spiralnet), [IntrinsicHarmony](https://github.com/zhenglab/IntrinsicHarmony) and [DHT](https://github.com/zhenglab/HarmonyTransformer)
