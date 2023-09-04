<base target="_blank"/>

# Official codes for High-Resolution Image Harmonization with a Simple Hybrid CNN-Transformer Network 


## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Train/Test
- Download [iHarmony4](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4) dataset.

- Train our model:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model dht --light_use_mask --tr_r_enc_head x --tr_r_enc_layers x  --tr_i_dec_head x --tr_i_dec_layers x --tr_l_dec_head x --tr_l_dec_layers x --name DHT_experiment_name --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
- Test our model:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model dht --light_use_mask --tr_r_enc_head x --tr_r_enc_layers x  --tr_i_dec_head x --tr_i_dec_layers x --tr_l_dec_head x --tr_l_dec_layers x --name DHT_experiment_name --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```

## Apply a pre-trained model
- Download pre-trained models from
```bash

```
## Evaluation
We provide the code in `ih_evaluation.py`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation.py --dataroot <dataset_dir> --result_root  results/experiment_name/test_latest/images/ --evaluation_type our --dataset_name ALL
```

## 




```

}
```

# Acknowledgement
For some of the data modules and model functions used in this source code, we need to acknowledge the repositories of [DoveNet](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4/tree/master/DoveNet), [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [
SpiralNet](https://github.com/zhenglab/spiralnet) and [IntrinsicHarmony](https://github.com/zhenglab/IntrinsicHarmony). 
