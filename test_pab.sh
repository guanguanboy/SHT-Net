echo "Begin init conda..."
. /mnt/data/optimal/zhaobin_gl/miniconda3/etc/profile.d/conda.sh
which conda

echo "Begin activating conda env..."
conda activate base

echo "Begin training!"
cd /mnt/data/optimal/zhaobin_gl/Codes/SwinHIH
python test.py --model lapswinih_pab --name lapswinIHPAB_0606 --dataset_root /mnt/data/optimal/zhaobin_gl/Datasets/iHarmony4/HAdobe5k/ --dataset_name HAdobe5k --batch_size 1 --init_port 55554 --local_rank 1 --crop_size 1024 --load_size 1024
