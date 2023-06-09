from PIL import Image
import numpy as np
import os
import torch
import argparse
import pytorch_ssim
import torchvision.transforms.functional as tf
import torchvision
import torch.nn.functional as f
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm
import pandas as pd

def save_psnr_to_csv(df, image_name, psnr_score):
    new_row = pd.DataFrame({
        "fileanme":[image_name],
        "PSNR":[psnr_score]
    })
    # 将新的行添加到 DataFrame 中
    df = df.append(new_row, ignore_index=True)


"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test', help='train or test ?')
    parser.add_argument('--dataroot', type=str, default='', help='dataset_dir')
    parser.add_argument('--result_root', type=str, default='', help='dataset_dir')
    parser.add_argument('--dataset_name', type=str, default='ihd', help='dataset_name')
    parser.add_argument('--evaluation_type', type=str, default="our", help='evaluation type')
    parser.add_argument('--ssim_window_size', type=int, default=11, help='ssim window size')
    parser.add_argument('--image_size', type=int, default=1024, help='evaluation image size')

    return parser.parse_args()



def main(dataset_name = None):
    cuda = True if torch.cuda.is_available() else False
    opt.dataset_name = dataset_name
    files = opt.dataroot+ dataset_name + '/' + opt.dataset_name+'_'+opt.phase+'.txt'
    comp_paths = []
    harmonized_paths = []
    mask_paths = []
    real_paths = []
    with open(files,'r') as f:
            for line in f.readlines():
                name_str = line.rstrip()
                if opt.evaluation_type == 'our':
                    harmonized_path = os.path.join(opt.result_root,name_str[:-4] + '_harmonized.jpg')
                    comp_path = os.path.join(opt.dataroot, dataset_name, 'composite_images', line.rstrip())

                    #print(harmonized_path)
                    if os.path.exists(harmonized_path):
                        real_path = os.path.join(opt.dataroot, dataset_name, 'real_images',line.rstrip())

                        name_parts=real_path.split('_')
                        real_path = real_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')
                        #print('real_path=', real_path)

                        mask_path = os.path.join(opt.dataroot,dataset_name, 'masks',line.rstrip())
                        mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')

                elif opt.evaluation_type == 'ori':
                    comp_path = os.path.join(opt.dataroot, 'composite_images', line.rstrip())
                    harmonized_path = comp_path
                    if os.path.exists(comp_path):
                        real_path = os.path.join(opt.dataroot,'real_images',line.rstrip())
                        name_parts=real_path.split('_')
                        real_path = real_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')
                        mask_path = os.path.join(opt.dataroot,'masks',line.rstrip())
                        mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')

                real_paths.append(real_path)
                mask_paths.append(mask_path)
                harmonized_paths.append(harmonized_path)
                comp_paths.append(comp_path)
    count = 0


    mse_scores = 0
    sk_mse_scores = 0
    fmse_scores = 0
    psnr_scores = 0
    fpsnr_scores = 0
    ssim_scores = 0
    fssim_scores = 0
    fore_area_count = 0
    fmse_score_list = []
    image_size = opt.image_size

    ## 打开已有表格文件或创建一个新表格
    try:
        csv_path = os.path.join(opt.result_root, "evaluation_output.csv")
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["filename", "PSNR"])

    for i, harmonized_path in enumerate(tqdm(harmonized_paths)):
        count += 1

        harmonized = Image.open(harmonized_path).convert('RGB')
        composite = Image.open(comp_paths[i]).convert('RGB')

        real = Image.open(real_paths[i]).convert('RGB')
        mask = Image.open(mask_paths[i]).convert('1')
        if mask.size[0] != image_size:
            harmonized = tf.resize(harmonized,[image_size,image_size], interpolation=Image.BILINEAR)
            mask = tf.resize(mask, [image_size,image_size], interpolation=Image.BILINEAR)
            real = tf.resize(real,[image_size,image_size], interpolation=Image.BILINEAR)
            composite = tf.resize(composite,[image_size,image_size], interpolation=Image.BILINEAR)

        harmonized_np = np.array(harmonized, dtype=np.float32)
        real_np = np.array(real, dtype=np.float32)
        composite_np = np.array(composite, dtype=np.float32)


        harmonized = tf.to_tensor(harmonized_np).unsqueeze(0).cuda()
        real = tf.to_tensor(real_np).unsqueeze(0).cuda()
        mask = tf.to_tensor(mask).unsqueeze(0).cuda()
        composite = tf.to_tensor(composite_np).unsqueeze(0).cuda()

        mse_score = mse(harmonized_np, real_np)
        psnr_score = psnr(real_np, harmonized_np, data_range=255)
        
        #使用原始composite数据evalution
        #mse_score = mse(composite_np, real_np)
        #psnr_score = psnr(real_np, composite_np, data_range=255)

        fore_area = torch.sum(mask)
        fmse_score = torch.nn.functional.mse_loss(harmonized*mask,real*mask)*256*256/fore_area

        mse_score = mse_score.item()
        fmse_score = fmse_score.item()
        fore_area_count += fore_area.item()
        fpsnr_score = 10 * np.log10((255 ** 2) / fmse_score)

        ssim_score, fssim_score = pytorch_ssim.ssim(harmonized, real, window_size=opt.ssim_window_size, mask=mask)

        #print('%s | mse %0.2f | psnr %0.2f | ssim %0.3f' % (image_name,mse_score,psnr_score, ssim_score)
        #save_psnr_to_csv(df, image_name=harmonized_path, psnr_score=psnr_score)
        new_row = pd.DataFrame({
            "fileanme":[harmonized_path],
            "PSNR":[psnr_score]
        })
        # 将新的行添加到 DataFrame 中
        df = df.append(new_row, ignore_index=True)

        psnr_scores += psnr_score
        mse_scores += mse_score
        fmse_scores += fmse_score
        fpsnr_scores += fpsnr_score
        ssim_scores += ssim_score
        fssim_scores += fssim_score


        image_name = harmonized_path.split("/")
        image_fmse_info = (image_name[-1], round(fmse_score,2), fore_area.item(), round(mse_score, 2), round(psnr_score, 2), round(fpsnr_scores, 4))
        fmse_score_list.append(image_fmse_info)

    mse_scores_mu = mse_scores/count
    psnr_scores_mu = psnr_scores/count
    fmse_scores_mu = fmse_scores/count
    fpsnr_scores_mu = fpsnr_scores/count
    ssim_scores_mu = ssim_scores/count
    fssim_score_mu = fssim_scores/count


    print(count)
    mean_sore = "%s MSE %0.2f | PSNR %0.2f | SSIM %0.4f |fMSE %0.2f | fPSNR %0.2f | fSSIM %0.4f" % (opt.dataset_name,mse_scores_mu, psnr_scores_mu,ssim_scores_mu,fmse_scores_mu,fpsnr_scores_mu,fssim_score_mu)
    print(mean_sore)  

    # 将 DataFrame 保存为表格文件
    df.to_csv(csv_path, index=False)

    return mse_scores_mu,fmse_scores_mu, psnr_scores_mu,fpsnr_scores_mu

def generstr(dataset_name='ALL'): 
    #datasets = ['HCOCO','HAdobe5k','HFlickr','Hday2night','IHD']
    datasets = ['HAdobe5k']
    if dataset_name == 'newALL':
        datasets = ['HCOCO','HAdobe5k','HFlickr','Hday2night','HVIDIT','newIHD']
    for i, item in enumerate(datasets):
        print(item)
        mse_scores_mu,fmse_scores_mu, psnr_scores_mu,fpsnr_scores_mu = main(dataset_name=item)
        

if __name__ == '__main__':
    opt = parse_args()
    if opt is None:
        exit()
    if opt.dataset_name == "ALL":
        generstr()
    elif opt.dataset_name == "newALL":
        generstr(dataset_name='newALL')
    else:
        main(dataset_name=opt.dataset_name)
