import torch.nn as nn
import torch.nn.functional as F
import torch
from models.swinir import SwinIR
#from swinir import SwinIR #for debug

#下面的几个模块是来自MPRNet的
from models.MPRNet import Encoder, Decoder, SAM, ORSNet,ORB
#from MPRNet import Encoder, Decoder, SAM, ORSNet # for debug

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, device=torch.device('cpu')):
        super(Lap_Pyramid_Conv, self).__init__()

        self.device_para = device

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(self.device_para)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image


import torch.nn as nn
import torch.nn.functional as F
import torch

class Lap_Pyramid_Bicubic(nn.Module):
    """
    """
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Bicubic, self).__init__()

        self.interpolate_mode = 'bicubic'
        self.num_high = num_high

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for i in range(self.num_high):
            down = nn.functional.interpolate(current, size=(current.shape[2] // 2, current.shape[3] // 2), mode=self.interpolate_mode, align_corners=True)
            up = nn.functional.interpolate(down, size=(current.shape[2], current.shape[3]), mode=self.interpolate_mode, align_corners=True)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            image = F.interpolate(image, size=(level.shape[2], level.shape[3]), mode=self.interpolate_mode, align_corners=True) + level
        return image
"""
class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image

"""

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Trans_low(nn.Module):
    def __init__(self, num_residual_blocks):
        super(Trans_low, self).__init__()

        model = [nn.Conv2d(3, 16, 3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        out = torch.tanh(out)
        return out

class Trans_high(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(Trans_high, self).__init__()

        self.num_high = num_high

        model = [nn.Conv2d(9, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 1, 3, padding=1)]

        self.model = nn.Sequential(*model)

        for i in range(self.num_high):
            trans_mask_block = nn.Sequential(
                nn.Conv2d(1, 16, 1),
                nn.LeakyReLU(),
                nn.Conv2d(16, 1, 1))
            setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_mask_block)

    def forward(self, x, pyr_original, fake_low):

        pyr_result = []
        mask = self.model(x) #预测得到一个这一level的高频的mask。

        for i in range(self.num_high):
            mask = nn.functional.interpolate(mask, size=(pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3])) #将mask放大
            trans_mask_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
            mask = trans_mask_block(mask) #转换这一层的mask
            result_highfreq = torch.mul(pyr_original[-2-i], mask) #原始高频信号与mask相乘得到这一次的输出高频信号。
            setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq) #将这一层的高频信号作为属性保存。

        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq) #取出转换后的高频信号。

        pyr_result.append(fake_low) #将输出后的低频信号也加入金字塔中来。

        return pyr_result

#将256分支的特征也传输到高频转换分支，并用SAM等机制实现特征的对齐。
class Trans_high_with_SAM(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(Trans_high_with_SAM, self).__init__()

        self.num_high = num_high

        model = [nn.Conv2d(9, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 1, 3, padding=1)]

        self.model = nn.Sequential(*model)

        for i in range(self.num_high):
            trans_mask_block = nn.Sequential(
                nn.Conv2d(1, 16, 1),
                nn.LeakyReLU(),
                nn.Conv2d(16, 1, 1))
            setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_mask_block)

    def forward(self, x, pyr_original, fake_low):

        pyr_result = []
        mask = self.model(x) #预测得到一个这一level的高频的mask。

        for i in range(self.num_high):
            mask = nn.functional.interpolate(mask, size=(pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3])) #将mask放大
            trans_mask_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
            mask = trans_mask_block(mask) #转换这一层的mask
            result_highfreq = torch.mul(pyr_original[-2-i], mask) #原始高频信号与mask相乘得到这一次的输出高频信号。
            setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq) #将这一层的高频信号作为属性保存。

        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq) #取出转换后的高频信号。

        pyr_result.append(fake_low) #将输出后的低频信号也加入金字塔中来。

        return pyr_result
    
class Trans_high_residual(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(Trans_high_residual, self).__init__()

        self.num_high = num_high

        model = [nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 3, 3, padding=1)]

        self.model = nn.Sequential(*model)

        for i in range(self.num_high):
            """
            model = []
            for _ in range(num_residual_blocks-1):
                model += [ResidualBlock(64)]

            model += [nn.Conv2d(64, 3, 3, padding=1)]
            """
            model = [nn.Conv2d(3, 16, 1),
                nn.LeakyReLU(),
                nn.Conv2d(16, 3, 1)]
            
            trans_residual_block = nn.Sequential(*model)
            setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_residual_block)

    def forward(self, x, pyr_original, fake_low):

        pyr_result = []
        residual = self.model(x) #预测得到一个这一level的高频的mask。

        for i in range(self.num_high):
            residual = nn.functional.interpolate(residual, size=(pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3])) #将mask放大
            trans_residual_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
            residual = trans_residual_block(residual) #转换这一层的mask
            #result_highfreq = torch.mul(pyr_original[-2-i], mask) #原始高频信号与mask相乘得到这一次的输出高频信号。
            result_highfreq = pyr_original[-2-i] + residual 
            setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq) #将这一层的高频信号作为属性保存。

        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq) #取出转换后的高频信号。

        pyr_result.append(fake_low) #将输出后的低频信号也加入金字塔中来。

        return pyr_result

##将下面的self.model使用一个轻量级的U-Net来实现，
class Trans_high_residual_with_UNet(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(Trans_high_residual_with_UNet, self).__init__()

        self.num_high = num_high

        self.model_convfirst = nn.Conv2d(3, 64, 3, padding=1)

        self.act = nn.LeakyReLU()

        self.model_encoder = Encoder(n_feat=64, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=48, csff=False)
        self.model_decoder = Decoder(n_feat=64, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=48)

        #for _ in range(num_residual_blocks):
        #    model += [ResidualBlock(64)]
        #model += [Encoder(n_feat=64, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=48, csff=False),
        #          Decoder(n_feat=64, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=48)]
        #model += [nn.Conv2d(64, 3, 3, padding=1)]

        self.model_conv_last = nn.Conv2d(64, 3, 3, padding=1)

        for i in range(self.num_high):
            """
            model = []
            for _ in range(num_residual_blocks-1):
                model += [ResidualBlock(64)]

            model += [nn.Conv2d(64, 3, 3, padding=1)]
            """
            model = [nn.Conv2d(3, 64, 1),
                nn.LeakyReLU(),
                ORB(64, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, num_cab=4),
                nn.Conv2d(64, 3, 1)]
            
            trans_residual_block = nn.Sequential(*model)
            setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_residual_block)

    def forward(self, x, pyr_original, fake_low):

        pyr_result = []

        x_inter = self.model_convfirst(x)
        x_inter = self.act(x_inter)
        x_encoder_feat = self.model_encoder(x_inter)
        x_decoder_feat = self.model_decoder(x_encoder_feat)

        residual = self.model_conv_last(x_decoder_feat[0]) #预测得到一个这一level的高频的mask。

        for i in range(self.num_high):
            residual = nn.functional.interpolate(residual, size=(pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3])) #将mask放大
            trans_residual_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
            residual = trans_residual_block(residual) #转换这一层的mask
            #result_highfreq = torch.mul(pyr_original[-2-i], mask) #原始高频信号与mask相乘得到这一次的输出高频信号。
            result_highfreq = pyr_original[-2-i] + residual 
            setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq) #将这一层的高频信号作为属性保存。

        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq) #取出转换后的高频信号。

        pyr_result.append(fake_low) #将输出后的低频信号也加入金字塔中来。

        return pyr_result



##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


class Trans_high_residual_with_encoder_decoder(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(Trans_high_residual, self).__init__()

        self.num_high = num_high

        model = [nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 3, 3, padding=1)]

        self.model = nn.Sequential(*model)

        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))

        self.stage2_encoder = Encoder(n_feat=80, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=48, csff=True)
        self.stage2_decoder = Decoder(n_feat=80, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=48)

        self.stage3_orsnet = ORSNet(n_feat=80, scale_orsnetfeats=32, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=48, num_cab=8)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat=80, kernel_size=1, bias=False)

        self.concat23  = conv(n_feat*2, n_feat+scale_orsnetfeats, kernel_size, bias=bias)
        self.tail     = conv(n_feat+scale_orsnetfeats, out_c, kernel_size, bias=bias)

        for i in range(self.num_high):
            """
            model = []
            for _ in range(num_residual_blocks-1):
                model += [ResidualBlock(64)]

            model += [nn.Conv2d(64, 3, 3, padding=1)]
            """
            model = [nn.Conv2d(3, 16, 1),
                nn.LeakyReLU(),
                nn.Conv2d(16, 3, 1)]
            
            trans_residual_block = nn.Sequential(*model)
            setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_residual_block)

    def forward(self, x, pyr_original, fake_low):
        x_256_img = x
        h_512_img = pyr_original[-2]
        h_1024_img = pyr_original[-3]
        ##-------------------------------------------


        ## Apply Supervised Attention Module (SAM)
        x256_samfeats, x_256_restored_img = self.sam12(x_256_feat, x_256_orig_img)

        ## Output image at Stage 1
        stage1_img = x_256_img
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x512_shallow_feat  = self.shallow_feat2(h_512_img)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x512_input_feature = self.concat12(torch.cat([x512_shallow_feat, x256_samfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        x512_endoder_feat = self.stage2_encoder(x512_input_feature, stage1_feature, stage1_img)

        ## Pass features through Decoder of Stage 2
        x512_decoder_feat = self.stage2_decoder(x512_endoder_feat)

        ## Apply SAM
        x512_samfeats, x_512_restored_img = self.sam23(x512_decoder_feat[0], h_512_img)


        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3     = self.shallow_feat3(h_1024_img)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(torch.cat([x3, x512_samfeats], 1))
        
        x3_cat = self.stage3_orsnet(x3_cat, x512_endoder_feat, x512_decoder_feat)

        stage3_img = self.tail(x3_cat)

        x_1024_restored_image = stage3_img + h_1024_img

        return [x_1024_restored_image, x_512_restored_img, x_256_restored_img]


"""
        pyr_result = []
        residual = self.model(x) #预测得到一个这一level的高频的mask。

        for i in range(self.num_high):
            residual = nn.functional.interpolate(residual, size=(pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3])) #将mask放大
            trans_residual_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
            residual = trans_residual_block(residual) #转换这一层的mask
            #result_highfreq = torch.mul(pyr_original[-2-i], mask) #原始高频信号与mask相乘得到这一次的输出高频信号。
            result_highfreq = pyr_original[-2-i] + residual 
            setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq) #将这一层的高频信号作为属性保存。

        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq) #取出转换后的高频信号。

        pyr_result.append(fake_low) #将输出后的低频信号也加入金字塔中来。

        return pyr_result

"""


class LPTNPaper(nn.Module):
    def __init__(self, nrb_low=5, nrb_high=3, num_high=3):
        super(LPTNPaper, self).__init__()

        self.lap_pyramid = Lap_Pyramid_Conv(num_high)
        trans_low = Trans_low(nrb_low)
        trans_high = Trans_high(nrb_high, num_high=num_high)
        self.trans_low = trans_low.cuda()
        self.trans_high = trans_high.cuda()

    def forward(self, real_A_full):

        pyr_A = self.lap_pyramid.pyramid_decom(img=real_A_full)
        fake_B_low = self.trans_low(pyr_A[-1])
        real_A_up = nn.functional.interpolate(pyr_A[-1], size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        fake_B_up = nn.functional.interpolate(fake_B_low, size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        high_with_low = torch.cat([pyr_A[-2], real_A_up, fake_B_up], 1)
        pyr_A_trans = self.trans_high(high_with_low, pyr_A, fake_B_low)
        fake_B_full = self.lap_pyramid.pyramid_recons(pyr_A_trans)

        return fake_B_full

class Trans_high_masked_residual(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(Trans_high_masked_residual, self).__init__()

        self.num_high = num_high

        model = [nn.Conv2d(10, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 1, 3, padding=1)]

        self.model = nn.Sequential(*model)

        for i in range(self.num_high):
            trans_mask_block = nn.Sequential(
                nn.Conv2d(1, 16, 1),
                nn.LeakyReLU(),
                nn.Conv2d(16, 1, 1))
            setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_mask_block)
        """
        self.trans_mask_block = nn.Sequential(
                nn.Conv2d(1, 16, 1),
                nn.LeakyReLU(),
                nn.Conv2d(16, 1, 1))
        """
    def forward(self, x, mask, pyr_original, fake_low): #Trans_high中x是将高频与上采样的低频concate起来的结果，这里我们将其设置为与mask也concate起来的结果

        pyr_result = []
        residual = self.model(torch.cat([x,mask], dim=1))

        for i in range(self.num_high):
            residual = nn.functional.interpolate(residual, size=(pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3]))
            mask = nn.functional.interpolate(mask, size=(pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3]))
            trans_mask_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
            residual = trans_mask_block(residual)
            result_highfreq = torch.add(pyr_original[-2-i], residual)
            result_highfreq = result_highfreq * mask + (1 - mask) * pyr_original[-2-i] #将mask也考虑进来
            setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq)

        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq)

        pyr_result.append(fake_low)

        return pyr_result


class Trans_high_Transformer(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(Trans_high_Transformer, self).__init__()

        self.num_high = num_high

        #model = [nn.Conv2d(9, 64, 3, padding=1),
            #nn.LeakyReLU()]

        #for _ in range(num_residual_blocks):
            #model += [ResidualBlock(64)]

        #model += [nn.Conv2d(64, 1, 3, padding=1)]

        #其实我们要做的就是把这个model换成一个transformer模型就行
        self.model = SwinIR(upscale=1, in_chans=9, img_size=(256, 256),
                   window_size=8, img_range=1., depths=[2],
                   embed_dim=60, num_heads=[6], mlp_ratio=2, upsampler='')


        for i in range(self.num_high):
            trans_mask_block = nn.Sequential(
                nn.Conv2d(1, 16, 1),
                nn.LeakyReLU(),
                nn.Conv2d(16, 1, 1))
            setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_mask_block)

    def forward(self, x, pyr_original, fake_low):

        pyr_result = []
        mask = self.model(x)

        for i in range(self.num_high):
            mask = nn.functional.interpolate(mask, size=(pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3]))
            trans_mask_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
            mask = trans_mask_block(mask)
            result_highfreq = torch.mul(pyr_original[-2-i], mask)
            setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq)

        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq)

        pyr_result.append(fake_low)

        return pyr_result
    
def test():
    lptn_model = Lap_Pyramid_Conv(num_high=2, device='cuda')
    lptn_model = lptn_model.cuda()
    input_t = torch.randn(1, 3, 224, 224).cuda()

    output_t = lptn_model.pyramid_decom(input_t)
    output_t_len = len(output_t)
    for i in range(output_t_len):
        print('output_t.shape =', output_t[i].shape)

    pyr_origin = output_t

    x = torch.randn(1, 9, 112, 112).cuda()
    mask = torch.randn(1, 1, 112, 112).cuda()
    fake_low = torch.randn(1, 3, 56, 56).cuda()

    high_trans_mask = Trans_high_masked_residual(num_residual_blocks=3, num_high=2)
    lptn_model = high_trans_mask.cuda()

    output = high_trans_mask(x, mask, pyr_origin, fake_low)
    output_t_len = len(output)
    for i in range(output_t_len):
        print('output.shape =', output[i].shape)


    high_trans = Trans_high_Transformer(num_residual_blocks=3, num_high=2)
    lptn_model = high_trans.cuda()

    output = high_trans(x, pyr_origin, fake_low)
    output_t_len = len(output)
    for i in range(output_t_len):
        print('output.shape =', output[i].shape)

    high_trans = Trans_high_residual_with_UNet(num_residual_blocks=3, num_high=2)
    lptn_model = high_trans.cuda()

    x_input = torch.randn(1, 3, 256, 256).cuda()
    mask = torch.randn(1, 1, 256, 256).cuda()
    fake_low = torch.randn(1, 3, 128, 128).cuda()

    output = high_trans(x_input, pyr_origin, fake_low)
    output_t_len = len(output)
    for i in range(output_t_len):
        print('output.shape =', output[i].shape)

if __name__ == "__main__":
    test()


