import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.optim import lr_scheduler
from torchvision import models
from util.tools import *
from util import util
from . import base_networks as networks_init
from . import transformer,swinir,swinir_lap,swinir_lap_refine,lap_swinih_arch,swinir_ds,lap_restormer,restormer_arch, lap_NAFNet_arch, SHTNet_arch
from basicsr.models.archs import NAFNet_arch
from . import uformer_arch,SPANET_arch,fftformer_arch,SAGNet_arch,NAFNet_WFFN_arch,SPANET_arch_backup_old,SPANET_arch_backup_0717
import math
from thop import profile
from thop import clever_format

def define_G(netG='retinex',init_type='normal', init_gain=0.02, opt=None):
    """Create a generator
    """
    if netG == 'CNNHT':
        net = CNNHTGenerator(opt)
    elif netG == 'HT':
        net = HTGenerator(opt)
    elif netG == 'DHT':
        net = DHTGenerator(opt)
    elif netG == 'SWINHIH':
        net = SWINHIHGenerator(opt)
    elif netG == 'LAPSWINHIH':
        net = LAPSWINHIHGenerator(opt)
    elif netG == 'LAPSWINIH_PAB':
        net = LAPSWINIHPABGenerator(opt)
    elif netG == 'IHDS':
        net = IHDSGenerator(opt)
    elif netG == 'LAPRESTORMER':
        net = LAPRESTORMERGenerator(opt)
    elif netG == 'LAPRESTORMERMUTLI':
        net = LAPRESTORMERMULTIGenerator(opt)
    elif netG == 'RESTORMER':
        net = RESTORMERGenerator(opt)
    elif netG == 'NAFNet':
        net = NAFNetGenerator(opt)
    elif netG == 'LAPNAFNET':
        net = LAPNAFNetGenerator(opt)
    elif netG == 'UFORMER':
        net = UFormerGenerator(opt)
    elif netG == 'SPANET':
        net = SPANetSmallGenerator(opt)
        #net = SPANetGenerator(opt)        
    elif netG == 'FFTFORMER':
        net = FFTFormerGenerator(opt)         
    elif netG == "SHTNet":
        net = SHTNetGenerator(opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    net = networks_init.init_weights(net, init_type, init_gain)
    net = networks_init.build_model(opt, net)
    return net

class LAPSWINIHPABGenerator(nn.Module):
    def __init__(self, opt=None):
        super(LAPSWINIHPABGenerator, self).__init__()

        self.swinhih = lap_swinih_arch.LapSwinIH(upscale=1, in_chans=4, img_size=256, window_size=16,
                    img_range=1., depths=[2, 4, 6, 6, 4, 2], embed_dim=120, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        
    def forward(self, inputs):
        harmonized = self.swinhih(inputs)
        return harmonized
    
class LAPSWINHIHGenerator(nn.Module):
    def __init__(self, opt=None):
        super(LAPSWINHIHGenerator, self).__init__()

        """
        self.swinhih = swinir_lap.LapSwinIR(upscale=1, in_chans=4, img_size=256, window_size=8,
                    img_range=1., depths=[2, 2, 2], embed_dim=180, num_heads=[6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        """
        self.swinhih = swinir_lap.LapSwinIR(upscale=1, in_chans=4, img_size=256, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=120, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')        

        """
        self.swinhih = swinir_lap_refine.LapSwinIR(upscale=1, in_chans=4, img_size=256, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=120, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        
        """        
    def forward(self, inputs):
        harmonized = self.swinhih(inputs)
        return harmonized
    
class SWINHIHGenerator(nn.Module):
    def __init__(self, opt=None):
        super(SWINHIHGenerator, self).__init__()

        self.swinhih = swinir.SwinIR(upscale=1, in_chans=4, img_size=256, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=120, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        
    def forward(self, inputs):
        harmonized = self.swinhih(inputs)
        return harmonized

class IHDSGenerator(nn.Module):
    def __init__(self, opt=None):
        super(IHDSGenerator, self).__init__()

        self.swinhih = swinir_ds.SwinIR_DS(upscale=1, in_chans=4, img_size=256, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=120, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv') 
        
    def forward(self, inputs):
        harmonized = self.swinhih(inputs)
        return harmonized

class NAFNetGenerator(nn.Module):
    def __init__(self, opt=None):
        super(NAFNetGenerator, self).__init__()
        
        img_channel = 4
        width = 32 #初始设置为32

        enc_blks = [1, 1, 1, 28]
        middle_blk_num = 1
        dec_blks = [1, 1, 1, 1]
    
        """
        self.nafnet = NAFNet_arch.NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                        enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        """        
        """
        self.nafnet = SAGNet_arch.SAGNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                        enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        """        
        self.nafnet = NAFNet_WFFN_arch.NAFWFNNNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                        enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        
    def forward(self, inputs):
        harmonized = self.nafnet(inputs)
        return harmonized

class UFormerGenerator(nn.Module):
    def __init__(self, opt=None):
        super(UFormerGenerator, self).__init__()
        
        input_size = 1024
        depths=[2, 2, 2, 2, 2, 2, 2, 2, 1]
    
        self.uformer = uformer_arch.Uformer(img_size=input_size, in_chans=3, dd_in=4, embed_dim=16,depths=depths,
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
                
    def forward(self, inputs):
        harmonized = self.uformer(inputs)
        return harmonized

class SHTNetGenerator(nn.Module):
    def __init__(self, opt=None):
        super(SHTNetGenerator, self).__init__()
        
        input_size = 1024
        depths=[1, 1, 1, 1, 28, 1, 1, 1, 1]

        self.shtnet = SHTNet_arch.SHTNet(img_size=input_size, in_chans=3, dd_in=4, embed_dim=32,depths=depths,
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
        
        self.evaluate_efficiency(image_size = 1024)

    def evaluate_efficiency(self,image_size = 256):
        size = image_size
        gt = torch.randn((1,3,size,size)).cuda()
        cond = torch.randn(1,4,size,size).cuda()
        mask = torch.randn(1,1,size,size).cuda()

        self.spanet = self.spanet.cuda()
        flops, params = profile(self.spanet, inputs=(cond,))
        flops, params = clever_format([flops, params], '%.3f')

        print('params=', params)
        print('FLOPs=',flops)
        """
        self.spanet = SPANET_arch_backup_old.SPANet(img_size=input_size, in_chans=3, dd_in=4, embed_dim=32,depths=depths,win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
        """
    def forward(self, inputs):
        harmonized_small, harmonized = self.shtnet(inputs)
        return harmonized_small, harmonized
    
class SPANetSmallGenerator(nn.Module):
    def __init__(self, opt=None):
        super(SPANetSmallGenerator, self).__init__()
        
        input_size = 1024
        depths=[1, 1, 1, 1, 28, 1, 1, 1, 1]

        self.spanet = SPANET_arch.SPANetSmall(img_size=input_size, in_chans=3, dd_in=4, embed_dim=16,depths=depths,
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
        
        self.evaluate_efficiency(image_size = 1024)

    def evaluate_efficiency(self,image_size = 256):
        size = image_size
        gt = torch.randn((1,3,size,size)).cuda()
        cond = torch.randn(1,4,size,size).cuda()
        mask = torch.randn(1,1,size,size).cuda()

        self.spanet = self.spanet.cuda()
        flops, params = profile(self.spanet, inputs=(cond,))
        flops, params = clever_format([flops, params], '%.3f')

        print('params=', params)
        print('FLOPs=',flops)
        """
        self.spanet = SPANET_arch_backup_old.SPANet(img_size=input_size, in_chans=3, dd_in=4, embed_dim=32,depths=depths,win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
        """
    def forward(self, inputs):
        harmonized = self.spanet(inputs)
        return harmonized
    
class SPANetGenerator(nn.Module):
    def __init__(self, opt=None):
        super(SPANetGenerator, self).__init__()
        
        input_size = 1024
        depths=[1, 1, 1, 1, 28, 1, 1, 1, 1]

        self.spanet = SPANET_arch.SPANet(img_size=input_size, in_chans=3, dd_in=4, embed_dim=16,depths=depths,
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
        
        self.evaluate_efficiency(image_size = 1024)

        """
        self.spanet = SPANET_arch_backup_old.SPANet(img_size=input_size, in_chans=3, dd_in=4, embed_dim=32,depths=depths,win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
        """
    def forward(self, inputs):
        harmonized = self.spanet(inputs)
        return harmonized
    
    def evaluate_efficiency(self,image_size = 256):
        size = image_size
        gt = torch.randn((1,3,size,size)).cuda()
        cond = torch.randn(1,4,size,size).cuda()
        mask = torch.randn(1,1,size,size).cuda()

        self.spanet = self.spanet.cuda()
        flops, params = profile(self.spanet, inputs=(cond,))
        flops, params = clever_format([flops, params], '%.3f')

        print('params=', params)
        print('FLOPs=',flops)

    def evaluate_inference_speed(self, image_size=256):
        size = image_size
        gt = torch.randn((1,3,size,size)).cuda()
        cond = torch.randn(1,3,size,size).cuda()
        mask = torch.randn(1,1,size,size).cuda()

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings=np.zeros((repetitions,1))
        #GPU-WARM-UP
        for _ in range(10):
            _ = self.spanet(gt, cond, mask)             
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = self.netG(gt, cond, mask)  
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        mean_fps = 1000. / mean_syn
        print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
        print(mean_syn)

class FFTFormerGenerator(nn.Module):
    def __init__(self, opt=None):
        super(FFTFormerGenerator, self).__init__()
        
        self.fftformer = fftformer_arch.fftformer(inp_channels=4, out_channels=3, dim=8)
    
    def forward(self, inputs):
        harmonized = self.fftformer(inputs)
        return harmonized
        
class LAPNAFNetGenerator(nn.Module):
    def __init__(self, opt=None):
        super(LAPNAFNetGenerator, self).__init__()
        
        img_channel = 4
        width = 32

        enc_blks = [1, 1, 1, 28]
        middle_blk_num = 1
        dec_blks = [1, 1, 1, 1]
    
        self.nafnet = lap_NAFNet_arch.LapNAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
                
    def forward(self, inputs):
        harmonized = self.nafnet(inputs)
        return harmonized
        
class RESTORMERGenerator(nn.Module):
    def __init__(self, opt=None):
        super(RESTORMERGenerator, self).__init__()
        
        self.restormer = restormer_arch.Restormer(inp_channels=4, 
                      out_channels= 3,
                      dim= 48,
                      num_blocks= [4,6,6,8],
                      num_refinement_blocks= 4,
                      heads= [1,2,4,8],
                      ffn_expansion_factor= 2.66,
                      bias= False,
                      LayerNorm_type= 'BiasFree',
                      dual_pixel_task= False)
                
    def forward(self, inputs):
        harmonized = self.restormer(inputs)
        return harmonized
    
class LAPRESTORMERGenerator(nn.Module):
    def __init__(self, opt=None):
        super(LAPRESTORMERGenerator, self).__init__()
        """
        self.lap_restormer = lap_restormer.LapRestormer(inp_channels=4, 
                      out_channels= 3,
                      dim= 48,
                      num_blocks= [4,6,6,8],
                      num_refinement_blocks= 4,
                      heads= [1,2,4,8],
                      ffn_expansion_factor= 2.66,
                      bias= False,
                      LayerNorm_type= 'BiasFree',
                      dual_pixel_task= False)
        
        """
        self.lap_restormer = lap_restormer.LapRestormerWithMPRHighBranch(inp_channels=4, 
                      out_channels= 3,
                      dim= 48,
                      num_blocks= [4,6,6,8],
                      num_refinement_blocks= 4,
                      heads= [1,2,4,8],
                      ffn_expansion_factor= 2.66,
                      bias= False,
                      LayerNorm_type= 'BiasFree',
                      dual_pixel_task= False)
                
    def forward(self, inputs):
        harmonized = self.lap_restormer(inputs)
        return harmonized

class LAPRESTORMERMULTIGenerator(nn.Module):
    def __init__(self, opt=None):
        super(LAPRESTORMERMULTIGenerator, self).__init__()

        self.lap_restormer = lap_restormer.LapRestormerMulti(inp_channels=4, 
                      out_channels= 3,
                      dim= 48,
                      num_blocks= [4,6,6,8],
                      num_refinement_blocks= 4,
                      heads= [1,2,4,8],
                      ffn_expansion_factor= 2.66,
                      bias= False,
                      LayerNorm_type= 'BiasFree',
                      dual_pixel_task= False)
        
    def forward(self, inputs):
        harmonized, low_freq = self.lap_restormer(inputs)
        return harmonized,low_freq
    
class HTGenerator(nn.Module):
    def __init__(self, opt=None):
        super(HTGenerator, self).__init__()
        dim = 256
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = opt.ksize, p2 = opt.ksize),
            nn.Linear(opt.ksize*opt.ksize*(opt.input_nc+1), dim)
        )
        self.transformer_enc = transformer.TransformerEncoders(dim, nhead=opt.tr_r_enc_head, num_encoder_layers=opt.tr_r_enc_layers, dim_feedforward=dim*opt.dim_forward, activation=opt.tr_act)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim, opt.output_nc, kernel_size=opt.ksize, stride=opt.ksize, padding=0),
            nn.Tanh()
        )
    def forward(self, inputs, pixel_pos=None):
        patch_embedding = self.patch_to_embedding(inputs)
        content = self.transformer_enc(patch_embedding.permute(1, 0, 2), src_pos=pixel_pos)
        bs, L, C  = patch_embedding.size()
        harmonized = self.dec(content.permute(1,2,0).view(bs, C, int(math.sqrt(L)), int(math.sqrt(L))))
        return harmonized

class CNNHTGenerator(nn.Module):
    def __init__(self, opt=None):
        super(CNNHTGenerator, self).__init__()
        dim = 256
        self.enc = ContentEncoder(opt.n_downsample, 0, opt.input_nc+1, dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.transformer_enc = transformer.TransformerEncoders(dim, nhead=opt.tr_r_enc_head, num_encoder_layers=opt.tr_r_enc_layers, dim_feedforward=dim*opt.dim_forward, activation=opt.tr_act)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim, opt.output_nc, kernel_size=opt.ksize, stride=opt.ksize, padding=0),
            nn.Tanh()
        )
        if opt.use_two_dec:
            self.dec = ContentDecoder(opt.n_downsample, 0, dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)

    def forward(self, inputs, pixel_pos=None):
        content = self.enc(inputs)
        bs,c,h,w = content.size()
        content = self.transformer_enc(content.flatten(2).permute(2, 0, 1), src_pos=pixel_pos)
        harmonized = self.dec(content.permute(1, 2, 0).view(bs, c, h, w))
        return harmonized

class DHTGenerator(nn.Module):
    def __init__(self, opt=None):
        super(DHTGenerator, self).__init__()
        self.reflectance_dim = 256
        self.device = opt.device
        self.reflectance_enc = ContentEncoder(opt.n_downsample, 0, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.reflectance_dec = ContentDecoder(opt.n_downsample, 0, self.reflectance_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)

        self.reflectance_transformer_enc = transformer.TransformerEncoders(self.reflectance_dim, nhead=opt.tr_r_enc_head, num_encoder_layers=opt.tr_r_enc_layers, dim_feedforward=self.reflectance_dim*opt.dim_forward, activation=opt.tr_act)

        self.light_generator = GlobalLighting(light_element=opt.light_element,light_mlp_dim=self.reflectance_dim, opt=opt)
        self.illumination_render= transformer.TransformerDecoders(self.reflectance_dim, nhead=opt.tr_i_dec_head, num_decoder_layers=opt.tr_i_dec_layers, dim_feedforward=self.reflectance_dim*opt.dim_forward, activation=opt.tr_act)
        self.illumination_dec = ContentDecoder(opt.n_downsample, 0, self.reflectance_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)
        self.opt = opt
    def forward(self, inputs=None, image=None, pixel_pos=None, patch_pos=None, mask_r=None, mask=None, layers=[], encode_only=False):
        r_content = self.reflectance_enc(inputs)
        bs,c,h,w = r_content.size()

        reflectance = self.reflectance_transformer_enc(r_content.flatten(2).permute(2, 0, 1), src_pos=pixel_pos, src_key_padding_mask=None)
        
        light_code, light_embed = self.light_generator(image, pos=patch_pos, mask=mask, use_mask=self.opt.light_use_mask)
        illumination = self.illumination_render(light_code, reflectance, src_pos=light_embed, tgt_pos=pixel_pos, src_key_padding_mask=None, tgt_key_padding_mask=None)

        reflectance = reflectance.permute(1, 2, 0).view(bs, c, h, w)
        reflectance = self.reflectance_dec(reflectance)
        reflectance = reflectance / 2 +0.5

        illumination = illumination.permute(1, 2, 0).view(bs, c, h, w)
        illumination = self.illumination_dec(illumination)
        illumination = illumination / 2 + 0.5
        
        harmonized = reflectance*illumination
        return harmonized, reflectance, illumination


class GlobalLighting(nn.Module):
    def __init__(self, light_element=27, light_mlp_dim=8, norm=None, activ=None, pad_type='zero', opt=None):
    
        super(GlobalLighting, self).__init__()
        self.light_with_tre = opt.light_with_tre

        patch_size = opt.patch_size
        image_size = opt.crop_size
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = opt.input_nc * patch_size ** 2
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        )
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, light_mlp_dim),
        )
        dim = light_mlp_dim
        if opt.light_with_tre:
            self.transformer_enc = transformer.TransformerEncoders(dim, nhead=opt.tr_l_enc_head, num_encoder_layers=opt.tr_l_enc_layers, dim_feedforward=dim*2, dropout=0.0, activation=opt.tr_act)
        self.transformer_dec = transformer.TransformerDecoders(dim, nhead=opt.tr_l_dec_head, num_decoder_layers=opt.tr_l_dec_layers, dim_feedforward=dim*2, dropout=0.0, activation=opt.tr_act)
        self.light_embed = nn.Embedding(light_element, dim)

    def forward(self, inputs, mask=None, pos=None, multiple=False, use_mask=False):
        b,c,h,w = inputs.size()
        light_embed = self.light_embed.weight.unsqueeze(1).repeat(1, b, 1)
        tgt = torch.zeros_like(light_embed)
        input_patch = self.patch_embedding(self.to_patch(inputs))
        input_patch = input_patch.permute(1,0,2)
        src_key_padding_mask = None
        if use_mask:
            mask_patch = self.to_patch(mask)
            mask_sum = torch.sum(mask_patch, dim=2)
            src_key_padding_mask = mask_sum.to(bool)
        if self.light_with_tre:
            input_patch = self.transformer_enc(input_patch, src_pos=pos, src_key_padding_mask=src_key_padding_mask)
        light = self.transformer_dec(input_patch, tgt, src_pos=pos, tgt_pos=light_embed, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=None)        
        return light, light_embed

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
           
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm='ln', activation=activ, pad_type=pad_type)]
        if not dim == output_dim:
            self.model += [Conv2dBlock(dim, output_dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ContentDecoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type):
        super(ContentDecoder, self).__init__()
        self.model = []
        dim = input_dim
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]

        # upsampling blocks
        for i in range(n_downsample):
            self.model += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)
            ]
            dim //= 2

        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.norm_type = norm
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'adain_ori':
            self.norm = AdaptiveInstanceNorm2d_IN(norm_dim)
        elif norm == 'remove_render':
            self.norm = RemoveRender(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(ConvTranspose2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'adain_ori':
            self.norm = AdaptiveInstanceNorm2d_IN(norm_dim)
        elif norm == 'adain_dyna':
            self.norm = AdaptiveInstanceNorm2d_Dyna(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias))
        else:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

"""
加入discriminator后需要加入的类和方法
"""
class Identity(nn.Module):
    def forward(self, x):
        return x
    
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(int(gpu_ids[0]))
        gpu_ids_int = []
        for c in gpu_ids:
            if c.isdigit():
                gpu_ids_int.append(int(c))
        net = torch.nn.DataParallel(net, gpu_ids_int)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def define_D(opt, input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    
    net = networks_init.init_weights(net, init_type, init_gain)
    net = networks_init.build_model(opt, net)

    return net



##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)