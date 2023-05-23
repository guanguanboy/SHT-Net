import torch
import os
import itertools
import torch.nn.functional as F
from .base_model import BaseModel
from util import util
from . import harmony_networks as networks
from . import base_networks as networks_init
from thop import profile
from thop import clever_format
import numpy as np

class HTModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='HT', dataset_mode='ihd')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.postion_embedding = None
        self.loss_names = ['G','G_L1']
        self.visual_names = ['mask', 'harmonized','comp','real']
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G'] 
        self.opt.device = self.device
        self.netG = networks.define_G(opt.netG, opt.init_type, opt.init_gain, self.opt)
        
        print(self.netG)  

        if self.isTrain:
            util.saveprint(self.opt, 'netG', str(self.netG))  
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

        #self.evaluate_efficiency(image_size = 256)
        #self.evaluate_inference_speed(image_size=256)        

    def set_position(self, pos, patch_pos=None):
        b = self.opt.batch_size
        self.pixel_pos = pos.unsqueeze(0).repeat(b, 1, 1, 1).to(self.device)
        self.pixel_pos = self.pixel_pos.flatten(2).permute(2, 0, 1)
        if self.opt.pos_none:
            self.input_pos = None
        else:
            input_pos = self.PatchPositionEmbeddingSine(self.opt)
            self.input_pos = input_pos.unsqueeze(0).repeat(b, 1, 1, 1).to(self.device)
            self.input_pos = self.input_pos.flatten(2).permute(2, 0, 1)
    def set_input(self, input):
        
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.inputs = input['inputs'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.image_paths = input['img_path']

        self.revert_mask = 1-self.mask

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        pass

    def evaluate_efficiency(self,image_size = 256):
        size = image_size
        gt = torch.randn((1,3,size,size)).cuda()
        cond = torch.randn(1,3,size,size).cuda()
        mask = torch.randn(1,1,size,size).cuda()

        flops, params = profile(self.netG, inputs=(gt, cond, mask))
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
            _ = self.netG(gt, cond, mask)             
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


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        print(self.inputs.shape, self.input_pos.shape)
        self.harmonized = self.netG(inputs=self.inputs, pixel_pos=self.input_pos)
        if not self.isTrain:
            self.harmonized = self.comp*self.revert_mask + self.harmonized*self.mask
    def compute_G_loss(self):
        """Calculate L1 loss for the generator"""
        self.loss_G_L1 = self.criterionL1(self.harmonized, self.real)*self.opt.lambda_L1
        self.loss_G = self.loss_G_L1
        return self.loss_G

    def optimize_parameters(self):
        # forward
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    def PatchPositionEmbeddingSine(self, opt):
        temperature=10000
        if opt.stride == 1:
            feature_h = int(256/opt.ksize)
        else:
            feature_h = int((256-opt.ksize)/opt.stride)+1
        num_pos_feats = 256//2
        mask = torch.ones((feature_h, feature_h))
        y_embed = mask.cumsum(0, dtype=torch.float32)
        x_embed = mask.cumsum(1, dtype=torch.float32)

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        return pos
