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

class lapswinhihModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
       
        parser.set_defaults(norm='instance', netG='LAPSWINHIH', dataset_mode='ihd')
        if is_train:
            parser.set_defaults(pool_size=0)
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.postion_embedding = None
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G','G_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
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
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        #self.evaluate_efficiency(image_size = 256)
        #self.evaluate_inference_speed(image_size=256)   

    def evaluate_efficiency(self,image_size = 256):
        size = image_size
        inputs = torch.randn((1,4,size,size)).cuda()
        image = torch.randn(1,3,size,size).cuda()
        pixel_position = torch.randn(4096,1,size).cuda()
        patch_position = torch.randn(1024,1,size).cuda()
        mask_r = torch.randn(1,1,64,64).cuda()
        mask = torch.randn(1,1,size,size).cuda()

        flops, params = profile(self.netG, inputs=(inputs, image, pixel_position,patch_position, mask_r, mask))
        flops, params = clever_format([flops, params], '%.3f')

        print('params=', params)
        print('FLOPs=',flops)

    def evaluate_inference_speed(self, image_size=256):
        size = image_size
        inputs = torch.randn((1,4,size,size)).cuda()
        image = torch.randn(1,3,size,size).cuda()
        pixel_position = torch.randn(4096,1,size).cuda()
        patch_position = torch.randn(1024,1,size).cuda()
        mask_r = torch.randn(1,1,64,64).cuda()
        mask = torch.randn(1,1,size,size).cuda()

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings=np.zeros((repetitions,1))
        #GPU-WARM-UP
        for _ in range(10):
            _ = self.netG(inputs, image, pixel_position,patch_position, mask_r, mask)             
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = self.netG(inputs, image, pixel_position,patch_position, mask_r, mask)  
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

    def set_position(self, pos, patch_pos=None):
        pass
        
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.inputs = input['inputs'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.image_paths = input['img_path']
        self.mask_r = F.interpolate(self.mask, size=[64,64])
        self.revert_mask = 1-self.mask

    def data_dependent_initialize(self, data):
        pass


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #image=self.comp*self.revert_mask
        #pixel_pos=self.pixel_pos.detach()
        #patch_pos=self.patch_pos.detach()
        #print(self.inputs.shape, image.shape,pixel_pos.shape, patch_pos.shape, self.mask_r.shape, self.mask.shape)

        self.harmonized = self.netG(inputs=self.inputs)
        if not self.isTrain:
            self.harmonized = self.comp*(1-self.mask) + self.harmonized*self.mask


    def compute_G_loss(self):
        """Calculate GAN and L1 loss for the generator"""
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
    
