import os.path
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from util import util

class RealHMDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.isTrain = opt.isTrain
        self.image_size = opt.crop_size
        data_root = opt.dataset_root
        self.image_small_list = []
        for root, dirs, files in os.walk(data_root  + "/vendor_testing_1/"):
            for file in files:
                if "small" in file:
                    self.image_small_list.append(os.path.join(root, file))
        for root, dirs, files in os.walk(data_root  + "/vendor_testing_2/"):
            for file in files:
                if "small" in file:
                    self.image_small_list.append(os.path.join(root, file))
                    
        for root, dirs, files in os.walk(data_root  + "/vendor_testing_3/"):
            for file in files:
                if "small" in file:
                    self.image_small_list.append(os.path.join(root, file))
        
        
        self.tfs = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = self.image_small_list[index]

        comp_path = path.replace("_small.jpg", "_composite.jpg")
        mask_path = path.replace("_small.jpg", "_mask.jpg")
        target_path = path.replace("_small.jpg", "_gt.jpg")

        comp = Image.open(comp_path).convert('RGB')
        real = Image.open(target_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')

        comp = tf.resize(comp, [self.image_size, self.image_size])
        mask = tf.resize(mask, [self.image_size, self.image_size])
        #mask = tf.resize(mask, [224, 224]) #对MAE训练，需要将这里修改为224,224
        real = tf.resize(real,[self.image_size,self.image_size])

        #apply the same transform to composite and real images
        comp = self.transform(comp)
        #mask = self.mask_transform(mask)
        mask = tf.to_tensor(mask)

        real = self.transform(real)

        inputs=torch.cat([comp,mask],0)
        
        return {'inputs': inputs, 'comp': comp, 'real': real,'img_path':path,'mask':mask}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_small_list)