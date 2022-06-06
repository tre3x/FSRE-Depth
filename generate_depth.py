import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.kitti_dataset import KittiDataset
from networks.cma import CMA    
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder
from options import Options
from utils.depth_utils import disp_to_depth
from torchsummary import summary

from PIL import Image
from matplotlib import pyplot as plt

cv2.setNumThreads(0)

def generate(opt, save):
    if opt.ext_disp_to_eval is None:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        encoder_dict = torch.load(encoder_path, map_location=torch.device('cpu'))
        encoder = ResnetEncoder(num_layers=opt.num_layers)
        if not opt.no_cma:
            depth_decoder = CMA(encoder.num_ch_enc, opt=opt)
            decoder_path = os.path.join(opt.load_weights_folder, "decoder.pth")
        else:
            depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=opt.scales, opt=opt)
            decoder_path = os.path.join(opt.load_weights_folder, "decoder.pth")
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))
        
        encoder.eval()
        depth_decoder.eval()
        pred_disps = []
        models = {}

        models['encoder'] = encoder
        models['depth'] = depth_decoder

        img = cv2.imread(opt.data_path)
        img = cv2.resize(img, (1248, 384))
        print(img.shape)
        img = img.reshape(-1, img.shape[0], img.shape[1], img.shape[2],)
        img = np.rollaxis(img, 3, 1)
        img = torch.tensor(img)
        print("Input image shape : {}".format(img.shape))
        features = models['encoder'](img)
        print("Encoder Output Dims : {}, {}, {}, {}, {}".format(features[0].shape, features[1].shape, features[2].shape, features[3].shape, features[4].shape))
        if not opt.no_cma:
            output, _ = models['depth'](features)
        else:
            output = models["depth"](features)
        pred_disp = output[("disp", 0)]
        pred_disp, _ = disp_to_depth(pred_disp, opt.min_depth, opt.max_depth)
        pred_disp = pred_disp.cpu()[:, 0].detach().numpy()
        '''
        print("**************************")
        print(models['encoder'](img))
        print("**************************")
        print(models['depth'])
        print("**************************")
        '''
        print(pred_disp.shape)

        plt.imshow(pred_disp[0], interpolation='nearest')
        plt.show()


if __name__=='__main__':
    options = Options()
    generate(options.parse(), save = False)