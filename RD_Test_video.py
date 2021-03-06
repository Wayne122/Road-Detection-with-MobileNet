import argparse
import os
import random
import re
import scipy.misc
import imageio
from glob import glob
import numpy as np
import math
from tqdm import tqdm
from PIL import Image
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from inference_depth import inference_depth

parser = argparse.ArgumentParser(description='PyTorch Road Detection Testing')
parser.add_argument('mode', metavar='MODE', choices=['RGB', 'depth'],
                    help='output mode')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('model_weight', metavar='model_DIR',
                    help='path to model.pth')
parser.add_argument('--output', metavar='DIR',
                    help='path to output file directory')

class Net(nn.Module):
    def __init__(self, pretrained=True):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return [nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True)]

        def conv_dw(inp, oup, stride):
            return [nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True)]

        def make_layers():
            model = []
            model += [nn.BatchNorm2d(4)]
            model += conv_bn(  4,  32, 2)
            model += conv_dw( 32,  64, 1)
            model += conv_dw( 64, 128, 2)
            model += conv_dw(128, 128, 1)
            model += conv_dw(128, 256, 2)
            model += conv_dw(256, 256, 1)
            model += conv_dw(256, 512, 2)
            model += conv_dw(512, 512, 1)
            model += conv_dw(512, 512, 1)
            model += conv_dw(512, 512, 1)
            model += conv_dw(512, 512, 1)
            model += conv_dw(512, 512, 1)
            model += conv_dw(512, 1024, 2)
            model += conv_dw(1024, 1024, 1)
            return nn.Sequential(*model)

        self.model = make_layers()
        if pretrained:
            self.load_state_dict(torch.load('./mobilenet_v1_1.0_224.pth'), strict=False)

    def forward(self, x):
        output = {}
        for idx, m in enumerate(self.model):
            x = m(x)
            if idx in (9, 21, 33, 69, 81):
                output["x%d"%(idx+1)] = x
        return output

class FCN(nn.Module):
    def __init__(self, pretrained_net):
        super().__init__()
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 2, kernel_size=1) 
        # classifier is 1x1 conv, to reduce channels from 32 to 2

    def forward(self, x):
        output = self.pretrained_net(x)
        x82 = output['x82']
        x70 = output['x70']
        x34 = output['x34']
        x22 = output['x22']
        x10 = output['x10']

        score = self.bn1(self.relu(self.deconv1(x82)))
        score = score + x70
        score = self.bn2(self.relu(self.deconv2(score)))
        score = score + x34
        score = self.bn3(self.relu(self.deconv3(score)))
        score = score + x22
        score = self.bn4(self.relu(self.deconv4(score)))
        score = score + x10
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score

def gen_data_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :return:
    """
    # load model
    weights = torch.load('./checkpoint_ep0.pth.tar')
    model = weights['model'].module
    model.load_state_dict(weights['state_dict'])
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    reader = imageio.get_reader(data_folder)
    def get_data_fn():
        """
        Create batches of training data
        :param
        :return: testing data
        """
        global t
        for image_file in reader:
            t = time.time()
            RGB_image = np.array(Image.fromarray(image_file).crop((160, 0, 1120, 720)).resize((640, 480)))
            depth_image = inference_depth(model, RGB_image, None, False)[0]
            
            yield np.concatenate((RGB_image, depth_image), axis=2)[None, :], image_file

    return get_data_fn, reader.get_meta_data()['fps']*reader.get_meta_data()['duration']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    global args, image_shape, mode, t, inftime
    args = parser.parse_args()
    image_shape = (480, 640)
    mode = args.mode
    t = 0
    inftime = AverageMeter()

    # create model
    print("=> creating model mobilenet")
    pre_model = Net()
    model = FCN(pre_model)
    model = torch.nn.DataParallel(model).cuda()

    print("=> loading model param '{}'".format(args.model_weight))
    ckpt = torch.load(args.model_weight)
    model.load_state_dict(ckpt['state_dict'])
    print("epoch: ", ckpt['epoch'])
    print('MaxF: ', ckpt['best_MaxF'])

    cudnn.benchmark = True

    # Data loading code
    data_loader, data_len = gen_data_function(args.data, image_shape)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print("=> inferring")
    inference(args.mode, data_loader, model, args.output, data_len)
    print('Done! inference speed: ', inftime.avg)

def inference(mode, data_loader, model, out_dir, data_len):
    model.eval()

    with torch.no_grad():
        with tqdm(total=data_len) as pbar:
            for i, (RGBD, _) in enumerate(data_loader()):
                input_var = torch.from_numpy(RGBD).float().permute(0, 3, 1, 2).cuda()

                output = model(input_var)
                inftime.update(1 / (time.time()-t))
                output = (output - torch.min(output)) / (torch.max(output) - torch.min(output))

                if mode.endswith('RGB'):
                    output_image = RGBD[0, :, :, :3]
                    output = np.array(output.cpu()[0][1])
                    zeros = np.zeros(output.shape)
                    output_image[np.concatenate((zeros[..., None], output[..., None], zeros[..., None]), axis=2) > 0.7] = 255
                else:
                    output_image = np.zeros(image_shape + (3,))

                    for y in range(output_image.shape[0]):
                        for x in range(output_image.shape[1]):
                            if output[0][1][y][x] <= 0.7:
                                output_image[y][x] = np.array([255, 0, 0])
                            else:
                                output_image[y][x] = np.array([255, 0, 255])
                scipy.misc.imsave(os.path.join(out_dir, os.path.basename(args.data).replace('.mov', '_'+'%06d'%i+'.png')), output_image)
                pbar.update()


if __name__ == '__main__':
    main()