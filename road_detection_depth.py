import argparse
import os
import shutil
import time
import re
import scipy.misc
import random
from glob import glob
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

parser = argparse.ArgumentParser(description='PyTorch Road Detection Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batchsize', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

best_prec1 = 0
best_loss = 10

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
            return nn.Sequential(*model)

        self.model = make_layers()
        if pretrained:
            self.load_state_dict(torch.load('./mobilenet_v1_1.0_224.pth'), strict=False)

    def forward(self, x):
        output = {}
        for idx, m in enumerate(self.model):
            x = m(x)
            if idx in (8, 14, 26, 38, 74):
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
        x75 = output['x75']
        x39 = output['x39']
        x27 = output['x27']
        x15 = output['x15']
        x9 = output['x9']

        score = self.bn1(self.relu(self.deconv1(x75)))
        score = score + x39
        score = self.bn2(self.relu(self.deconv2(score)))
        score = score + x27
        score = self.bn3(self.relu(self.deconv3(score)))
        score = score + x15
        score = self.bn4(self.relu(self.deconv4(score)))
        score = score + x9
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score

def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'images', '*.jpg'))
        label_paths = {
            re.sub(r'_train_color.png', '.jpg', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt', '*_train_color.png'))}
        road_color = np.array([128, 64, 128])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            d_images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imread(image_file)
                d_image = np.load(image_file.replace('jpg', 'npy'))[0, 0]
                gt_image = scipy.misc.imread(gt_image_file)[:, :, :3]

                gt_bg = np.all(gt_image == road_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((np.invert(gt_bg), gt_bg), axis=2)

                images.append(image)
                d_images.append(d_image)
                gt_images.append(gt_image)

            images = np.array(images)
            d_images = np.array(d_images)[..., None]


            yield np.concatenate((images, d_images), axis=3), np.array(gt_images, 'uint8'), math.ceil(len(image_paths)/batch_size)

    return get_batches_fn

def main():
    global args, best_MaxF
    args = parser.parse_args()
    image_shape = (480, 640)
    best_MaxF = 0

    # create model
    print("=> creating model")
    pre_model = Net()
    model = FCN(pre_model)
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

   # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_MaxF = checkpoint['best_MaxF']
            model.load_state_dict(checkpoint['state_dict'], False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    data_dir = os.path.join(args.data, 'train')
    val_data_dir = os.path.join(args.data, 'val')

    data_loader = gen_batch_function(data_dir, image_shape)
    val_data_loader = gen_batch_function(val_data_dir, image_shape)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(data_loader, args.batchsize, model, criterion, optimizer, epoch)

        # evaluate on validation set
        MaxF = validate(val_data_loader, args.batchsize, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = MaxF > best_MaxF
        best_MaxF = max(MaxF, best_MaxF)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'mobilenet',
            'state_dict': model.state_dict(),
            'best_MaxF': best_MaxF,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(data_loader, batchsize, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, batchlength) in enumerate(data_loader(batchsize)):
        # measure data loading time
        data_time.update(time.time() - end)

        #target = target.cuda(async=True)
        input_var = torch.from_numpy(input).float().permute(0, 3, 1, 2).cuda()
        target_var = torch.from_numpy(target).float().permute(0, 3, 1, 2).cuda()

        # compute output
        output = model(input_var)
        output = (output - torch.min(output)) / (torch.max(output) - torch.min(output))
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch+1, i+1, batchlength, batch_time=batch_time,
                   data_time=data_time, loss=losses))

def validate(data_loader, batchsize, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    precs = AverageMeter()
    recs = AverageMeter()
    # switch to eval mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input_i, target, batchlength) in enumerate(data_loader(batchsize)):
            # measure data loading time
            data_time.update(time.time() - end)

            #target = target.cuda(async=True)
            input_var = torch.from_numpy(input_i).float().permute(0, 3, 1, 2).cuda()
            target_var = torch.from_numpy(target).float().permute(0, 3, 1, 2).cuda()

            # compute output
            output = model(input_var)
            output = (output - torch.min(output)) / (torch.max(output) - torch.min(output))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            out_np = output.permute(1, 0, 2, 3).cpu().numpy()
            out_np = out_np[1]
            output_bool = np.zeros(out_np.shape, dtype=bool)
            target_bool = target_var.permute(1, 0, 2, 3)[1].cpu().numpy().astype(bool)
            TP = 0
            FP = 0
            FN = 0
            for b in range(out_np.shape[0]):
                for y in range(out_np.shape[1]):
                    for x in range(out_np.shape[2]):
                        if out_np[b][y][x] > 0.7:
                            output_bool[b][y][x] = True
                            if target_bool[b][y][x]:
                                TP += 1
                            else:
                                FP += 1
                        elif target_bool[b][y][x]:
                            FN += 1

            prec = TP/(TP+FP) if (TP+FP) != 0 else 0.0
            rec = TP/(TP+FN) if (TP+FN) != 0 else 0.0

            precs.update(prec, batchsize)
            recs.update(rec, batchsize)
            MaxF = (2*precs.avg*recs.avg)/(precs.avg+recs.avg) if (precs.avg+recs.avg) > 0 else 0.0
            if i % args.print_freq == 0:
                print(
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Precision {prec.val:.4f} ({prec.avg:.4f})\t'
                    'Recall {rec.val:.4f} ({rec.avg:.4f})\t'
                    'MaxF {0}\t'.format(
                    MaxF, batch_time=batch_time, data_time=data_time, prec=precs, rec=recs))

    return MaxF

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
