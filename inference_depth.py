import torch
import errno
import shutil
import os.path as osp
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


def test():
    def mkdir_if_missing(directory):
        if not osp.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
        mkdir_if_missing(osp.dirname(fpath))
        torch.save(state, fpath)
        if is_best:
            shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

    test_save = False
    compared = True
    if compared:
        img1 = Image.open('./test_model/test_depth.png')
        img2 = Image.open('./test_model/test_True_depth.png')
        img1_np = np.array(img1)
        img2_np = np.array(img2)

        img1_np = np.load('./test_model/test_False.npy')
        img2_np = np.load('./test_model/test_True.npy')

        print(np.all(img1_np == img2_np))

    else:
        if test_save:
            checkpoint = torch.load('./results/mobilenet-nnconv5dw-skipadd-pruned.pth.tar')
            if type(checkpoint) is dict:
                model = checkpoint['model']

            weights = torch.load('./checkpoints/kitti/RMSE_log_Gradient_Loss/best_model.pth.tar')
            model.load_state_dict(weights['state_dict'])
            best_epoch = weights['epoch']

            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            model = torch.nn.DataParallel(model).cuda()

            save_checkpoint({
                'epoch': 0,
                'rmse': 'inf',
                'arch': 'fast-depth',
                'state_dict': model.module.state_dict(),
                'optimizer': None,
                'model': model}, False, osp.join('./test_model', 'checkpoint_ep' + str(0) + '.pth.tar'))

        else:
            weights = torch.load('./test_model/checkpoint_ep0.pth.tar')
            model = weights['model'].module
            model.load_state_dict(weights['state_dict'])
            best_epoch = weights['epoch']

            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            model = torch.nn.DataParallel(model).cuda()

        model.eval()


        test = './test_model/test.png'
        trans = transforms.ToTensor()
        cmap = plt.cm.viridis

        img = Image.open(test)
        tensor_img = trans(img)
        tensor_img = tensor_img.resize(1, 3, 288, 384).cuda()
        output = model(tensor_img)

        output = np.squeeze(output.data.cpu().numpy())

        np.save('./test_model/test_'+str(test_save), output)

        depth_relative = output / 80
        output = 255 * cmap(depth_relative)[:, :, :3]  # H, W, C
        output = Image.fromarray(output.astype('uint8'))
        output.save('./test_model/{}_depth{}'.format('test_'+str(test_save), '.png'))

def inference_depth(model, image, output_dir, writefile):

    trans = transforms.ToTensor()

    # image preprocess
    #img = Image.open(test)
    tensor_img = trans(image)
    tensor_img = tensor_img.resize(1, 3, 480, 640).cuda()

    # inference
    result = model(tensor_img)

    # output
    if writefile:
        cmap = plt.cm.viridis
        result = np.squeeze(result.data.cpu().numpy())
        depth_relative = result / 80
        result = 255 * cmap(depth_relative)[:, :, :3]  # H, W, C
        result = Image.fromarray(result.astype('uint8'))
        result.save(output_dir+'/{}_depth{}'.format('test', '.png'))
    else:
        return result.cpu().permute(0, 2, 3, 1).numpy()