import argparse
import os
import re
import scipy.misc
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Road Detection Image Preprocess')
parser.add_argument('data', metavar='in_DIR',
                    help='path to dataset')
parser.add_argument('output', metavar='out_DIR',
                    help='path to output file directory')

def main():
    args = parser.parse_args()

    # Data loading code
    image_paths = glob(os.path.join(args.data, '*_road_*.png'))

    if args.output is None:
        args.output = args.data

    for img_name in tqdm(image_paths):
        image = scipy.misc.imread(img_name)
        image = image[image.shape[0]-288:, image.shape[1]//2-192:image.shape[1]//2+192]
        scipy.misc.imsave(os.path.join(args.output, os.path.basename(img_name)), image)
    

if __name__ == '__main__':
    main()