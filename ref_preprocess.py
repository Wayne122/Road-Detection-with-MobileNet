import argparse
import os
from shutil import copyfile

parser = argparse.ArgumentParser(description='PyTorch Road Detection Reference Preprocess')
parser.add_argument('ref_file', metavar='in_DIR',
                    help='path to ref_file')
parser.add_argument('data', metavar='data_DIR',
                    help='path to data root directory')
parser.add_argument('output', metavar='out_DIR',
                    help='path to output file directory')
                    
def main():
    args = parser.parse_args()

    with open(os.path.join(args.ref_file, 'um_mapping.txt'), 'r') as f:
        for i, file in enumerate(f.readlines()):
            try:
                copyfile(os.path.join(args.data, file.split()[0]+'_sync_02', file.split()[1]+'.png'), os.path.join(args.output, 'um_'+'%06d'%i+'.png'))
                copyfile(os.path.join(args.data, file.split()[0]+'_sync_02', file.split()[1]+'.npy'), os.path.join(args.output, 'um_'+'%06d'%i+'.npy'))
            except:
                print('File "' + os.path.join(args.data, file.split()[0]+'_sync_02', file.split()[1]+'.npy'), os.path.join(args.output, 'um_'+'%06d'%i+'.npy') + '" not found.')

    with open(os.path.join(args.ref_file, 'umm_mapping.txt'), 'r') as f:
        for i, file in enumerate(f.readlines()):
            try:
                copyfile(os.path.join(args.data, file.split()[0]+'_sync_02', file.split()[1]+'.png'), os.path.join(args.output, 'umm_'+'%06d'%i+'.png'))
                copyfile(os.path.join(args.data, file.split()[0]+'_sync_02', file.split()[1]+'.npy'), os.path.join(args.output, 'umm_'+'%06d'%i+'.npy'))
            except:
                print('File "' + os.path.join(args.data, file.split()[0]+'_sync_02', file.split()[1]+'.npy'), os.path.join(args.output, 'um_'+'%06d'%i+'.npy') + '" not found.')

    with open(os.path.join(args.ref_file, 'uu_mapping.txt'), 'r') as f:
        for i, file in enumerate(f.readlines()):
            try:
                copyfile(os.path.join(args.data, file.split()[0]+'_sync_02', file.split()[1]+'.png'), os.path.join(args.output, 'uu_'+'%06d'%i+'.png'))
                copyfile(os.path.join(args.data, file.split()[0]+'_sync_02', file.split()[1]+'.npy'), os.path.join(args.output, 'uu_'+'%06d'%i+'.npy'))
            except:
                print('File "' + os.path.join(args.data, file.split()[0]+'_sync_02', file.split()[1]+'.npy'), os.path.join(args.output, 'um_'+'%06d'%i+'.npy') + '" not found.')
    

if __name__ == '__main__':
    main()