import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import model_zoo, data

from model.Segmentation_Network import Segmentation
from model.Deeplabv2 import DeepLabv2
from model.Discriminator import Discriminator
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
MODEL = 'Adapt'  # Adapt/ Deeplab
DATA_DIRECTORY = '/tmp/pycharm_project_127/data/Cityscapes/data'
DATA_LIST_PATH = '/tmp/pycharm_project_127/dataset/cityscapes_list/val.txt'

if MODEL == 'Adapt':
    SAVE_PATH = '/tmp/pycharm_project_127/result/cityscapes_val_'
    RESTORE_FROM = '/tmp/pycharm_project_127/snapshots/GTA5_'
elif MODEL == 'Deeplab':
    SAVE_PATH = '/tmp/pycharm_project_127/result_deeplabv2/cityscapes_val_'
    RESTORE_FROM = '/tmp/pycharm_project_127/snapshots_deeplabv2/GTA5_'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500  # Number of images in the validation set.

SET = 'val'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=1,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Which model is being used")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    gpu = args.gpu

    validationloader = data.DataLoader(
        cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False,
                          mirror=False, set=args.set),
        batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

    if args.model == 'Adapt':
        model = Segmentation(num_classes=args.num_classes)
        models_num = list(range(5000, 150001, 5000))
    elif args.model == 'Deeplab':
        model = DeepLabv2(num_classes=args.num_classes)
        models_num = list(range(1000, 20001, 1000))

    for num in models_num:
        print('model number: ' + str(num))

        if not os.path.exists(args.save + str(num)):
            os.makedirs(args.save + str(num))

        if (args.restore_from + str(num) + '.pth')[:4] == 'http':
            saved_state_dict = model_zoo.load_url(args.restore_from + str(num) + '.pth')
        else:
            saved_state_dict = torch.load(args.restore_from + str(num) + '.pth')

        model.load_state_dict(saved_state_dict)

        model.eval()
        model.cuda(gpu)

        for index, batch in enumerate(validationloader):
            if index % 100 == 0:
                print('%d processed' % index)
            image, _, name = batch
            if args.model == 'Adapt':
                output1, output2 = model(Variable(image, volatile=True).cuda(gpu))
                output = interp(output2).cpu().data[0].numpy()
            elif args.model == 'Deeplab':
                output = model(Variable(image, volatile=True).cuda(gpu))
                output = interp(output).cpu().data[0].numpy()

            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)                  #selecting class of each pixel

            output_col = colorize_mask(output)
            output = Image.fromarray(output)

            name = name[0].split('/')[-1]
            output.save('%s/%s' % (args.save + str(num), name))
            output_col.save('%s/%s_color.png' % (args.save + str(num), name.split('.')[0]))


if __name__ == '__main__':
    main()
