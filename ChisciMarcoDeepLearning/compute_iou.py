import numpy as np
import argparse
import json

import pandas as pd
from PIL import Image
from os.path import join
import matplotlib.pyplot as plt
from pandas import *

MODEL = 'Deeplab'  # Adapt/ Deeplab
if MODEL == 'Adapt':
    PRED_DIR = '/tmp/pycharm_project_127/result/cityscapes_val_'
    START = 5000
    END = 150000
elif MODEL == 'Deeplab':
    PRED_DIR = '/tmp/pycharm_project_127/result_deeplabv2/cityscapes_val_'
    START = 1000
    END = 20000
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):
    """
    Compute IoU given the predicted colorized images and
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    image_path_list = devkit_dir + 'val.txt'
    label_path_list = devkit_dir + 'label.txt'
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [gt_dir + x for x in gt_imgs]
    models_num = list(range(args.start, args.end+1, args.start))
    list_mIoU = list()
    matrix_mIoU = np.empty((int(args.end/args.start), num_classes))
    for num in models_num:
        print('model number: ' + str(num))
        pred_imgs = open(image_path_list, 'r').read().splitlines()
        pred_imgs = [pred_dir + str(num) + '/' + x.split('/')[-1] for x in pred_imgs]

        for ind in range(len(gt_imgs)):
            pred = np.array(Image.open(pred_imgs[ind]))
            label = np.array(Image.open(gt_imgs[ind]))
            label = label_mapping(label, mapping)
            if len(label.flatten()) != len(pred.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                      len(pred.flatten()), gt_imgs[ind],
                                                                                      pred_imgs[ind]))
                continue
            hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
            if ind > 0 and ind % 10 == 0:
                print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))

        mIoUs = per_class_iu(hist)
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            matrix_mIoU[int(num / args.start - 1)][ind_class] = round(mIoUs[ind_class] * 100, 2)
        print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        list_mIoU.append(round(np.nanmean(mIoUs) * 100, 2))

    matrix_mIoU = pd.DataFrame(matrix_mIoU, index=models_num)
    print(name_classes)
    return list_mIoU, matrix_mIoU


def main(args):
    list_mIoU, matrix_mIoU = compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        print(matrix_mIoU)

    print(list_mIoU)

    models_num = list(range(args.start, args.end+1, args.start))
    plt.plot(models_num, list_mIoU)
    plt.xlabel('Model number')
    plt.ylabel('mIoU')
    plt.title('mIoU for all the models')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', default='/tmp/pycharm_project_127/data/Cityscapes/data/gtFine/val/', type=str,
                        help='directory which stores CityScapes val gt images')
    parser.add_argument('--pred_dir', default=PRED_DIR, type=str,
                        help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='/tmp/pycharm_project_127/dataset/cityscapes_list/',
                        help='base directory of cityscapes')
    parser.add_argument('--start', default=START,
                        help='first model number')
    parser.add_argument('--end', default=END,
                        help='last model number')
    args = parser.parse_args()
    main(args)
