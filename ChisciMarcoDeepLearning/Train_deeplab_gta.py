import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data, model_zoo

from dataset.gta5_dataset import GTA5DataSet
from model.Deeplabv2 import DeepLabv2
from utils.loss import CrossEntropy2d

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

BATCH_SIZE = 1
ITER_SIZE = 10
NUM_WORKERS = 4
DATA_DIRECTORY = '/tmp/pycharm_project_127/data/GTA5'
DATA_LIST_PATH = '/tmp/pycharm_project_127/dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'

LEARNING_RATE = 2.5e-4  # default
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 20000
POWER = 0.9  # to decrease learning rate
RESTORE_FROM = '/tmp/pycharm_project_127/MS_DeepLab_resnet_pretrained_COCO_init.pth'
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = '/tmp/pycharm_project_127/snapshots_deeplabv2/'
WEIGHT_DECAY = 0.0005


def get_arguments():
    """
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=1,
                        help="choose gpu device.")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():
    w_in, h_in = map(int, args.input_size.split(','))
    input_size = (w_in, h_in)

    cudnn.enabled = True
    gpu = args.gpu

    deeplabv2 = DeepLabv2(num_classes=args.num_classes)

    if args.restore_from[:4] == 'http':
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)

    new_params = deeplabv2.state_dict().copy()
    for i in saved_state_dict:
        # Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        # print i_parts
        if not args.num_classes == 19 or not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            # print i_parts

    deeplabv2.load_state_dict(new_params)

    deeplabv2.train()
    deeplabv2.cuda(args.gpu)

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(deeplabv2.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # final upsampling layer after deeplabv2
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    for i_iter in range(args.num_steps + 1):
        loss = 0
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        for sub_i in range(args.iter_size):
            # train deeplabv2

            _, batch = next(trainloader_iter)
            images, labels, _, _ = batch  # ignore other values
            images = Variable(images).cuda(args.gpu)

            pred = deeplabv2(images)
            pred = interp(pred)

            loss_temp = loss_calc(pred, labels, args.gpu)
            loss_temp = loss_temp / args.iter_size
            loss_temp.backward()

            # proper normalization
            loss += loss_temp.data.cpu().numpy()

        optimizer.step()

        print('exp = {}'.format(args.snapshot_dir))
        print(
            'iter = {0:8d}/{1:8d}, loss = {2:.3f}'.format(
                i_iter, args.num_steps, loss))

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(deeplabv2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))


if __name__ == '__main__':
    main()
