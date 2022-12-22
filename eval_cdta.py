import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import math
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from utils.pretrained_models import *
from utils.encoder import *
from utils.cdta import *
from utils.val_dataloader import *

parser = argparse.ArgumentParser(description='eval CDTA')
parser.add_argument('-d', '--dataset', nargs='+', type=str)
parser.add_argument('-a', '--arch', type=str)
parser.add_argument('-g', '--gpu', default=0, type=int)
parser.add_argument('-b', '--batch-size', default=50, type=int)
parser.add_argument('--pretrained', 
                    default='./pretrained/surrogate/simsiam_bs256_100ep_cst.tar',
                    type=str)

parser.add_argument('--eps', default=16/255, type=float)
parser.add_argument('--nb-iter', default=30, type=int)
parser.add_argument('--step-size', default=4/255, type=float)


def enc_attack_test(dataset, arch, device, adversary, bs):
    if arch.startswith('inception'):
        val_loader = get_val_dataloader(dataset, bs, image_size=299)
    else:
        val_loader = get_val_dataloader(dataset, bs)

    model_t = read_from_checkpoint(arch, dataset, device)

    tot, acc = 0, 0
    tot_adv, acc_adv = 0, 0

    for x, y in val_loader:
        img, label = x.to(device), y.to(device)
        
        pred = model_t(normalize(img))
        pred = torch.argmax(pred, dim=1)

        pred_acc_v = pred == label
        
        tot += pred_acc_v.shape[0]
        acc += pred_acc_v.sum()
        tot_adv += pred_acc_v.sum()

        xadv = adversary.perturb(img)
        pred_adv = model_t(normalize(xadv))
        pred_adv = torch.argmax(pred_adv, dim=1)

        adv_acc_v = pred_adv == label
        for i in range(adv_acc_v.shape[0]):
            adv_acc_v[i] = adv_acc_v[i] and pred_acc_v[i]

        acc_adv += adv_acc_v.sum()

        if tot >= 500:
            break

    asr = 1 - acc_adv / tot_adv
    acc = acc / tot
    print(f'{dataset} - {arch}: ASR = {asr * 100:.2f}%')
    
def main():
    args = parser.parse_args()
    print(args)

    device = torch.device(f'cuda:{args.gpu}')
    
    enc = AttackEncoder(args.pretrained)
    enc.eval()
    enc = enc.to(device)

    adversary = CDTAttack(
        enc, 
        eps=args.eps, 
        nb_iter=args.nb_iter, 
        alpha=args.step_size
    )
    
    enc_attack_test(' '.join(args.dataset), args.arch, device, adversary, args.batch_size)

if __name__ == '__main__':
    main()