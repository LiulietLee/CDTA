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
parser.add_argument('--pretrained', default='./pretrained/surrogate/simsiam_bs256_100ep_cst.tar',
                    type=str)

parser.add_argument('--eps', default=16/255, type=float)
parser.add_argument('--nb-iter', default=30, type=int)
parser.add_argument('--alpha', default=4/255, type=float)


def enc_attack_test(dataset, arch, device, adversary):
    if arch.startswith('inception'):
        val_loader = get_val_dataloader(dataset, image_size=299)
    else:
        val_loader = get_val_dataloader(dataset)

    model_t = read_from_checkpoint(arch, dataset, device)

    tot, acc = 0, 0
    tot_adv, acc_adv = 0, 0
    batch_count = 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        for i in range(x.shape[0]):
            img = x[i].unsqueeze(0)
            label = y[i].unsqueeze(0)
            
            pred = model_t(normalize(img))
            pred = torch.argmax(pred, dim=1)
            
            tot += 1
            if pred == label:
                acc += 1
                tot_adv += 1

                xadv = adversary.perturb(img)
                pred_adv = model_t(normalize(xadv))
                pred_adv = torch.argmax(pred_adv, dim=1)
                if pred == pred_adv:
                    acc_adv += 1

        batch_count += 1
        if batch_count >= 5:
            break

    asr = 1 - acc_adv / tot_adv
    acc = acc / tot
    print(f'{dataset} - {arch}')
    print(f' ASR = {asr * 100:.2f}%')
    
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
        alpha=args.alpha
    )
    
    enc_attack_test(' '.join(args.dataset), args.arch, device, adversary)

if __name__ == '__main__':
    main()