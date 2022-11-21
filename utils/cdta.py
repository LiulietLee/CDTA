import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch import clamp
from utils.gaussian_smoothing import *
from utils.val_dataloader import normalize


class CDTAttack():

    def __init__(self, enc, eps=0.3, nb_iter=40, alpha=0.01):
        super(CDTAttack, self).__init__()
        self.enc = enc
        self.eps = eps
        self.nb_iter = nb_iter
        self.alpha = alpha

    def perturb(self, x):        
        x = x.detach().clone()

        vec = self.enc(normalize(x)) * -1
        y = [
            self.enc.net.feature1,
            self.enc.net.feature2,
            self.enc.net.feature3,
            vec,
        ]

        y = [_.detach().clone() for _ in y]

        xadv = self.perturb_iterative(x, y)

        return xadv.data
    
    def perturb_iterative(self, xvar, yvar):
        
        kernel = get_gaussian_kernel(15, 8).to(xvar.device)
        cos = nn.CosineSimilarity(dim=1)
        
        delta = torch.zeros_like(xvar)
        delta.requires_grad_()

        for _ in range(self.nb_iter):        
            xt = xvar.detach().clone()
            xt = self.random_transform(xt)
            xt.requires_grad_()

            outputs = self.enc(normalize(xt + delta))

            loss1 = self.mid_layer_loss(self.enc.net.feature1, yvar[0])
            loss2 = self.mid_layer_loss(self.enc.net.feature2, yvar[1])
            loss3 = self.mid_layer_loss(self.enc.net.feature3, yvar[2])
            loss4 = cos(outputs, yvar[3])
            loss = loss1 + loss2 + loss3 + loss4

            loss.backward()

            grad_sign = kernel(delta.grad.data.sign())
            delta.data = delta.data + self.alpha * grad_sign
            
            delta.data = clamp(delta.data, -self.eps, self.eps)
            delta.data = clamp(xvar.data + delta.data, 0., 1.) - xvar.data

            delta.grad.data.zero_()

        x_adv = clamp(xvar + delta, 0., 1.)
        return x_adv

    def random_transform(self, X):
        s = torch.randint(0, 3, (1,)).item()
        shape_size = X.shape[-1]

        if s == 0:
            extend = torch.randint(0, 10, (1,)).item()
            pad_resize = transforms.Compose([
                transforms.Pad(padding=extend),
                transforms.Resize(shape_size),
            ])
            return pad_resize(X)
        if s == 1:
            extend = torch.randint(10, 20, (1,)).item()
            resize_crop = transforms.Compose([
                transforms.Resize(shape_size + extend),
                transforms.RandomCrop(shape_size),
            ])
            return resize_crop(X)
        elif s == 2:
            extend = torch.randint(10, 20, (1,)).item()
            padding_crop = transforms.Compose([
                transforms.Pad(padding=extend),
                transforms.RandomCrop(shape_size)
            ])
            return padding_crop(X)
        else:
            return X

    def mid_layer_loss(self, xvar, yvar):
        return ((xvar - yvar) ** 2).sum() / (yvar ** 2).sum()

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)
