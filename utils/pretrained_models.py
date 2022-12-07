import torch
import torchvision.models as models


def read_from_checkpoint(model_name, dataset, device):
    assert model_name in ['resnet34', 'inception_v3', 'vgg16_bn', 'densenet161'], print(model_name)
    assert dataset in ['birds-400', 'comic books', 'food-101', 'oxford 102 flower'], print(dataset)
    
    path = f'./pretrained/target/{dataset}/{model_name}.pth.tar'
    
    if dataset == 'comic books':
        num_classes = 86
    elif dataset == 'oxford 102 flower':
        num_classes = 102
    elif dataset == 'birds-400':
        num_classes = 400
    elif dataset == 'food-101':
        num_classes = 101
        
    if model_name.startswith('inception'):
        net = models.__dict__['inception_v3'](aux_logits=False, init_weights=True, num_classes=num_classes)
    else:
        net = models.__dict__[model_name](num_classes=num_classes)
        
    net.load_state_dict(torch.load(path))
    net.eval()
    net = net.to(device)
    return net
