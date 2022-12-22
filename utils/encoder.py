import torch
import torch.nn as nn
import torchvision.models as models
import utils.aresnet as aresnet


class AttackEncoder(nn.Module):
        
    def __init__(self, ckpt_path):
        super(AttackEncoder, self).__init__()
        
        model = aresnet.resnet50()
        checkpoint = torch.load(ckpt_path)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']

            for k in list(state_dict.keys()):
                if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                    state_dict[k[len("module.encoder."):]] = state_dict[k]
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        else:
            model.load_state_dict(checkpoint)
        
        model.fc = nn.Identity()
        self.net = model
        
    def forward(self, x):
        return self.net(x)

