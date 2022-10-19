import os
import torch
from .model import Network

def get_network(args):
    return Network(args).to(args.device)


def load_checkpoint(args, net, optimizer=None):
    assert os.path.exists(args.init_weight
        ), f"Pretrained checkpoint not found! {args.init_weight} not exists."
    checkpoint = torch.load(args.init_weight)
    try:
        net.load_state_dict(checkpoint['model'])
    except:
        net.load_state_dict(checkpoint['FusionNet'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print(f'Successfully loaded pretrained model ! {args.init_weight}')
    return net, optimizer, start_epoch
