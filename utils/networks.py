import torch
from pathlib import Path
from utils.experiment_manager import CfgNode

import models
from models.packed_resnet import *


def create_network(cfg: CfgNode):
    if cfg.MODEL.TYPE == 'resnet':
        if cfg.MODEL.RESNET_SIZE == 18:
            if cfg.MODEL.ENSEMBLE:
                if cfg.MODEL.PACKED:
                    net = PackedResNet18(cfg.MODEL.IN_CHANNELS, num_estimators=cfg.MODEL.NUM_ESTIMATORS,
                                         alpha=cfg.MODEL.ALPHA, gamma=cfg.MODEL.GAMMA,
                                         num_classes=cfg.MODEL.OUT_CHANNELS, groups=1)
                else:
                    net = models.DeepEnsembleResNet18(cfg.MODEL.NUM_ESTIMATORS)
            else:
                net = models.ResNet18()
        elif cfg.MODEL.RESNET_SIZE == 50:
            if cfg.MODEL.ENSEMBLE:
                if cfg.MODEL.PACKED:
                    net = PackedResNet50(cfg.MODEL.IN_CHANNELS, num_estimators=cfg.MODEL.NUM_ESTIMATORS,
                                         alpha=cfg.MODEL.ALPHA, gamma=cfg.MODEL.GAMMA,
                                         num_classes=cfg.MODEL.OUT_CHANNELS, groups=1)
                else:
                    net = models.DeepEnsembleResNet50(cfg.MODEL.NUM_ESTIMATORS)
            else:
                net = models.ResNet50()
        else:
            raise NotImplementedError()
        # if not cfg.MODEL.PACKED:
        #     # net = ResNet(cfg)
        #     net = resnet18(in_channels=cfg.MODEL.IN_CHANNELS, num_classes=cfg.MODEL.OUT_CHANNELS)
        # else:
        #     net = packed_resnet18(
        #         in_channels=cfg.MODEL.IN_CHANNELS,
        #         num_estimators=cfg.MODEL.NUM_ESTIMATORS,
        #         alpha=cfg.MODEL.ALPHA,
        #         gamma=cfg.MODEL.GAMMA,
        #         num_classes=cfg.MODEL.OUT_CHANNELS,
        #     )
    else:
        raise Exception(f'Unknown network ({cfg.MODEL.TYPE}).')
    # return nn.DataParallel(net)
    return nn.DataParallel(net)


def save_checkpoint(network, optimizer, epoch, cfg: CfgNode, run = None):
    if run:
        save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_run_{run}.pt'
    else:
        save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}.pt'
    checkpoint = {
        'epoch': epoch,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(cfg: CfgNode, device: torch.device, net_file: Path = None):
    net = create_network(cfg)
    net.to(device)

    if not net_file:
        net_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}.pt'

    checkpoint = torch.load(net_file, map_location=device)
    # optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)
    net.load_state_dict(checkpoint['network'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    return net


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
