from utils.experiment_manager import CfgNode
from torch import optim


def get_optimizer(cfg: CfgNode, net):
    optimizer = cfg.TRAINER.OPTIMIZER
    if optimizer.TYPE == 'adamw':
        return optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)
    elif optimizer.TYPE == 'sdg':
        weight_decay = float(optimizer.WEIGHT_DECAY)
        return optim.SGD(net.parameters(), lr=cfg.TRAINER.LR, momentum=0.9, weight_decay=5e-4)
        # return optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    else:
        raise Exception('Unkown optimizer!')
