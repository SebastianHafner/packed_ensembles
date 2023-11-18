from torch.optim import lr_scheduler
from utils.experiment_manager import CfgNode


def get_scheduler(cfg: CfgNode, optimizer):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    scheduler = cfg.TRAINER.LR_SCHEDULER
    if scheduler.TYPE == 'none':
        scheduler = None
    elif scheduler.TYPE == 'linear':
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(cfg.TRAINER.EPOCHS - 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif scheduler.TYPE == 'step':
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
        step_size = cfg.TRAINER.EPOCHS // 3
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif scheduler.TYPE == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=scheduler.MILESTONES, gamma=scheduler.GAMMA)
    elif scheduler.TYPE == 'cosineannealing':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAINER.EPOCHS)
    else:
        raise Exception('Unkown learning rate scheduler!')
    return scheduler
