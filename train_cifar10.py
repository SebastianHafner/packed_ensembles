import sys
import os
import timeit

import torch

from torch.utils import data as torch_data

import wandb
import numpy as np
from pathlib import Path

from utils import networks, loss_functions, evaluation, experiment_manager, parsers, schedulers, optimizers
import torchvision
import torchvision.transforms as transforms
from einops import rearrange

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def run_training(cfg):
    run = wandb.init(
        name=cfg.NAME,
        config=cfg,
        project='FDD3412_packed_ensembles',
        tags=['run', 'cifar10', 'packed', 'ensemble', ],
        mode='online' if not cfg.DEBUG else 'disabled',
    )

    net = networks.create_network(cfg)
    net.to(device)

    optimizer = optimizers.get_optimizer(cfg, net)
    scheduler = schedulers.get_scheduler(cfg, optimizer)

    criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)

    # reset the generators
    dataset = torchvision.datasets.CIFAR10(root=Path(cfg.PATHS.DATASET), train=True, download=True, transform=transform)
    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0
    # _ = evaluation.model_evaluation_cifar10(net, cfg, False, epoch_float)
    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')
        wandb.log({'lr': scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR, 'epoch': epoch})

        start = timeit.default_timer()
        loss_set = []

        for i, (images, labels) in enumerate(dataloader):
            net.train()
            optimizer.zero_grad()

            y_hat = net(images.to(device))
            labels = labels.long().to(device)
            if cfg.MODEL.ENSEMBLE:
                y_hat = rearrange(y_hat, 'm b c -> (m b) c')
                pe_labels = labels.repeat(cfg.MODEL.NUM_ESTIMATORS)
                loss = criterion(y_hat, pe_labels)
                # loss = torch.tensor([0], device=device, dtype=torch.float)
                # for y_hat_m in y_hat:
                #     loss += criterion(y_hat_m, labels)
            else:
                loss = criterion(y_hat, labels)
            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ_STEP == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')
                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'loss': np.mean(loss_set),
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set = []

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        if scheduler is not None:
            scheduler.step()

        if epoch % cfg.LOG_FREQ_EPOCH == 0:
            # _ = evaluation.model_evaluation_cifar10(net, cfg, True, epoch_float)
            _ = evaluation.model_evaluation_cifar10(net, cfg, False, epoch_float)

    networks.save_checkpoint(net, optimizer, cfg.RUN_NUM, cfg.TRAINER.EPOCHS, cfg)

    run.finish()


if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED + cfg.RUN_NUM)
    np.random.seed(cfg.SEED + cfg.RUN_NUM)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
