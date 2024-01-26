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

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# https://github.com/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb
if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    sweep_dir = Path(cfg.PATHS.OUTPUT) / 'sweeps' / cfg.NAME
    sweep_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('=== Runnning on device: p', device)


    def run_training(sweep_cfg=None):

        with wandb.init(config=sweep_cfg):
            sweep_cfg = wandb.config

            # overwriting config parameters with sweep parameters
            cfg.RUN_NUM = sweep_cfg.run_num
            cfg.MODEL.NUM_ESTIMATORS = sweep_cfg.num_estimators
            cfg.MODEL.ALPHA = sweep_cfg.alpha
            cfg.MODEL.GAMMA = sweep_cfg.gamma

            # make training deterministic
            torch.manual_seed(cfg.SEED + cfg.RUN_NUM)
            np.random.seed(cfg.SEED + cfg.RUN_NUM)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            net = networks.create_network(cfg)
            net.to(device)

            optimizer = optimizers.get_optimizer(cfg, net)
            scheduler = schedulers.get_scheduler(cfg, optimizer)

            criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)

            # reset the generators
            dataset = torchvision.datasets.CIFAR10(root=Path(cfg.PATHS.DATASET), train=True, download=True,
                                                   transform=transform)
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
                wandb.log(
                    {'lr': scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR, 'epoch': epoch})

                start = timeit.default_timer()
                loss_set = []

                for i, (images, labels) in enumerate(dataloader):
                    net.train()
                    optimizer.zero_grad()

                    y_hat = net(images.to(device))
                    print(labels.shape)
                    print(y_hat.shape)
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


    if args.sweep_id is None:
        # Step 2: Define sweep config
        sweep_config = {
            'method': 'grid',
            'name': cfg.NAME,
            'metric': {'goal': 'maximize', 'name': 'best val f1'},
            'parameters':
                {
                    'run_num': {'values': [1, 2]},
                    'num_estimators': {'values': [2, 4]},
                    'alpha': {'values': [2, 4]},
                    'gamma': {'values': [2, 4]},
                }
        }

        # Step 3: Initialize sweep by passing in config or resume sweep
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project)
        # Step 4: Call to `wandb.agent` to start a sweep
        wandb.agent(sweep_id, function=run_training)
    else:
        # Or resume existing sweep via its id
        # https://github.com/wandb/wandb/issues/1501
        sweep_id = args.sweep_id
        wandb.agent(sweep_id, project=args.wandb_project, function=run_training)