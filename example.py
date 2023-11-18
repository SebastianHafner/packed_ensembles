from pathlib import Path

from torch import nn

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import ResNet
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.optimization_procedures import get_procedure

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':
    root = Path(__file__).parent.absolute().parents[1]

    args = init_args(ResNet, CIFAR10DataModule)
    
    net_name = f"{args.version}-resnet{args.arch}-cifar10"

    # datamodule
    args.root = str(root / "data")
    dm = CIFAR10DataModule(**vars(args))

    # model
    model = ResNet(
        num_classes=dm.num_classes,
        in_channels=3,
        loss=nn.CrossEntropyLoss(),
        optimization_procedure=get_procedure(
            f"resnet{args.arch}", "cifar10", args.version
        ),
        imagenet_structure=False,
        **vars(args),
    )

    cli_main(model, dm, root, net_name, args)