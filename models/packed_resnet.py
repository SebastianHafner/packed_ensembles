# https://github.com/ENSTA-U2IS/torch-uncertainty/blob/main/torch_uncertainty/models/resnet/packed.py
from typing import Any, Dict, List, Type, Union

import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from models.packed_layers import PackedConv2d, PackedLinear


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, alpha: float = 2, num_estimators: int = 4,
                 gamma: int = 1, groups: int = 1):
        super(BasicBlock, self).__init__()

        # No subgroups for the first layer
        self.conv1 = PackedConv2d(in_planes, planes, kernel_size=3, alpha=alpha, num_estimators=num_estimators,
                                  groups=groups, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * alpha)
        self.conv2 = PackedConv2d(
            planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes * alpha)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                PackedConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    groups=groups,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes * alpha),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int = 1,
            alpha: float = 2,
            num_estimators: int = 4,
            gamma: int = 1,
            groups: int = 1,
    ):
        super(Bottleneck, self).__init__()

        # No subgroups for the first layer
        self.conv1 = PackedConv2d(
            in_planes,
            planes,
            kernel_size=1,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=1,  # No groups from gamma in the first layer
            groups=groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes * alpha)
        self.conv2 = PackedConv2d(
            planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes * alpha)
        self.conv3 = PackedConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes * alpha)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                PackedConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    groups=groups,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes * alpha),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class _PackedResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            num_blocks: List[int],
            in_channels: int,
            num_classes: int,
            num_estimators: int,
            alpha: int = 2,
            gamma: int = 1,
            groups: int = 1,
            style: str = "imagenet",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.alpha = alpha
        self.gamma = gamma
        self.groups = groups
        self.num_estimators = num_estimators
        self.in_planes = 64
        block_planes = self.in_planes

        if style == "imagenet":
            self.conv1 = PackedConv2d(
                self.in_channels,
                block_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=1,  # No groups for the first layer
                groups=groups,
                bias=False,
                first=True,
            )
        else:
            self.conv1 = PackedConv2d(
                self.in_channels,
                block_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=1,  # No groups for the first layer
                groups=groups,
                bias=False,
                first=True,
            )

        self.bn1 = nn.BatchNorm2d(block_planes * alpha)

        if style == "imagenet":
            self.optional_pool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1
            )
        else:
            self.optional_pool = nn.Identity()

        self.layer1 = self._make_layer(
            block,
            block_planes,
            num_blocks[0],
            stride=1,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
        )
        self.layer2 = self._make_layer(
            block,
            block_planes * 2,
            num_blocks[1],
            stride=2,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
        )
        self.layer3 = self._make_layer(
            block,
            block_planes * 4,
            num_blocks[2],
            stride=2,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
        )
        self.layer4 = self._make_layer(
            block,
            block_planes * 8,
            num_blocks[3],
            stride=2,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = PackedLinear(
            block_planes * 8 * block.expansion,
            num_classes,
            alpha=alpha,
            num_estimators=num_estimators,
            last=True,
        )

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            num_blocks: int,
            stride: int,
            alpha: float,
            num_estimators: int,
            gamma: int,
            groups: int,
    ) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    groups=groups,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = rearrange(
            out, "e (m c) h w -> (m e) c h w", m=self.num_estimators
        )

        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = rearrange(out, '(m b) c -> m b c', m=self.num_estimators)
        return out

    def check_config(self, config: Dict[str, Any]) -> bool:
        """Check if the pretrained configuration matches the current model."""
        return (
                (config["alpha"] == self.alpha)
                * (config["gamma"] == self.gamma)
                * (config["groups"] == self.groups)
                * (config["num_estimators"] == self.num_estimators)
        )


def PackedResNet18(in_channels: int, num_estimators: int, alpha: int, gamma: int, num_classes: int, groups: int,
                   style: str = "imagenet") -> _PackedResNet:
    net = _PackedResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], in_channels=in_channels,
                        num_estimators=num_estimators, alpha=alpha, gamma=gamma, groups=groups, num_classes=num_classes,
                        style=style)
    return net


def PackedResNet50(in_channels: int, num_estimators: int, alpha: int, gamma: int, num_classes: int, groups: int,
                   style: str = "imagenet") -> _PackedResNet:
    net = _PackedResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        groups=groups,
        num_classes=num_classes,
        style=style,
    )
    return net
