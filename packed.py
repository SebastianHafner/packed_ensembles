import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from torch_uncertainty.layers import PackedConv2d, PackedLinear


import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.set_num_threads(1)

DATASET_PATH = Path('C:/Users/shafner/datasets/cifar10')
OUTPUT_PATH = Path('C:/Users/shafner/pop_uncertainty/output')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PackedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        M = 4
        alpha = 2
        gamma = 1
        self.conv1 = PackedConv2d(3, 6, 5, alpha=alpha, num_estimators=M, gamma=gamma, first=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = PackedConv2d(6, 16, 5, alpha=alpha, num_estimators=M, gamma=gamma)
        self.fc1 = PackedLinear(16 * 5 * 5, 120, alpha=alpha, num_estimators=M, gamma=gamma)
        self.fc2 = PackedLinear(120, 84, alpha=alpha, num_estimators=M, gamma=gamma)
        self.fc3 = PackedLinear(84, 10 * M, alpha=alpha, num_estimators=M, gamma=gamma, last=True)

        self.num_estimators = M

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = rearrange(x, "e (m c) h w -> (m e) c h w", m=self.num_estimators)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def test_loader(loader):
    # get some random training images
    dataiter = iter(loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))


def train_packed_ensemble(packed_net, trainloader):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(packed_net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            pe_outputs = packed_net(inputs)
            pe_labels = labels.repeat(packed_net.num_estimators)
            loss = criterion(pe_outputs, pe_labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")

    torch.save(packed_net.state_dict(), OUTPUT_PATH / 'packed_net.pt')


if __name__ == '__main__':
    batch_size = 4

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",)
    net = Net()
    packed_net = PackedNet()

    file = OUTPUT_PATH / 'packed_net.pt'
    if not file.exists():
        train_packed_ensemble(packed_net, trainloader)

    packed_net.load_state_dict(torch.load(file))

    dataiter = iter(testloader)
    images, labels = next(dataiter)
    logits = packed_net(images)
    logits = rearrange(logits, "(n b) c -> b n c", n=packed_net.num_estimators)
    probs_per_est = F.softmax(logits, dim=-1)
    outputs = probs_per_est.mean(dim=1)

    _, predicted = torch.max(outputs, 1)

    print(
        "Predicted: ",
        " ".join(f"{classes[predicted[j]]:5s}" for j in range(batch_size)),
    )