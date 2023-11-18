import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm

from pathlib import Path
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Allow reproducability
torch.manual_seed(0)
np.random.seed(0)

dataset_path = Path('C:/Users/shafner/datasets/cifar10')
output_path = Path('C:/Users/shafner/multitemporal_urban_mapping/output')

if __name__ == '__main__':

    epochs = 2

    # Normalize the images by the imagenet mean/std since the nets are pretrained
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [45000, 5000])
    test_set = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)

    num_classes = 10
    net = models.resnet101(pretrained=True)
    net.fc = nn.Linear(2048, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=True, num_workers=2)
    for epoch in range(epochs):
        running_loss = 0.0
        net.train()
        print(f'Training iteration {epoch}')
        for i, data in enumerate(tqdm(train_loader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        corrects = []
        net.eval()
        classified_right = 0
        print('Evaluating on validation set')
        for i, data in enumerate(tqdm(val_loader, 0)):
            with torch.no_grad():
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                _, pred_classes = torch.max(outputs, 1)

                loss = criterion(outputs, labels)
                classified_right += (pred_classes == labels).sum().item()

        acc = classified_right / len(val_set)

        print(f'Epoch {epoch}  Acc: {acc}')