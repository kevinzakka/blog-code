import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler, WeightedRandomSampler


def get_data_loader(data_dir, batch_size, num_workers=3, pin_memory=False):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    dataset = MNIST(root=data_dir, train=True, download=True, transform=transform)

    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader


def get_test_loader(data_dir, batch_size, num_workers=3, pin_memory=False):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    dataset = MNIST(root=data_dir, train=False, download=True, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


class SmallConv(nn.Module):
    def __init__(self):
        super(SmallConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        out = out.view(-1, 320)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)


def accuracy(predicted, ground_truth):
    predicted = torch.max(predicted, 1)[1]
    total = len(ground_truth)
    correct = (predicted == ground_truth).sum().double()
    acc = 100 * (correct / total)
    return acc.item()


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        acc = accuracy(output, target)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 25 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), acc))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    data_dir = './data/'
    plot_dir = './imgs/'
    dump_dir = './dump/'

    # ensuring reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False

    GPU = False
    device = torch.device("cuda" if GPU else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if GPU else {}

    num_epochs = 5
    num_epochs_steady = 3
    learning_rate = 1e-3
    mom = 0.99
    batch_size = 64

    torch.manual_seed(SEED)

    # instantiate convnet
    model = SmallConv().to(device)

    # relu init
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in')

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)

    # instantiate loaders
    train_loader = get_data_loader(data_dir, batch_size, **kwargs)
    test_loader = get_test_loader(data_dir, 128, **kwargs)

    # transient training
    tic = time.time()
    for epoch in range(1, num_epochs+1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    toc = time.time()
    print("Time Elapsed: {}s".format(toc-tic))


if __name__ == '__main__':
    main()
