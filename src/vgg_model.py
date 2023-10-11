# -* UTF-8 *-
'''
==============================================================
@Project -> File : pytorch -> vgg_model.py
@Author : yge
@Date : 2023/4/4 18:54
@Desc :

==============================================================
'''
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from torch.utils.tensorboard import SummaryWriter

from src.model import VGG
from torch.utils.data import DataLoader

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("mps")
device_cpu = torch.device("cpu")
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='../dataset/cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='../dataset/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

net = VGG('VGG19')
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.002,
                      momentum=0.9, weight_decay=5e-4)

writer = SummaryWriter(log_dir="../board/vgg")
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        outputs = outputs.to(device_cpu)
        targets = targets.to(device_cpu)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    writer.add_scalar("train_loss",train_loss/len(trainset),epoch)
    writer.add_scalar("train_acc", correct/len(trainset), epoch)

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            outputs = outputs.to(device_cpu)
            targets = targets.to(device_cpu)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        writer.add_scalar("test_loss", test_loss / len(testset), epoch)
        writer.add_scalar("test_acc", correct / len(testset), epoch)

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


if __name__ == "__main__":
    start = time.time()
    for epoch in range(start_epoch, start_epoch+20):
        train(epoch)
        test(epoch)
    torch.save(net,"../saved_models/vgg19_01.pth")
    writer.close()
    end = time.time()
    print(f"***** total time:{end-start}")