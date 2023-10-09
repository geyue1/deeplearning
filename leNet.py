# -* UTF-8 *-
'''
==============================================================
@Project -> File : pytorch -> leNet.py
@Author : yge
@Date : 2023/10/6 15:23
@Desc :

==============================================================
'''
import torch
from torch import nn, optim
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision import transforms


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            Conv2d(1, 6, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            # MaxPool2d(kernel_size=2),
            nn.AvgPool2d(kernel_size=2,stride=2),
            Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            Flatten(),
            Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            Linear(120, 84),
            nn.ReLU(),
            Linear(84, 10),
        )
    def forward(self,x):
        return self.model(x)

def load_data(batch):
    transform_ = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset_train = torchvision.datasets.MNIST(root="./dataset/mnist",download=True,
                                         train=True,
                                         transform=transform_)
    dataset_test = torchvision.datasets.MNIST(root="./dataset/mnist",download=True,
                                         train=False,
                                         transform=transform_)
    data_train = DataLoader(dataset=dataset_train,
                            batch_size=batch,
                            shuffle=True,
                            drop_last=False)
    data_test = DataLoader(
        dataset=dataset_test,
        batch_size=batch,
        shuffle=True,
        drop_last=False
    )
    return data_train,data_test

def train(net,lr,epoch,device,data,loss_fn):

    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    net.to(device)
    net.train()

    train_loss = 0
    losses = AverageMeter('Loss', ':.4e')
    correct = 0
    total = 0

    for batch_id, (inputs, targets) in enumerate(data):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*inputs.size(0)
        losses.update(loss.item(), inputs.size(0))
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # correct /= total

        if batch_id%50==0 or batch_id==len(data)-1:
            print(
                f"Train Epoch:{epoch+1} [{batch_id * len(inputs)}/{len(data)} {100. * (batch_id+1) / len(data):.2f}%]"
                f" Losss:{100. * train_loss / total:.2f}% Acc:{100. * correct/total:.2f}%")
    print(f"Train Epoch:{epoch+1} Losss:{100. * train_loss / (total):.2f}% Acc:{100. * correct/total :.2f}%")
    print(losses)

def test(net,device,data,loss_fn,epoch):
    test_loss = 0
    correct = 0
    total = 0
    global  best_acc
    net.to(device)
    net.eval()
    with torch.no_grad():
        for batch_id, (inputs, targets) in enumerate(data):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_id % 50 == 0 or batch_id == len(data) - 1:
                print(
                    f"Test Epoch:{epoch + 1} [{batch_id * len(inputs)}/{len(data)} {100. * (batch_id + 1) / len(data):.2f}%]"
                    f" Losss:{100. * test_loss / total:.2f}% Acc:{100. * correct/total:.2f}%")
        print(f"Test Epoch:{epoch + 1} Losss:{100. * test_loss / (total):.2f}% Acc:{100. * correct/total :.2f}%")
        temp = correct / total
        if temp>best_acc:
            best_acc = temp
            print(f"******best_acc={best_acc}")
            torch.save(net.state_dict(), './saved_models/leNet.pth')
def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    print('Device: {}'.format(device))
    return device

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == '__main__':
    lr=0.1
    batch=64
    device = get_device()
    epoch = 10
    data_train,data_test = load_data(batch)
    loss_fn = nn.CrossEntropyLoss()
    net = LeNet()
    best_acc = 0
    for e in range(epoch):
       train(net,lr=lr,epoch=e,device=device,data=data_train,loss_fn=loss_fn)
       test(net,device,data_test,loss_fn,e)

    print(f"best_acc={best_acc}")
