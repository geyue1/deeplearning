# -* UTF-8 *-
'''
==============================================================
@Project -> File : pytorch -> pool_layer.py
@Author : yge
@Date : 2023/3/31 14:10
@Desc :

==============================================================
'''
import torch.nn
import torchvision.transforms
from PIL import Image
from torch import Tensor
import torch.nn.functional as f
import numpy as np
from torch.nn import Module, Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def test_max_pool(x):
    if not type(x) is Tensor:
        transform = torchvision.transforms.ToTensor()
        x = transform(x)

    max_pool = torch.nn.MaxPool2d(2,stride=1,padding=0)
    return max_pool(x)

class CNN(Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv2d = Conv2d(in_channels=3,
                             out_channels=3,
                             kernel_size=3,
                             stride=1,
                             padding=0)
        self.max_pool = MaxPool2d(kernel_size=2,
                                  stride=1,
                                  padding=0)

    def forward(self,x):
        x = self.conv2d(x)
        return self.max_pool(x)

def test_CNN():
    to_tensor = torchvision.transforms.ToTensor()
    data_test = torchvision.datasets.CIFAR10(root="./dataset/cifar10",
                                             train=False,
                                             download=True,
                                             transform=to_tensor)
    loader_test = DataLoader(dataset=data_test,
                             batch_size=64,
                             drop_last=False)
    writer = SummaryWriter(log_dir="./board/pool")
    step = 0
    cnn_model = CNN()
    for data in loader_test:
        imgs,labels = data
        if step == 0:
            print(imgs.shape)
        imgs = cnn_model(imgs)
        if step == 0:
            print(imgs.shape)
        writer.add_images("max_pool",imgs,step)
        step+=1

    j20 = Image.open("./image/j20.png").convert(mode="RGB")
    j20_tensor = to_tensor(j20)
    print(f"j20_tensor shape->{j20_tensor.shape}")
    writer.add_image("j20_max_pool", j20_tensor, 0)
    j20_cnn = cnn_model(j20_tensor)
    print(f"j20_cnn shape->{j20_cnn.shape}")
    writer.add_image("j20_max_pool", j20_cnn, 1)
    writer.close()

if __name__ == "__main__":
    input = torch.tensor([
                 [2,4,5],
                 [3,6,1],
                 [10,9,3]
              ],dtype=torch.float)
    print(input.shape)
    input = torch.reshape(input,(-1,1,3,3))
    output = test_max_pool(input)
    print(output)
    print("*****************************")
    test_CNN()
