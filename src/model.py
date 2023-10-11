# -* UTF-8 *-
'''
==============================================================
@Project -> File : pytorch -> model.py
@Author : yge
@Date : 2023/3/31 20:58
@Desc :

==============================================================
'''
import torch
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU
import torch.nn as nn

class CNN_CIFAR10(Module):
    '''
      This CNN model is used for CIFAR10 dataset
      input: 3x32x32

    '''
    def __init__(self) -> None:
        super().__init__()
        self.model = Sequential(
            Conv2d(3,32,kernel_size=5,padding=2,stride=1),
            MaxPool2d(kernel_size=2),
            Conv2d(32,32,kernel_size=5,stride=1,padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(64*4*4,64),
            Linear(64,10),
            ReLU()
        )
    def forward(self,x):
        x = self.model(x)
        return x

vgg_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

'''
https://github.com/kuangliu/pytorch-cifar
'''
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(vgg_cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

if __name__ == "__main__":
    cnn = CNN_CIFAR10()
    input= torch.ones(64,3,32,32)
    output = cnn(input)
    print(output.shape)