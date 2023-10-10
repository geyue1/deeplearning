# -* UTF-8 *-
'''
==============================================================
@Project -> File : pytorch -> download.py
@Author : yge
@Date : 2023/3/29 14:56
@Desc :

==============================================================
'''

import torchvision

train_set = torchvision.datasets.CIFAR10(root="./dataset/cifar10",train=True,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset/cifar10",train=False,download=True)