# -* UTF-8 *-
'''
==============================================================
@Project -> File : pytorch -> dataloader_.py
@Author : yge
@Date : 2023/3/30 17:29
@Desc :

==============================================================
'''
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset/cifar10",
                                       train=False,
                                       download=False,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset=dataset,
                        batch_size=32,
                        shuffle=True,
                        drop_last=False)

img,label = dataset[0]
print(img.shape)
print(label)

writer = SummaryWriter(log_dir="./board/dataloader")
step = 0
for data in dataloader:
    imgs,labels = data
    if step==0:
        print(imgs.shape)
        print(labels)
    writer.add_images("dataloader",imgs,step)
    step+=1

writer.close()