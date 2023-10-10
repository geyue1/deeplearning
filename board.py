# -* UTF-8 *-
'''
==============================================================
@Project -> File : pytorch -> board.py
@Author : yge
@Date : 2023/3/30 10:46
@Desc :

==============================================================
'''
import torch.utils.data
import torchvision.datasets
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from src.utils.util import del_files

del_files("./board")
wirter = SummaryWriter(log_dir="./board")

img = Image.open("./image/j20.png").convert(mode="RGB")
print(img)
print(img)
img_array = np.array(img)
print(img_array.shape)
for i in range(3):
    wirter.add_image("J20",img_array,i,dataformats="HWC")

for i in range(50):
    wirter.add_scalar("y=2*x",2*i,i)

# dataset = torchvision.datasets.CIFAR10(root="./dataset",train=True)

wirter.close()