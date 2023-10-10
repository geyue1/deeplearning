# -* UTF-8 *-
'''
==============================================================
@Project -> File : pytorch -> transforms_.py
@Author : yge
@Date : 2023/3/30 16:40
@Desc :

==============================================================
'''
import torchvision.transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from src.utils.util import del_files

del_files("./board")
img_j20 = Image.open("./image/j20.png").convert(mode="RGB")
transforms_tool = torchvision.transforms.Compose([
                  torchvision.transforms.ToTensor(),
                  torchvision.transforms.Resize(32)
                  ])
board_id = "transforms_"
writer = SummaryWriter(log_dir="./board")
img_j20_tensor = transforms_tool(img_j20)
print(img_j20_tensor)
writer.add_image(board_id,img_j20_tensor,0)

dataset_train = torchvision.datasets.CIFAR10(root="./dataset/cifar10",train=True,transform=transforms_tool,download=True)
for i in range(19):
    img,label = dataset_train[i]
    print(img)
    print(label)
    writer.add_image(board_id,img,i+1)
writer.close()