# -* UTF-8 *-
'''
==============================================================
@Project -> File : pytorch -> conv2d_.py
@Author : yge
@Date : 2023/3/30 21:57
@Desc :

==============================================================
'''
import torch
import torch.nn.functional as f
import torchvision.datasets
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class conv2d_Model(torch.nn.Module):

    def __init__(self,kernel) -> None:
        super().__init__()
        self.kernel = kernel

    def forward(self,x):
        x = f.conv2d(x,kernel,stride=1)
        return x

class conv2d_Model_(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3,3,3)
    def forward(self,x):
        return self.conv2d(x)

if __name__ == "__main__":
    input = torch.tensor([
                  [1,2,8,5,3],
                  [0.5,2.5,5,6,1],
                  [3,1.5,1,3,2.5],
                  [0.4,0,4,2.5,2.5],
                  [0.6,1,1,1.5,5]
                        ],dtype=torch.float)
    print(input.shape)
    input = torch.reshape(input,(1,5,5))
    print(input.shape)
    print(input)
    kernel = torch.tensor([
        [1.0,0,0],
        [0,1,0],
        [0,0,1]
    ],dtype=torch.float)
    print(kernel.shape)
    kernel = torch.reshape(kernel,(1,1,3,3))
    print(kernel)
    m = conv2d_Model(kernel)
    output = m(input)
    print(output)
    print("*************************************")
    dataset_test = torchvision.datasets.CIFAR10(root="./dataset/cifar10",train=False,download=False,
                                                transform=torchvision.transforms.ToTensor())
    test_loader = DataLoader(dataset=dataset_test,batch_size=64,shuffle=True)

    writer = SummaryWriter(log_dir="./board/conv2d")
    step = 0
    m = conv2d_Model_()
    for data in test_loader:
        imgs,labels = data
        if step==0:
            print(imgs.shape)
        imgs = m(imgs)
        if step==0:
            print(imgs.shape)
        writer.add_images("conv2d_model",imgs,step)
        step+=1
    j20 = Image.open("./image/j20.png").convert(mode="RGB")
    transform = torchvision.transforms.ToTensor()
    parameters = m.parameters()
    for param in parameters:
        print(param,type(param), param.size())
    j20_tensor = transform(j20)
    j20_output = m(j20_tensor)
    writer.add_image("j20_conv2d",j20_tensor,0)
    writer.add_image("j20_conv2d",j20_output,1)
    writer.close()