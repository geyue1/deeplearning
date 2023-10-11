# -* UTF-8 *-
'''
==============================================================
@Project -> File : pytorch -> evaluate.py
@Author : yge
@Date : 2023/4/1 11:24
@Desc :

==============================================================
'''
import torch
import torchvision
from PIL import Image

from src.vgg import VGG

result_dict = {
    0:"airplane",
    1:"automobile",
    2:"bird",
    3:"cat",
    4:"deer",
    5:"dog",
    6:"frog",
    7:"horse",
    8:"ship",
    9:"truck"
}
def eval_cifar10(path):
     img = Image.open(path).convert(mode="RGB")
     transforms_tool = torchvision.transforms.Compose([
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Resize(size=(32,32),antialias=True)
     ])
     img_tensor = transforms_tool(img)
     img_tensor = torch.reshape(img_tensor,(1,3,32,32))
     print(type(img_tensor))
     print(img_tensor.shape)
     with torch.no_grad():
         model = torch.load("../saved_models/cifar10_cnn_05.pth",
                            map_location=torch.device("cpu"))
         print(model.parameter())
         output = model(img_tensor)
         print(output,output.argmax(1).item())
         output = torch.nn.functional.softmax(output, 1)
         print(f"output:{output}")
         return output.argmax(1).item()


def eval_vgg(path):
    img = Image.open(path).convert(mode="RGB")
    transforms_tool = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=(32, 32), antialias=True)
    ])
    img_tensor = transforms_tool(img)
    img_tensor = torch.reshape(img_tensor, (1, 3, 32, 32))
    print(type(img_tensor))
    print(img_tensor.shape)
    with torch.no_grad():
        model = torch.load("../saved_models/vgg19_01.pth",
                           map_location=torch.device("cpu"))
        for p in model.parameters():
            print(f"p->{p}")
        output = model(img_tensor)
        print(output, output.argmax(1).item())
        output = torch.nn.functional.softmax(output, 1)
        print(f"output:{output}")
        return output.argmax(1).item()

def eval_vgg2(path):
    img = Image.open(path).convert(mode="RGB")
    transforms_tool = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=(32, 32), antialias=True)
    ])
    img_tensor = transforms_tool(img)
    img_tensor = torch.reshape(img_tensor, (1, 3, 32, 32))
    print(type(img_tensor))
    print(img_tensor.shape)
    with torch.no_grad():
        # model = torch.load("checkpoint/ckpt.pth",
        #                    map_location=torch.device("cpu"))
        model = VGG('VGG19')
        model.load_state_dict(torch.load("checkpoint/ckpt01.pth",
                            map_location=torch.device("cpu"))["net"])

        output = model(img_tensor)
        print(output, output.argmax(1).item())
        output = torch.nn.functional.softmax(output, 1)
        print(f"output:{output}")
        return output.argmax(1).item()

if __name__ == "__main__":

    path = input("please input the path of a picture:")
    # a = eval_cifar10(path)
    a = eval_vgg2(path)
    print(f"the path is :{path}")
    if a in result_dict:
        print(f"this picture is {result_dict[a]}")
    else:
        print("sorry, this model can't identify this picture")