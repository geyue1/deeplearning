# -* UTF-8 *-
'''
==============================================================
@Project -> File : pytorch -> cifar10_model.py
@Author : yge
@Date : 2023/3/31 21:30
@Desc :

==============================================================
'''
import time

import torch
import torchvision.datasets
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model import CNN_CIFAR10

to_tensor = torchvision.transforms.ToTensor()

def train_model():
    data_train = torchvision.datasets.CIFAR10(
        root="../dataset/cifar10",
        train=True,
        transform=to_tensor,
        download=True
    )
    print(f"train data : {len(data_train)}")
    loader_train = DataLoader(
        dataset=data_train,
        batch_size=64,
        shuffle=True,
        drop_last=False
    )
    # test model
    data_test = torchvision.datasets.CIFAR10(
        root="../dataset/cifar10",
        train=False,
        transform=to_tensor,
        download=True
    )
    print(f"test data : {len(data_test)}")
    loader_test = DataLoader(
        dataset=data_test,
        batch_size=64,
        shuffle=True,
        drop_last=False
    )

    device = torch.device("mps")
    # argmax()函数在GPU和CPU上计算值不同，以CPU为准
    device_cpu = torch.device("cpu")
    cnn = CNN_CIFAR10()
    cnn.to(device)
    r_loss = 0.0
    r_accuray = 0.0
    #r_list = [0.001,0.002,0.003,0.004,0.005,0.01,0.015,0.02]
    r_list = [0.002]
    r_step = 0
    for r in r_list:
        print(f"$$$$$$$$$$$$$$$$$$$$->{r}")
        optimizer = optim.SGD(cnn.parameters(), lr=r, momentum=0.9)
        loss_fn = CrossEntropyLoss()

        writer = SummaryWriter(log_dir="../board/train_model")
        step = 0
        test_step = 0
        for i in range(20):
            sum_loss = 0.0
            for imgs,targets in loader_train:
                imgs = imgs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                imgs_output = cnn(imgs)
                loss = loss_fn(imgs_output, targets)
                sum_loss+=loss
                loss.backward()
                optimizer.step()
                if step%100 ==0:
                   writer.add_scalar("loss_fn_relu",loss,step)
                step+=1
            print(f"-----the {i+1} time training,sum_loss = {sum_loss}")

            with torch.no_grad():
                test_loss = 0.0
                accuracy = 0.0
                for imgs, targets in loader_test:
                    imgs = imgs.to(device)
                    targets = targets.to(device_cpu)
                    outputs = cnn(imgs)
                    outputs = outputs.to(device_cpu)
                    loss = loss_fn(outputs, targets)
                    test_loss += loss.item()
                    outputs = torch.nn.functional.softmax(outputs,1)
                    predicte = outputs.argmax(1)
                    accuracy += (predicte == targets).sum()
                    test_step += 1
                print(f"test_loss:{test_loss}")
                print(f"accuracy:{accuracy},{accuracy / len(data_test)}")
                writer.add_scalar("loss_test_relu", test_loss, test_step)
                writer.add_scalar("accuracy_relu", accuracy / len(data_test), test_step)
                r_loss = test_loss
                r_accuray = accuracy / len(data_test)
        # writer.add_scalar("loss_r", r_loss, r_step)
        # writer.add_scalar("accuracy_r", r_accuray, r_step)
        r_step+=1

    print(f"step:{step}")





    writer.close()
    return cnn

if __name__ == "__main__":
    start = time.time()
    model_cnn = train_model()
    print(model_cnn)
    torch.save(model_cnn,"../saved_models/cifar10_cnn_05.pth")
    end = time.time()
    print(f"total time for train_model:{end-start}")
