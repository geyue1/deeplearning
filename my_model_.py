# -* UTF-8 *-
'''
==============================================================
@Project -> File : pytorch -> my_model_.py
@Author : yge
@Date : 2023/3/30 18:27
@Desc :

==============================================================
'''
import torch
import torch.nn as nn

class myModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        print("------myModel init--------")
        self.w = 5
        self.b = 3

    def forward(self, x):
        print("------myModel forward--------")
        # x = F.relu(self.conv1(x))
        # return F.relu(self.conv2(x))
        x = self.w*x+self.b
        return x

if __name__ == "__main__":

    m = myModel()
    x = torch.tensor(2)
    print(m(x))
    # 保存模型
    # torch.save(m,"./saved_models/my_model.pth")
    # 加载模型
    print(torch.load("./saved_models/my_model.pth")(3))
