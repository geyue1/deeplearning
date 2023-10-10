# -* UTF-8 *-
'''
==============================================================
@Project -> File : pytorch -> activation_function_.py
@Author : yge
@Date : 2023/3/31 16:54
@Desc :

==============================================================
'''
import torch
from torch import nn, Tensor


def test_ReLu(x):
    relu_fun = nn.ReLU()
    if not type(x) is Tensor:
        x = torch.Tensor(x)
        print(x)
        print(f"x shape->{x.shape}")
    return relu_fun(x)


if __name__ == "__main__":
    print(test_ReLu([2]))
    input = torch.randn(2).unsqueeze(0)
    print(input,test_ReLu(input))