# code :rule
# 开发日期：2024/4/7
import torchvision
from torch import nn, relu
from torch.nn import Conv2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor)

dataloader = DataLoader(dataset=dataset, batch_size=64)

class Rule(nn.Module):
    def __init__(self):
        super(Rule, self).__init__()
        self.conv2d = Conv2d(in_channels=3,out_channels=6,kernel_size=3,padding=0)

    def forward(self,x):
        self.conv2d(x)
        return x

rule = Rule()
print(rule)