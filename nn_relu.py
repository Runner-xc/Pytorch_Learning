# code :rule
# 开发日期：2024/4/8
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data_train = torchvision.datasets.CIFAR10('dataset', train=True, transform=torchvision.transforms.ToTensor(),download=True)

data_loader = DataLoader(dataset=data_train, batch_size=64)

# 创建自己的类
class Run(nn.Module):
    def __init__(self):
        super(Run,self).__init__()
        self.relu1 = ReLU()                         # 定义relu激活函数层
        self.sigmoid1 = Sigmoid()                   # 定义sigmoid激活函数层

    def forward(self,input):                        # 前向传播
        output = self.relu1(input)
        output = self.sigmoid1(output)
        return output

writer = SummaryWriter('./logs_relu')               # 使用tensorboard可视化运行结果
step = 0
for data in data_loader:
    imgs, targets = data

    writer.add_images("img", imgs, step)
    run = Run()
    output = run(imgs)
    writer.add_images("img_activate", output, step)
    step += 1

writer.close()                                       # 关闭文件