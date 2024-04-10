# code :rule
# 开发日期：2024/4/8
import torch
import torchvision.transforms
from PIL import Image
import os

from torch import nn
from torch.nn import Conv2d, ReLU, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Run(nn.Module):
    def __init__(self):
        super(Run,self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32,kernel_size=3, stride=1, padding=0)
        self.relu1 = ReLU(inplace=False)
        self.conv2 = Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1,padding=0)
        self.relu2 = ReLU(inplace=False)
        self.linear = Linear(18432,10)

    #   前向传播
    def forward(self,input):
        output = self.conv1(input)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        return output

writer = SummaryWriter('my_code_logs')
# 实例化Run()
run = Run()

img_dir = './images/train2017'
img_list = os.listdir(img_dir)

# # 这里需要设置 antialias=True
# img_resize = torchvision.transforms.Resize(64,antialias=True)

# PIL -> tensor
img_trans = torchvision.transforms.ToTensor()

# 取20张图片
for i in range(20):
    img = Image.open(os.path.join(img_dir, img_list[i]))   # Image.open需要一个完整的文件路径。
    img_tensor = img_trans(img)
    img_tensor = img_tensor.float()
    # # resize=64
    # # img_tensor = img_resize(img_tensor)
    # out = torch.flatten(img_tensor)

    writer.add_image('img', img_tensor, i)   # .add_image需要指定'CHW'的顺序，具体可以观看官方文档
    output = run(img_tensor)
    writer.add_image('img_forward', output.float(), i)

writer.close()