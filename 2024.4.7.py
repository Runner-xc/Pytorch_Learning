# code :rule
# 开发日期：2024/4/7
import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter


class Happy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input + 1
        return output

happy = Happy()
x = torch.tensor(1.0)
output = happy(x)
print(output)


# maxpooling
class Fan(nn.Module):
    def __init__(self):
        super(Fan,self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=10,stride=1,ceil_mode=True,padding=0)

    def forward(self,x):
        output = self.maxpool1(x)
        output = self.maxpool1(output)
        return output

img_path = 'images/train2017/1-1F91H11612.jpg'
img = Image.open(img_path)
img_tenor = torchvision.transforms.ToTensor() # 将格式转为tensor
img = img_tenor(img)

writer = SummaryWriter('maxpooling_logs')
# 实例化 Fan()
fan = Fan()
output = fan(img)
writer.add_image("img",img,0)
writer.add_image("img_maxpooling",output,1)
writer.close()

print(img.shape)
# torch.Size([3, 881, 1280])
print(output.shape)
# torch.Size([3, 863, 1262])    池化不改变通道数
