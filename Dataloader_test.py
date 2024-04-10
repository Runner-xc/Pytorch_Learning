# code :rule
# 开发日期：2024/4/7
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset",train=False, transform=torchvision.transforms.ToTensor(), download=True)
print(type(test_data))

data_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=False,num_workers=0,drop_last=True)

# 测试第一张图片
img, target = test_data[0]
print(img.shape)
print(target)

# 使用SummaryWriter
writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in data_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{},shuffle=false".format(epoch),imgs,step)
        step += 1

writer.close()
