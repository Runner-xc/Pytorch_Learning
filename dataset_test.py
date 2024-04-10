# code :rule
# 开发日期：2024/4/6
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()
])

train_data = torchvision.datasets.CIFAR10(root='./dataset',train=True,transform=dataset_transform,download=True)
test_data = torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=dataset_transform,download=True)

# print(test_data[0])
# img, target = test_data[0]
# print(img)
# print(target)
# img.show()

writer = SummaryWriter('p10')
for i in range(10):
    img, target = test_data[i]
    writer.add_image("test",img,i)


writer.close()