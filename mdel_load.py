# code :rule
# 开发日期：2024/4/10
import torch
import torchvision

# 方式1 -> 保存方式1 加载模型
model = torch.load("vgg16_method1.pth")
print(model)

print("\n第二种方式：\n")
# 方式2 -> 加载模型 （字典形式的参数）
# model2 = torch.load("vgg16_method2.pth")
# print(model2)

    # 若想还原成模型
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)