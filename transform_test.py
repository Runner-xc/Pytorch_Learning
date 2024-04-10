# code :rule
# 开发日期：2024/4/6
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

# python 的用法->tensor数据类型

img_path = 'images/train2017/1-1F91H11612.jpg'
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer = SummaryWriter('logs')
writer.add_image('Tensor_img', tensor_img)

# normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(tensor_img)
print(tensor_img[0][0][0])
writer.add_image("normalize", img_norm,2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
# img PIL ->
img_resize = trans_resize(img)
img_resize = tensor_trains(img_resize)
writer.add_image("resize",img_resize,0)
print(img_resize)

# Compose -resize- 2
img_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([img_resize_2,tensor_trains])
img_resize_2 = trans_compose(img)
writer.add_image("RESIZE",img_resize_2,1)

writer.close()