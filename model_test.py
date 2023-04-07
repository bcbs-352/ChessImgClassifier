import torch
import torchvision
from PIL import Image
from torch import nn

img_path = "./Sample_Images/2.jpg"
img = Image.open(img_path)
# img.show()
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
img = transform(img)
img = torch.reshape(img, (1, 3, 32, 32))
img = img.cuda()
print(img.shape)
chess_list = ['黑士', '红仕', '黑象', '红相', '黑炮', '红炮', '黑将', '红帅', '黑马', '红马', '黑卒', '红兵', '黑車',
              '红車']

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 14)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# class AlexNet(nn.Module):
#     def __init__(self, num_classes=14, init_weights=False):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
#             # 这里的填充应该是左上各补一行零，右下补两行零，但是最后会有小数出现，会自动舍弃一行零。input[3, 224, 224]  output[48, 55, 55]
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
#             nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
#             nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(128 * 6 * 6, 2048),  # output[2048]
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(2048, 2048),  # output[2048]
#             nn.ReLU(inplace=True),
#             nn.Linear(2048, num_classes),  # output[1000]
#         )
#         if init_weights:
#             self._initialize_weights()
#
#     def forward(self, x):
#         x = self.features(x)
#         # pytorch的tensor通道排列顺序为[batch, channel, height, weight].
#         # start_dim=1也就是从第一维度channel开始展平成一个一维向量
#         x = torch.flatten(x, start_dim=1)
#         x = self.classifier(x)
#         return x
#
#     # 权重初始化，现阶段pytorch版本自动进行权重初始化
#     def _initialize_weights(self):
#         # 遍历定义的各个模块
#         for m in self.modules():
#             # 如果模块为nn.Conv2d，就用kaiming_normal_()方法对权重进行初始化，如果偏置为空，就用0进行初始化
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             # 如果模块为全连接层，就用normal_()正态分布来给权重赋值，均值为0，方差为0.01
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)


model = torch.load("myModel_tmp.pth")
model.cuda()
print(model)

model.eval()
with torch.no_grad():
    output = model(img)

print(output)
print(chess_list[output.argmax(1).item()])
