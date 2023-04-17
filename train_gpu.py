import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, dataset
from torchvision import transforms, utils, datasets
from torch.utils.tensorboard import SummaryWriter
import time


class AlexNet(nn.Module):
    def __init__(self, num_classes=14, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            # 这里的填充应该是左上各补一行零，右下补两行零，但是最后会有小数出现，会自动舍弃一行零。input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),  # output[2048]
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),  # output[2048]
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),  # output[1000]
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # pytorch的tensor通道排列顺序为[batch, channel, height, weight].
        # start_dim=1也就是从第一维度channel开始展平成一个一维向量
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    # 权重初始化，现阶段pytorch版本自动进行权重初始化
    def _initialize_weights(self):
        # 遍历定义的各个模块
        for m in self.modules():
            # 如果模块为nn.Conv2d，就用kaiming_normal_()方法对权重进行初始化，如果偏置为空，就用0进行初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 如果模块为全连接层，就用normal_()正态分布来给权重赋值，均值为0，方差为0.01
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        ##Branch的池化层,用卷积1X1来处理，1X1的卷积可以直接将Channel层数
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 24, kernel_size=1)
        )

        ##Branch1X1层
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1)
        )

        ##Branch5x5层, 5X5保持原图像大小需要padding为2，像3x3的卷积padding为1即可
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=5, padding=2)
        )

        ##Branch3x3层
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.Conv2d(24, 24, kernel_size=3, padding=1)
        )

    def forward(self, x):
        ##池化层
        branch_pool = self.branch_pool(x)
        ##branch1X1
        branch1x1 = self.branch1x1(x)
        ##Branch5x5
        branch5x5 = self.branch5x5(x)
        ##Branch3x3
        branch5x5 = self.branch3x3(x)

        ##然后做拼接
        outputs = [branch_pool, branch1x1, branch5x5, branch5x5]
        ##dim=1是为了将channel通道数进行统一， 正常是 B,C,W,H  batchsize,channels,width,height
        ##输出通道数这里计算，branch_pool=24， branch1x1=16， branch5x5=24， branch3x3=24
        ##计算结果就是 24+16+24+24 = 88，在下面Net训练时就知道输入是88通道了
        return torch.cat(outputs, dim=1)


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        ##训练的图像为1X28X28,所以输入通道为1,图像转为10通道后再下采样,再使用用Inception
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Inception(10)
        )

        ##训练的通道由上面的Inception输出，上面计算的输出通道为88，所以这里输入通道就为88
        self.conv2 = nn.Sequential(
            nn.Conv2d(88, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Inception(20)
        )

        ##全链接层,1408是结过上面的网络全部计算出来的，不用自己算，可以输入的时候看Error来修改
        self.fc = nn.Sequential(
            nn.Linear(1408, 10)
        )

        ##定义损失函数
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


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
            nn.Dropout(),
            nn.Linear(64 * 4 * 4, 512),
            #nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 14)
        )

    def forward(self, x):
        x = self.model(x)
        return x


data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    # transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_path = "C:\\Users\\asus\\Desktop\\ChessDataset\\train"
test_path = "C:\\Users\\asus\\Desktop\\ChessDataset\\test"
train_dataset = datasets.ImageFolder(root=train_path, transform=data_transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=data_transform)

print("训练集长度:{}".format(len(train_dataset)))

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128)
# img, label = img_dataset[66]
# print(label)
# print(len(img_dataset.imgs))


# 模型对象
myModel = MyModel()
myModel.cuda()
# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
# 优化器
optimizer = torch.optim.SGD(myModel.parameters(), lr=0.001)

# tensorboard
writer = SummaryWriter('./log')

total_train_step = 0
total_test_step = 0

epoch = 600
learning_rate = 0.01
while epoch > 0:

    optimizer = torch.optim.SGD(myModel.parameters(), lr=learning_rate)
    # 每轮训练
    start_time = time.time()
    for i in range(epoch):
        # print("第{}次训练".format((i + 1)))
        # 训练部分
        myModel.train()
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = myModel(imgs)
            # 损失函数
            loss = loss_fn(outputs, targets)
            # 优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                print("训练次数:{},Loss:{},花费时间:{}".format(total_train_step, loss.item(), time.time() - start_time))
                writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 测试部分
        myModel.eval()
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.cuda()
                targets = targets.cuda()
                outputs = myModel(imgs)
                loss = loss_fn(outputs, targets)
                test_loss += loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                test_accuracy += accuracy
        print("测试集Loss:{}".format(test_loss))
        print("测试集正确率:{}".format(test_accuracy / len(test_dataset)))
        writer.add_scalar("test_loss", test_loss, total_test_step)
        total_test_step += 1

    epoch = input('下一轮训练次数:')
    epoch = int(epoch)
    learning_rate = input('下一轮学习率:')
    learning_rate = float(learning_rate)

# 保存为pth/onnx模型
myModel.eval()
torch.save(myModel, 'model.pth')
generate_input = Variable(torch.randn(1, 3, 32, 32, device='cuda'))
torch.onnx.export(myModel, generate_input, "model_3.onnx", export_params=True, verbose=True, input_names=["input"],
                  output_names=["output"], opset_version=11)

writer.close()
