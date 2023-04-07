from model import *
import torch
from torch import nn
from torch.utils.data import DataLoader, dataset
from torchvision import transforms, utils, datasets
from torch.utils.tensorboard import SummaryWriter

data_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

train_path = "C:\\Users\\asus\\Desktop\\ChessDataset\\train"
test_path = "C:\\Users\\asus\\Desktop\\ChessDataset\\test"
train_dataset = datasets.ImageFolder(root=train_path, transform=data_transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=data_transform)

train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)
# img, label = img_dataset[66]
# print(label)
# print(len(img_dataset.imgs))


# 模型对象
myModel = MyModel()
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.SGD(myModel.parameters(), lr=0.005)

# tensorboard
writer = SummaryWriter('./log')

total_train_step = 0
total_test_step = 0
epoch = 20

for i in range(epoch):
    print("第{}次训练".format((i + 1)))

    # 训练部分
    myModel.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = myModel(imgs)
        # 损失函数
        loss = loss_fn(outputs, targets)
        # 优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数:{},Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试部分
    myModel.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = myModel(imgs)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            test_accuracy += accuracy
    print("测试集Loss:{}".format(test_loss))
    print("测试集正确率:{}".format(test_accuracy/len(test_dataset)))
    writer.add_scalar("test_loss", test_loss, total_test_step)
    total_test_step += 1

writer.close()
