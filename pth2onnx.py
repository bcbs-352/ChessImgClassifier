import torch
from torch import nn
from torch.autograd import Variable


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

model = torch.load("myModel_tmp.pth")

model.eval()
generate_input = Variable(torch.randn(1, 3, 32, 32, device='cuda'))
torch.onnx.export(model, generate_input, "myModel_new.onnx", export_params=True, verbose=True, input_names=["input"],
                  output_names=["output"])