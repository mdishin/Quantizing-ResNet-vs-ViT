import torch
import torch.nn as nn


class BasicBlock(nn.Module): #Residual block
    def __init__(self, in_ch, out_ch, stride): #Setup - Runs once when a block is created
        super().__init__() #Run nn.module's setup first (makes the parts list)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False) #3x3 Conv: Changes channels and shrinks map if stride = 2
        self.bn1 = nn.BatchNorm2d(out_ch) #Norm each of the out_ch maps
        self.relu = nn.ReLU() #ReLU
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch: #Coral block: shapes change
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), #1x1 conv: match channels and size
                nn.BatchNorm2d(out_ch)
            )
        else: #Normal block: shapes already match
            self.skip = nn.Identity() #Copy x as is

    def forward(self, x): #Wiring: runs every time data passes through
        out = self.conv1(x) #Main path starts from x
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out) #out is now F(x)
        out = out + self.skip(x) #Add the skip: original x (or its 1x1 conv version)
        out = self.relu(out) #Last ReLU after the add
        return out


class ResNet20(nn.Module):
    def __init__(self): #Setup - Runs once when the model is created
        super().__init__() #Run nn.module's setup first (makes the parts list)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16) #Norm each of the 16 maps
        self.relu = nn.ReLU() #ReLU
        self.stage1 = nn.Sequential(BasicBlock(16, 16, 1), BasicBlock(16, 16, 1), BasicBlock(16, 16, 1))
        self.stage2 = nn.Sequential(BasicBlock(16, 32, 2), BasicBlock(32, 32, 1), BasicBlock(32, 32, 1)) #16 to 32 ch, 32 to 16 size
        self.stage3 = nn.Sequential(BasicBlock(32, 64, 2), BasicBlock(64, 64, 1), BasicBlock(64, 64, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x): #Wiring: runs every time data passes through
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
