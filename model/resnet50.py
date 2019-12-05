import torch
import torch.nn as nn

from torchvision.models import resnet50

original_resnet = resnet50(pretrained=True)
class ResNetNormalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = original_resnet

    def forward(self, x):
        # x: B x 3 x H x W
        # x: B x 64 x H//2 x W//2
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # l1: B x 256 x H//4 x W//4
        l1 = self.layer1(x)
        # l2: B x 3 x H//8 x W//8
        l2 = self.layer2(l1)
        # l3: B x 3 x H//16 x W//16
        l3 = self.layer3(l2)
        # l4: B x 3 x H//32 x W//32
        l4 = self.layer4(l3)


        return l1, l2, l3, l4




