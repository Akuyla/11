import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        reduced_channels = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # SE 模块：对通道维进行重标定，突出更有判别力的响应。
        scale = self.avg_pool(x)
        scale = self.fc(scale)
        return x * scale

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class SSHSE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSHSE, self).__init__()
        self.ssh = SSH(in_channel, out_channel)
        self.se = SEModule(out_channel)

    def forward(self, input):
        # 先提取 SSH 多分支上下文，再使用 SE 做通道注意力增强。
        out = self.ssh(input)
        out = self.se(out)
        return out


def build_ssh(ssh_type, in_channel, out_channel):
    if ssh_type == 'ssh':
        return SSH(in_channel, out_channel)
    if ssh_type == 'ssh_se':
        return SSHSE(in_channel, out_channel)
    raise ValueError('Unsupported ssh_type: {}'.format(ssh_type))

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.use_p2 = len(in_channels_list) == 4
        self.output_convs = nn.ModuleList([
            conv_bn1X1(in_channels, out_channels, stride = 1, leaky = leaky)
            for in_channels in in_channels_list
        ])
        self.merge_convs = nn.ModuleList([
            conv_bn(out_channels, out_channels, leaky = leaky)
            for _ in range(len(in_channels_list) - 1)
        ])
        if self.use_p2:
            self.p2_refine = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        # 同时兼容原始的 3 层金字塔与加入 P2 后的 4 层金字塔。
        outputs = [conv(feature) for conv, feature in zip(self.output_convs, input)]

        for idx in range(len(outputs) - 1, 0, -1):
            upsample = F.interpolate(outputs[idx], size=[outputs[idx - 1].size(2), outputs[idx - 1].size(3)], mode="nearest")
            outputs[idx - 1] = outputs[idx - 1] + upsample
            outputs[idx - 1] = self.merge_convs[idx - 1](outputs[idx - 1])

        # 在 P2 上额外增加一层卷积，进一步增强最小人脸对应的特征表达。
        if self.use_p2:
            outputs[0] = self.p2_refine(outputs[0])

        return outputs



class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

