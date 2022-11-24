import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from base_oc_block import BaseOC_Module

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            if groups != 1:
                self.bn_prelu = BN(nOut)
            else:
                self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class BN(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)

    def forward(self, input):
        output = self.bn(input)
        return output


class MFModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                             padding=(1, 0), groups=nIn // 2, bn_acti=True)
        # self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
        #                      padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                              padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                              padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 3, 1, padding=1, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)

        br1 = self.dconv3x1(output)
        # br1 = self.dconv1x3(br1)
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)

        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = Conv(in_chan, out_chan, kSize=1, stride=1, padding=0, bn_acti=True)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class MFNet(nn.Module):
    def __init__(self, classes=19, block_1=2, block_2=5):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(4)  # down-sample the image 3 times

        self.bn_prelu_1 = BNPReLU(32 + 3)

        # MF Block 1
        self.downsample_1 = DownSamplingBlock(32 + 3, 64)
        self.MF_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.MF_Block_1.add_module("MF_Module_1_" + str(i), MFModule(64, d=2))
        self.bn_prelu_2 = BNPReLU(128 + 3)

        # MF Block 2
        dilation_block_2 = [2, 4, 4, 16, 16]
        self.downsample_2 = DownSamplingBlock(128 + 3, 128)
        self.downsample_3 = DownSamplingBlock(128, 128)
        self.MF_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.MF_Block_2.add_module("MF_Module_2_" + str(i),
                                        MFModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(256 + 3)

        self.classifier = nn.Sequential(Conv(259, 64, 1, 1, padding=0),
                                        nn.BatchNorm2d(64),
                                        nn.PReLU(),)
        self.ocm = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            BaseOC_Module(128, 128, 64, 64, 0.05)
        )
        self.convocm = nn.Sequential(Conv(128, 64, kSize=1, stride=1, padding=0),
                                     nn.BatchNorm2d(64),
                                     nn.PReLU())

        self.convfuse = nn.Sequential(Conv(128, 64, kSize=3, stride=1, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.PReLU())
        self.SP = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.PReLU())
        self.ffm = FeatureFusionModule(96, 64)
        self.classifier2 = nn.Sequential(Conv(64, classes, 1, 1, padding=0),
                                         nn.BatchNorm2d(classes),
                                         nn.PReLU(),
                                         )

    def forward(self, input):

        output0 = self.init_conv(input)

        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)

        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))

        # MF Block 1
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.MF_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2], 1))

        # MF Block 2
        output2_0 = self.downsample_2(output1_cat)
        output2_0 = self.downsample_3(output2_0)
        output2 = self.MF_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3], 1))

        out0 = self.classifier(output2_cat)

        # out = torch.cat([out, output1], 1)
        outocm = self.ocm(output2_0)
        outocm = self.convocm(outocm)
        out1 = self.convfuse(torch.cat([outocm, out0], 1))
        out1 = F.interpolate(out1, scale_factor=4, mode='bilinear', align_corners=False)
        out2 = self.SP(output1)
        out = self.ffm(out2, out1)
        out = self.classifier2(out)
        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)

        return out

if __name__ == '__main__':
    t_start = time.time()
    iteration = 5
    for _ in range(iteration):
        img = torch.randn(2, 3, 256, 512)
        model = MFNet(11)
        outputs = model(img)
        # print(outputs.size())
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))

    def netParams(model):
        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p
        return total_paramters

    print(netParams(model))

