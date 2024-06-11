import torch.nn as nn
from models.modules.common import Conv, C3, SPP, BottleneckCSP, C3TR, C2f, SPPF, RepNCSPELAN4, ADown, SPPELAN
from utils.general import make_divisible


class YOLOv9(nn.Module):
    def __init__(self, input_channel=3, version='C'):
        super(YOLOv9, self).__init__()
        self.version = version

        gains = {'n': {'gd':0.33, 'gw':0.25},
                 's': {'gd': 0.33, 'gw': 0.5},
                 'm': {'gd': 0.67, 'gw': 0.75},
                 'c': {'gd': 1, 'gw': 1},
                 'x': {'gd': 1, 'gw': 1.25}}

        if self.version.lower() in gains:
            self.gd = gains[self.version.lower()]['gd']  # depth gain
            self.gw = gains[self.version.lower()]['gw']  # width gain
        else:
            self.gd = 0.33
            self.gw = 0.5

        self.channels_out = {
            'ch1':64,
            'ch2':128,
            'ch3':256,
            'ch4':512
            }

        self.re_channels_out()

        self.input_channels = input_channel  # ch=3 rgb

        self.conv_1 = Conv(self.input_channels, self.channels_out['ch1'], 3, 2)  # in:3, out:64

        self.conv_2 = Conv(self.channels_out['ch1'], self.channels_out['ch2'], 3, 2)  # in:64, out:128

        self.rep_1 = RepNCSPELAN4(self.channels_out['ch2'], self.channels_out['ch3'],
                                  self.channels_out['ch2'], self.channels_out['ch1'], c5=1)  # in:128, out:256

        self.adown_1 = ADown(self.channels_out['ch3'], self.channels_out['ch3'])   # in:256, out:256

        self.rep_2 = RepNCSPELAN4(self.channels_out['ch3'], self.channels_out['ch4'],
                                  self.channels_out['ch3'], self.channels_out['ch2'], c5=1)  # in:256, out:512

        self.adown_2 = ADown(self.channels_out['ch4'], self.channels_out['ch4'])  # in:512, out:512

        self.rep_3 = RepNCSPELAN4(self.channels_out['ch4'], self.channels_out['ch4'],
                                  self.channels_out['ch4'], self.channels_out['ch3'], c5=1)  # in:512, out:512

        self.adown_3 = ADown(self.channels_out['ch4'], self.channels_out['ch4'])  # in:512, out:512

        self.rep_4 = RepNCSPELAN4(self.channels_out['ch4'], self.channels_out['ch4'],
                                  self.channels_out['ch4'], self.channels_out['ch3'], c5=1)   # in:512, out:512

        self.sppfelan = SPPELAN(self.channels_out['ch4'], self.channels_out['ch4'], self.channels_out['ch3'])  # in:512, out:512


        self.out_shape = {'C3_size': self.channels_out['ch4'],
                          'C4_size': self.channels_out['ch4'],
                          'C5_size': self.channels_out['ch4']}  # 512, 512, 512

        print("backbone output channel: C3 {}, C4 {}, C5 {}".format(self.channels_out['ch4'],
                                                                    self.channels_out['ch4'],
                                                                    self.channels_out['ch4']))  # 512,512,512

    def forward(self, x):

        backbone_0 = self.conv_1(x)  # 0

        backbone_1 = self.conv_2(backbone_0)  # 1

        backbone_2 = self.rep_1(backbone_1)   # 2

        backbone_3 = self.adown_1(backbone_2)   # 3

        c3 = self.rep_2(backbone_3)  # backbone_4  4

        backbone_5 = self.adown_2(c3)  # 5

        c4 = self.rep_3(backbone_5)  # backbone_6  6

        backbone_7 = self.adown_3(c4)  # 7

        backbone_8 = self.rep_4(backbone_7)  # 8

        c5 = self.sppfelan(backbone_8)  # backbone9  9

        return c3, c4, c5


    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels_out.items():
            self.channels_out[k] = self.get_width(v)