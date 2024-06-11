import torch.nn as nn
from models.modules.common import BottleneckCSP, Conv, Concat, C3, C2f, RepNCSPELAN4
from utils.general import make_divisible


class FPNv9(nn.Module):

    def __init__(self, C3_size=512, C4_size=512, C5_size=512, version='C'):
        super(FPNv9, self).__init__()
        self.C3_size = C3_size
        self.C4_size = C4_size
        self.C5_size = C5_size

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
            'ch1': 128,
            'ch2': 256,
            'ch3': 512
        }

        self.re_channels_out()

        self.concat = Concat()

        self.upsample_1 = nn.Upsample(size=None, scale_factor=2, mode='nearest')   # in:512, out:512

        self.rep_1 = RepNCSPELAN4(self.C5_size + self.C4_size, self.channels_out['ch3'],
                                  self.channels_out['ch3'], self.channels_out['ch2'], c5=1)   # in:512+512, out:512

        self.upsample_2 = nn.Upsample(size=None, scale_factor=2, mode='nearest')   # in:512, out:512

        self.rep_2 = RepNCSPELAN4(self.channels_out['ch3'] + self.C4_size, self.channels_out['ch2'],
                                  self.channels_out['ch2'], self.channels_out['ch1'], c5=1)  # in:512+512, out:256


        self.out_shape = {'F3_size': self.C5_size,
                          'F4_size': self.channels_out['ch3'],
                          'F5_size': self.channels_out['ch2']}   # 512, 512, 256

        print("FPN input channel size: C3 {}, C4 {}, C5 {}".format(self.C3_size,
                                                                   self.C4_size,
                                                                   self.C5_size))   # 512, 512, 512

        print("FPN output channel size: F3 {}, F4 {}, F5 {}".format(self.C5_size,
                                                                    self.channels_out['ch3'],
                                                                    self.channels_out['ch2']))  # 512, 512, 256


    def forward(self, inputs):

        C3, C4, C5 = inputs

        up_1 = self.upsample_1(C5)   # 10

        concat_1 = self.concat([up_1, C4])  # 11

        F4 = self.rep_1(concat_1)  # 12

        up_2 = self.upsample_2(F4)  # 13

        concat_2 = self.concat([up_2, C3])  # 14

        F5 = self.rep_2(concat_2)  # 15

        F3 = C5

        return F3, F4, F5

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels_out.items():
            self.channels_out[k] = self.get_width(v)
