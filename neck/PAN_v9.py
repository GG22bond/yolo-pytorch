import torch.nn as nn
from models.modules.common import BottleneckCSP, Conv, Concat, C3, C2f, ADown, RepNCSPELAN4
from utils.general import make_divisible


class PANv9(nn.Module):

    def __init__(self, F3_size=512, F4_size=512, F5_size=256, version='C'):
        super(PANv9, self).__init__()
        self.F3_size = F3_size
        self.F4_size = F4_size
        self.F5_size = F5_size

        self.version = version

        gains = {'n': {'gd':0.33, 'gw':0.25},
                 's': {'gd': 0.33, 'gw': 0.5},
                 'm': {'gd': 0.67, 'gw': 0.75},
                 'c': {'gd': 1, 'gw': 1},
                 'x': {'gd': 1, 'gw': 1.25}}

        if self.version.lower() in gains:
            self.gd = gains[self.version.lower()]['gd']  # depth gain 😊
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

        self.adown_1 = ADown(self.F5_size, self.channels_out['ch2'])  # in:256, out:256

        self.rep_1 = RepNCSPELAN4(self.channels_out['ch2'] + self.F4_size, self.channels_out['ch3'],
                                  self.channels_out['ch3'], self.channels_out['ch2'], c5=1)   # in:256+512, out:512

        self.adown_2 = ADown(self.channels_out['ch3'], self.channels_out['ch3'])   # in:512, out:512

        self.rep_2 = RepNCSPELAN4(self.channels_out['ch3'] + self.F3_size, self.channels_out['ch3'],
                                  self.channels_out['ch3'], self.channels_out['ch2'], c5=1)     # in:512+512, out:512


        self.out_shape = {'P3_size': self.F5_size,
                          'P4_size': self.channels_out['ch3'],
                          'P5_size': self.channels_out['ch3']}  # 256, 512, 512

        print("PAN input channel size: F3 {}, F4 {}, F5 {}".format(self.F3_size,
                                                                   self.F4_size,
                                                                   self.F5_size))  # 512, 512, 256

        print("FPN output channel size: P3 {}, P4 {}, P5 {}".format(self.F5_size,
                                                                    self.channels_out['ch3'],
                                                                    self.channels_out['ch3']))  # 256, 512, 512


    def forward(self, inputs):

        F3, F4, F5 = inputs

        con_1 = self.con_1(F5)  # 16

        concat_1 = self.concat([con_1, F4])  # 17

        P4 = self.c2f_1(concat_1)  # 18

        con_2 = self.con_2(P4)  # 19

        concat_2 = self.concat([con_2, F3])  # 20

        P5 = self.c2f_2(concat_2)   # 21

        P3 = F5

        return P3, P4, P5

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels_out.items():
            self.channels_out[k] = self.get_width(v)
