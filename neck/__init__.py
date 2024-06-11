# -*- coding: utf-8 -*-
from .FPN import PyramidFeatures as FPN
from .PAN import PAN
from .FPN_v8 import FPNv8
from .PAN_v8 import PANv8
from .FPN_v9 import FPNv9
from .PAN_v9 import PANv9

__all__ = ['build_neck']
support_neck = ['FPN', 'PAN', 'FPNv8', 'PANv8', 'FPNv9', 'PANv9']


def build_neck(neck_name, **kwargs):
    assert neck_name in support_neck, f'all support neck is {support_neck}'
    neck = eval(neck_name)(**kwargs)
    return neck