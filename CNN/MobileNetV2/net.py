from hmac import new
import torch
from torch import nn as nn

def _makedivisible(ch,divisor=8,min_ch=None):
    if min_ch==None:
        min_ch=divisor
    
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)

    if new_ch < 0.9*ch:
        new_ch += divisor
    
    return new_ch