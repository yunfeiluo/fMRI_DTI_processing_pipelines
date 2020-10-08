import torch
from torch import nn

class invResBlock(nn.Module):
    def __init__(self, channel):
        super(invResBlock, self).__init__()
        self.deconv_liner = nn.Sequential(
            nn.ConvTranspose3d(channel, channel, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            # nn.BatchNorm3d(channel),
            nn.ReLU(),
            nn.ConvTranspose3d(channel, channel, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            # nn.BatchNorm3d(channel),
        )
    
    def forward(self, x):
        return x + self.deconv_liner(x)

class invResTransBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(invResTransBlock, self).__init__()
        self.deconv_liner = nn.Sequential(
            nn.ConvTranspose3d(in_channel, in_channel, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            # nn.BatchNorm3d(in_channel),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channel, out_channel, (2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
            # nn.BatchNorm3d(out_channel)
        )
    
    def forward(self, x):
        return self.deconv_liner(x)

class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.conv_liner = nn.Sequential(
            nn.Conv3d(channel, channel, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(channel),
            nn.ReLU(),
            nn.Conv3d(channel, channel, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(channel)
        )
    
    def forward(self, x):
        return x + self.conv_liner(x)

class ResTransBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResTransBlock, self).__init__()
        self.conv_liner = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, (2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.Conv3d(out_channel, out_channel, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channel)
        )
    
    def forward(self, x):
        return self.conv_liner(x)