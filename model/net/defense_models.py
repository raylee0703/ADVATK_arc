import torch
from torch import nn
import torch.nn.functional as F

channel = 1 # grayscale=1 RGB=3

class autoencoder(nn.Module):
    def __init__(self, in_channel = channel):
        super(autoencoder, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(in_channel, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding = 1)
        self.conv4 = nn.Conv2d(32, 3, 3, padding = 1)
        self.conv5 = nn.Conv2d(3, in_channel, 3, padding = 1)
        self.conv64 = nn.Conv2d(64, 64, 3, padding = 1)
        self.avg = nn.AvgPool2d(2)
        
        self.up = nn.Upsample(scale_factor = 2)
        
        self.sig = nn.Sigmoid()
        #self.BN = nn.BatchNorm2d(channel,affine=True)

        self.dequant = torch.quantization.DeQuantStub()

        
    def forward(self, x):
        #x = self.BN(x)
        x = self.quant(x) ###
        x = self.sig(self.conv1(x))
        x = self.avg(x)
        x = self.sig(self.conv2(x))        
        x = self.sig(self.conv64(x))
        x = self.sig(self.conv3(x))
        x = self.up(x)
        x = self.sig(self.conv4(x))
        x = self.sig(self.conv5(x))
        x = self.dequant(x) ###
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'conv2', 'conv3', 'conv4', 'conv5']], inplace=True)
    
class autoencoder2(nn.Module):
    def __init__(self, in_channel):
        super(autoencoder2, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, in_channel, 3, padding = 1)
        
        self.sig = nn.Sigmoid()
        self.BN = nn.BatchNorm2d(3,affine=True)
    def forward(self, x):
        x = self.BN(x)
        x = self.sig(self.conv1(x))
        x = self.sig(self.conv2(x))
        x = self.sig(self.conv2(x))
        x = self.sig(self.conv3(x))
        return x
