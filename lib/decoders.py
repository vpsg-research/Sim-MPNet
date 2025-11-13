import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import dmwla

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        #print(x.shape)
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3, stride=1, padding=1, groups=1):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
        
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        #print(x.shape)
        max_pool_out= self.max_pool(x) #torch.topk(x,3, dim=1).values

        max_out = self.fc2(self.relu1(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out) 

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)    

def attention(k, inchannel):
        mylayer = dmwla.DynamicMemoryWeightsLossAttention(
            k, inchannel, nn.Conv2d, pool_size_per_cluster=100, 
            num_k=10, feature_dim=128, warmup_total_iter=2000, cp_momentum=0.3,
            cp_phi_momentum=0.95)
        return mylayer
        
class DMWLADecoder(nn.Module):
    def __init__(self, channels=[768, 384, 192, 96]):
        super(DMWLADecoder,self).__init__()
        self.k = nn.Parameter(torch.tensor(50.0), requires_grad=False)
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])

        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])
        
        self.DMWLA_1 = attention(self.k, channels[3])
        self.DMWLA_2 = attention(self.k, channels[2])
        self.DMWLA_3 = attention(self.k, channels[1])
        self.DMWLA_4 = attention(self.k, channels[0])
        
        self.SA = SpatialAttention()
      
    def forward(self, x, skips):
        # up4
        d4 = self.Conv_1x1(x)
        d4 = self.DMWLA_4(d4, device="cuda", evaluation=False)
        d4 = self.ConvBlock4(d4)
        
        # up3
        d3 = self.Up3(d4)
        x3 = self.CA3(d3+skips[0]) * (d3+skips[0])
        x3 = self.SA(x3)*(x3) 
        d3 = d3 + x3    
        d3 = self.DMWLA_3(d3, device="cuda", evaluation=False)
        d3 = self.ConvBlock3(d3)
        # up2
        d2 = self.Up2(d3)
        x2 = self.CA2(d2+skips[1]) * (d2+skips[1])
        x2 = self.SA(x2)*(x2) 
        d2 = d2 + x2
        d2 = self.DMWLA_2(d2, device="cuda", evaluation=False)
        d2 = self.ConvBlock2(d2)

        # up1
        d1 = self.Up1(d2)
        x1 = self.CA1(d1+skips[2]) * (d1+skips[2])
        x1 = self.SA(x1)*(x1) 
        d1 = d1 + x1
        d1 = self.DMWLA_1(d1, device="cuda", evaluation=False)
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1
