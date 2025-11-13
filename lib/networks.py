import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging

from scipy import ndimage
from einops.layers.torch import Rearrange
from lib.decoders import DMWLADecoder
from lib import dmwla
from lib.maxxvit_4out import maxvit_tiny_rw_224 as maxvit_tiny_rw_224_4out
from lib.maxxvit_4out import maxvit_rmlp_tiny_rw_256 as maxvit_rmlp_tiny_rw_256_4out
from lib.maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out
from lib.maxxvit_4out import maxvit_rmlp_small_rw_224 as maxvit_rmlp_small_rw_224_4out
from lib.maxxvit_4out import maxvit_rmlp_small_rw_224 as maxvit_base_rw_224_4out
from lib.segformer import *
from lib.dsgim import DSGIM

logger = logging.getLogger(__name__)

def load_pretrained_weights(img_size, model_scale):
    
    if(model_scale=='tiny'):
        if img_size==224:
            backbone = maxvit_tiny_rw_224_4out()  # [64, 128, 320, 512]
            print('Loading:', './pretrained_pth/maxvit/maxvit_tiny_rw_224_sw-7d0dffeb.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxvit_tiny_rw_224_sw-7d0dffeb.pth')
        elif(img_size==256):
            backbone = maxvit_rmlp_tiny_rw_256_4out()
            print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_tiny_rw_256_sw-bbef0ff5.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_tiny_rw_256_sw-bbef0ff5.pth')
        else:
            sys.exit(str(img_size)+" is not a valid image size! Currently supported image sizes are 224 and 256.")
    elif(model_scale=='small'):
        if img_size==224:
            backbone = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]
            print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
        elif(img_size==256):
            backbone = maxxvit_rmlp_small_rw_256_4out()
            print('Loading:', './pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        else:
            sys.exit(str(img_size)+" is not a valid image size! Currently supported image sizes are 224 and 256.")
    else:
        sys.exit(model_scale+" is not a valid model scale! Currently supported model scales are 'tiny' and 'small'.")
       
    backbone.load_state_dict(state_dict, strict=False)
    print('Pretrain weights loaded.')
    return backbone
    
def decy_coefficient(layers):
    decay = torch.exp(-(0.25 - 2 ** (-2.5 - 5* torch.arange(layers, dtype=torch.float) / layers)))
    return decay
    
def attention(k, inchannel):
        mylayer = dmwla.DynamicMemoryWeightsLossAttention(
            k, inchannel, nn.Conv2d, pool_size_per_cluster=100, 
            num_k=4, feature_dim=128, warmup_total_iter=2000, cp_momentum=0.3,
            cp_phi_momentum=0.95)
            
        return mylayer
        
class SimMPNet(nn.Module):
    def __init__(self, n_class=1, img_size_s1=(256,256), img_size_s2=(224,224), model_scale='samll', interpolation='bilinear'):
        super(SimMPNet, self).__init__()
        self.in_dim = [96, 192, 384, 768]
        in_dim, key_dim, value_dim, layers = [[96, 192, 384, 768], [96, 192, 384, 768], [96, 192, 384, 768], [2, 2, 6, 2]]
        head_count = 1
        token_mlp="mix_skip"
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        image_size = 224
        self.k = nn.Parameter(torch.tensor(50.0), requires_grad=False)
        self.DSGIM = DSGIM(in_dim[2])
        
        #DMW-LA
        self.DMWLA_1 = nn.ModuleList(
            [attention(self.k, in_dim[0]) for _ in range(layers[0])]
        )
        self.DMWLA_2 = nn.ModuleList(
            [attention(self.k, in_dim[1]) for _ in range(layers[1])]
        )
        
        self.DMWLA_3 = nn.ModuleList(
            [attention(self.k, in_dim[2]) for _ in range(layers[2])]
        )
        self.DMWLA_4 = nn.ModuleList(
            [attention(self.k, in_dim[3]) for _ in range(layers[3])]
        )
        self.patch_embed1 = OverlapPatchEmbeddings(
            image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0]
        )
        self.patch_embed2 = OverlapPatchEmbeddings(
            image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], in_dim[0], in_dim[1]
        )
        self.patch_embed3 = OverlapPatchEmbeddings(
            image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], in_dim[1], in_dim[2]
        )
        self.patch_embed4 = OverlapPatchEmbeddings(
            image_size // 16, patch_sizes[3], strides[3], padding_sizes[3], in_dim[2], in_dim[3]
        )
        
        self.n_class = n_class
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.model_scale = model_scale     
        self.interpolation = interpolation
        
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        # backbone network initialization with pretrained weight
        self.backbone1 = load_pretrained_weights(self.img_size_s1[0], self.model_scale)
        backbone3 = self.backbone1
        self.stem_1 = backbone3.stem
        self.encoder1_1 = backbone3.stages[0]
        self.encoder1_2 = backbone3.stages[1]
        self.encoder1_3 = backbone3.stages[2]
        self.encoder1_4 = backbone3.stages[3]
        
        if(self.model_scale=='tiny'):
            self.channels = [512, 256, 128, 64]
        elif(self.model_scale=='base'):
            self.channels = [768, 384, 192, 96]
        elif(self.model_scale=='small'):
            self.channels = [768, 384, 192, 96]
            
        self.out_head1_1_in = nn.Conv2d(3, self.channels[3], 1)
        self.out_head1_2_in = nn.Conv2d(3, self.channels[2], 1)
        self.out_head1_3_in = nn.Conv2d(3, self.channels[1], 1)
        self.out_head1_4_in = nn.Conv2d(3, self.channels[0], 1)
        
        self.out_head2_1_in = nn.Conv2d(3, self.channels[3], 1)
        self.out_head2_2_in = nn.Conv2d(3, self.channels[2], 1)
        self.out_head2_3_in = nn.Conv2d(3, self.channels[1], 1)
        self.out_head2_4_in = nn.Conv2d(3, self.channels[0], 1)
        
        self.decoder = DMWLADecoder(channels=self.channels)

        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)

        self.out_head4_in = nn.Conv2d(self.channels[0], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.size()[1] == 1:
            x = self.conv(x)
        
        #MERIT encoder
        x = F.interpolate(x, size=self.img_size_s1, mode=self.interpolation)
        f1 = self.stem_1(x)  

        f1 = self.encoder1_1(f1)

        f1 = f1 + self.sigmoid(f1) * (self.out_head1_1_in(F.interpolate(x, size=(64, 64), mode=self.interpolation)))

        f2 = self.encoder1_2(f1)

        f2 = f2 + self.sigmoid(f2) * (self.out_head1_2_in(F.interpolate(x, size=(32, 32), mode=self.interpolation)))

        f3 = self.encoder1_3(f2)
        f3 = f3 + self.sigmoid(f3) * (self.out_head1_3_in(F.interpolate(x, size=(16, 16), mode=self.interpolation)))

        
        f4 = self.encoder1_4(f3)
        f4 = f4 + self.sigmoid(f4) * (self.out_head1_4_in(F.interpolate(x, size=(8, 8), mode=self.interpolation)))

        p14_in = self.out_head4_in(f4)
        p14_in = self.sigmoid(p14_in)
        
        p14_in = F.interpolate(p14_in, scale_factor=32, mode=self.interpolation)
        x_in = x * p14_in
        
        
        #DMW-LA encoder
        e1 = F.interpolate(x_in, size=self.img_size_s2, mode=self.interpolation)
        B = e1.shape[0]
        e1, H, W = self.patch_embed1(e1)
        e1 = e1.permute(0, 2, 1).contiguous().reshape(-1, self.in_dim[0], H, W)

        for DMWLA in self.DMWLA_1:
            e1 = DMWLA(e1, device="cuda", evaluation=False)
        e1 = e1 + self.sigmoid(e1) * (self.out_head2_1_in(F.interpolate(x_in, size=(56, 56), mode=self.interpolation)))

        e2, H, W = self.patch_embed2(e1)
        e2 = e2.permute(0, 2, 1).contiguous().reshape(-1, self.in_dim[1], H, W)
        
        for DMWLA in self.DMWLA_2:
            e2 = DMWLA(e2, device="cuda", evaluation=False)
        e2 = e2 + self.sigmoid(e2) * (self.out_head2_2_in(F.interpolate(x_in, size=(28, 28), mode=self.interpolation)))
        
        e3, H, W = self.patch_embed3(e2)
        e3 = e3.permute(0, 2, 1).contiguous().reshape(-1, self.in_dim[2], H, W)

        for DMWLA in self.DMWLA_3:
            i = 1
            e3 = DMWLA(e3, device="cuda", evaluation=False)
            gamm = decy_coefficient(12)   
            e3 = self.DSGIM(e3,gamm[i])
            i = i + 1
            
        e3 = e3 + self.sigmoid(e3) * (self.out_head2_3_in(F.interpolate(x_in, size=(14, 14), mode=self.interpolation)))
        
        e4, H, W = self.patch_embed4(e3)
        e4 = e4.permute(0, 2, 1).contiguous().reshape(-1, self.in_dim[3], H, W)
        for DMWLA in self.DMWLA_4:
            e4 = DMWLA(e4, device="cuda", evaluation=False)
        e4 = e4 + self.sigmoid(e4) * (self.out_head2_4_in(F.interpolate(x_in, size=(7, 7), mode=self.interpolation)))

        #DMW-LA decoder
        skip1_0 = F.interpolate(f1, size=(e1.shape[-2:]), mode=self.interpolation)
        skip1_1 = F.interpolate(f2, size=(e2.shape[-2:]), mode=self.interpolation)
        skip1_2 = F.interpolate(f3, size=(e3.shape[-2:]), mode=self.interpolation)
        skip1_3 = F.interpolate(f4, size=(e4.shape[-2:]), mode=self.interpolation)
        
        x21_o, x22_o, x23_o, x24_o = self.decoder(e4+skip1_3, [e3+skip1_2, e2+skip1_1, e1+skip1_0])

        p21 = self.out_head1(x21_o)
        p22 = self.out_head2(x22_o)
        p23 = self.out_head3(x23_o)
        p24 = self.out_head4(x24_o)
              
              
        #Prediction head
        p21 = F.interpolate(p21, size=(self.img_size_s1), mode=self.interpolation)
        p22 = F.interpolate(p22, size=(self.img_size_s1), mode=self.interpolation)
        p23 = F.interpolate(p23, size=(self.img_size_s1), mode=self.interpolation)
        p24 = F.interpolate(p24, size=(self.img_size_s1), mode=self.interpolation)
        
        p1 = p21
        p2 = p22
        p3 = p23
        p4 = p24

        return p1, p2, p3, p4
        
if __name__ == '__main__':
    model = SimMPNet().cuda()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()

    p1, p2, p3, p4 = model(input_tensor)
    print(p1.size(), p2.size(), p3.size(), p4.size())

