import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging

softmax = nn.Softmax(dim=-1)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W,  = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)

    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(B, -1, window_size*window_size, C)
    return windows
    
def windows_select(x, k):
    a_0 = window_partition(x, 7)  # (B, num_windows, 7*7, C)
    cos_sim = torch.cosine_similarity(a_0.unsqueeze(-2), a_0.unsqueeze(-3), dim=-1)
    triu = torch.triu(cos_sim) 
    sum_sim = torch.sum(torch.sum(triu, dim=-2), dim=-1, keepdim = True)
    top_k_values, top_k_indices = torch.topk(sum_sim, k=k, dim=-2)  
    return top_k_indices

def third_layer_mask(n):
    x_1 = torch.arange(7)  
    x_2 = torch.arange(7)
    x = torch.stack(torch.meshgrid([x_1, x_1]))
    x = torch.flatten(x, 1)
    b = torch.from_numpy(np.tile(x[0:1, :], (7, 1))[:, 0:14])
    b_1 = b + 2
    mask_2_layer = torch.cat([b, b_1], dim = -2).unsqueeze(0).repeat(n, 1, 1)  # B, H, W
    return mask_2_layer

def mask_assignment(x, y):
    a = torch.zeros_like(y).cuda()
    B, k, _ = x.shape
    q = []
    for i in range(B):
        x_1 = x[i:i+1, :, :].cuda()  #Mask by batch
        y_1 = y[i:i+1, :, :].cuda()
        mask = torch.isin(y_1, x_1)  #Find the same elements, the reserved area
        y_1[mask] = 100  
        y_1[y_1 < 99] = 1
        y_1[y_1 == 100] = 0 
        a = torch.cat([a, y_1], dim=0)
    return a[B:, :, :]

def spitial_sim_weight(x):
    '''
    x.shape: (B*N, num_heads, window_size * window_size, C/num_heads)
    '''
    BN, HW, HW = x.shape

    spa = torch.mean(x.cuda(), dim=-1, keepdim=True) 
    spa = spa.repeat(1, 1, HW)
    os_spa = abs(spa - torch.transpose(spa, -1, -2))
    
    spa_values, spa_indices = torch.topk(os_spa, k=HW//2, dim=-1, largest=True)  #Select the top k most relevant elements for each element
    last_value = spa_values[:, :, -1:]  #Pick out the k-th maximum value
    last_value = last_value.repeat(1, 1, HW).cuda()
    os_spa_1 = os_spa.detach().clone()

    a = torch.eye(HW).cuda()
    os_spa_1 = os_spa_1 + a * 10000
    os_spa_1 = torch.where(os_spa_1 < last_value, torch.tensor(0).cuda(), torch.tensor(1).cuda())  #mask_one
    
    os_spa_2 = torch.where(os_spa_1 > last_value, torch.tensor(0).cuda(), torch.tensor(1).cuda())  #mask_two
    
    x_spi_weig_1 = os_spa_1 * x
    x_spi_weig_1[x_spi_weig_1 == 0] = -100
    x_spi_weig_1 = softmax(x_spi_weig_1)

    x_spi_weig_2 = os_spa_2 * x
    x_spi_weig_2[x_spi_weig_2 == 0] = -100
    x_spi_weig_2 = softmax(x_spi_weig_2)
    
    return x_spi_weig_1 + x_spi_weig_2

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_in, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class DSGIM(nn.Module):
    def __init__(self, in_channel):
        super(DSGIM, self).__init__()
        self.in_channel = in_channel
        self.query = nn.Conv2d(self.in_channel,self.in_channel,kernel_size=1,stride=1,padding=0)
        self.key = nn.Conv2d(self.in_channel,self.in_channel,kernel_size=1,stride=1,padding=0)
        self.value = nn.Conv2d(self.in_channel,self.in_channel,kernel_size=1,stride=1,padding=0)
        self.ConvBlock = conv_block(self.in_channel, self.in_channel//4)
        self.fc = nn.Linear(in_channel, in_channel)
        
    def forward(self, x, gamm):
        B1, C1, H1, W1 = x.shape
        third_mask = third_layer_mask(B1)
        mask = mask_assignment(windows_select(x, 2), third_mask).unsqueeze(1)
        mask_1 = mask.clone().detach()
        mask_1[mask_1==1] = 100
        mask_1[mask_1==0] = 1
        mask_1[mask_1==100] = 0
        mask[0] = gamm
        mask_1[0] = gamm
        
        x_1 = x * mask
        x_2 = x * mask_1
        x_1 = self.query(x_1).view(B1, C1, -1).transpose(-1,-2)
        x_2 = self.key(x_2).view(B1, C1, -1).transpose(-1,-2)
        x = self.value(x).view(B1, C1, -1).transpose(-1,-2)
        spital_attn_1 = torch.matmul(x_1, x_2.transpose(-1, -2))
        spital_attn_weights = spitial_sim_weight(spital_attn_1)
        spital_feature = torch.matmul(spital_attn_weights, x)
        
        out = self.fc(spital_feature).permute(0, 2, 1).view(-1, 384, 14, 14)
        out = self.ConvBlock(out)
        return out