import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B3_Weights , EfficientNet_B0_Weights 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation as R  
import numpy as np
import os
import cv2
from scipy.io import loadmat
import time
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
from memory_profiler import profile




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
        

# Connector Class (sama)
class Connector(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=1):
        super(Connector, self).__init__()
        self.downsampler = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1 convolution
        self.scale_factor = scale_factor  # Faktor S untuk downsampling spasial (opsional)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.downsampler(x)  # Mengurangi channel
        
        # Downsampling spasial jika skala > 1
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=1/self.scale_factor, mode='bilinear', align_corners=False)
            _, _, H, W = x.shape
        
        x = x.view(B, -1, H * W)  # Mengubah jadi sekuensial
        x = x.permute(0, 2, 1)  
        return x

# Arsitektur Model yang Dimodifikasi
class HeadPosr(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(HeadPosr, self).__init__()
        # Backbone menggunakan EfficientNet
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Potong sampai layer terakhir sebelum pooling
        
        # Connector untuk downsampling dan reshaping
        self.connector = Connector(in_channels=1280, out_channels=16)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=16,max_len=64)
        
        # Transformer Encoder dengan dropout
        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, activation='relu', batch_first=True, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Dropout layer setelah transformer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Head untuk prediksi yaw, pitch, roll
        self.pose_head = nn.Linear(16, 3)

    def forward(self, x):
        # Backbone
        x = self.backbone(x)  # Output size: [batch_size, 1280, H, W]
        
        # Connector: downsampler and reshaper
        x = self.connector(x)  # Output size: [batch_size, A, 512]
        
        # Positional encoding
        x = self.positional_encoding(x)
        
        # Transformer 
        x = self.transformer(x)  # Output size: [A, batch_size, 512]

        # Global average pooling over the sequence length dimension (A)
        x = x.mean(dim=1)  # Output size: [batch_size, 512]

        # Pose prediction (yaw, pitch, roll)
        x = self.pose_head(x)  # Output size: [batch_size, 3]
        
        return x

