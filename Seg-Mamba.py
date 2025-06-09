import os
import time
import datetime
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import weight_norm
from torch.nn import LayerNorm

import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from mamba_ssm import Mamba  



class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class Model(nn.Module):  
    def __init__(self, H, L, enc_in, num_layer, dropout, seg_len, channel_id, revin, D_STATE, DCONV, E_FACT):

        super(Model, self).__init__()


        self.H = H
        self.L = L
        self.enc_in = enc_in
        self.num_layer = num_layer
        self.dropout = dropout
        self.seg_len = seg_len
        self.channel_id = channel_id
        self.revin = revin
        self.D_STATE = D_STATE
        self.DCONV = DCONV
        self.E_FACT = E_FACT

        self.seg_num_x = self.H // self.seg_len
        self.seg_num_y = self.L // self.seg_len

        # embedding
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.num_layer),
            nn.ReLU()
        )
        self.linear = None  
        self.layernorm = nn.LayerNorm(self.num_layer)
        self.mamba1 = Mamba(d_model=self.num_layer, d_state=self.D_STATE, d_conv=self.DCONV, expand=self.E_FACT)
        self.mamba2 = Mamba(d_model=self.num_layer, d_state=self.D_STATE, d_conv=self.DCONV, expand=self.E_FACT)
        self.mhn = nn.Linear(self.H, self.num_layer)
        if self.channel_id:
            self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.num_layer // 2))
            self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.num_layer // 2))
        else:
            self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.num_layer))
        
    

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.num_layer, self.seg_len)
        )

        if self.revin:
            self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=False)

    def forward(self, x):
  
        batch_size = x.size(0)

        if self.revin:
            x = self.revinLayer(x, 'norm').permute(0, 2, 1)
        else:
            seq_last = x[:, -1:, :].detach()
            x = (x - seq_last).permute(0, 2, 1)  

        xhn = self.mhn(x)
        xhn = xhn.permute(1, 0, 2)
  
        # segment and embedding
        try:
            x_1 = x.view(-1, 1)  
            x_1 = self.linear1(x_1)  

    
            x_reshaped = x_1.view(-1, self.H)

            output1 = self.m(x_reshaped)

            x = output1.view(self.batch_size, self.seg_num_x, self.hidden_size)  

        except Exception as e:
            x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))
        else:
            if not self.reshaping_success: 
                print("Reshaping was successful, continuing with program...")
                self.reshaping_success = True  
           
            pass

        x_front = x
   
        x = self.mamba1(x)  # (bc, n, num_layer)
    
        x_back = x
     
        if self.channel_id:
      
            self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.num_layer // 2))
            self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.num_layer // 2))

            pos_emb = self.pos_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)  # (batch, seg_num_y, num_layer//2)
            channel_emb = self.channel_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)  # (batch, enc_in, num_layer//2)

          
            if self.enc_in > self.seg_num_y:
                channel_emb = channel_emb[:, :self.seg_num_y, :] 
            elif self.enc_in < self.seg_num_y:
                channel_emb = channel_emb.repeat(1, self.seg_num_y // self.enc_in + 1, 1)[:, :self.seg_num_y, :]  

            pos_emb = torch.cat([pos_emb, channel_emb], dim=-1)
        else:
           
            pos_emb = self.pos_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)


        x = torch.cat([x[:, -1:, :].repeat(1, self.seg_num_y, 1), pos_emb], dim=1)  
        a, b, c = x.shape
        if self.linear is None or self.linear.in_features != b:
            self.linear = nn.Linear(b, self.seg_num_x).to(x.device)  

        x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Mamba decoder
        x = self.mamba2(x)  ###(bc, num_x, num_layer) 

        y = self.predict(x[:, -self.seg_num_y:, :])  
 
        y = y.view(batch_size, self.seg_num_y, self.seg_len)  
 
        y = y.permute(0, 2, 1).contiguous().view(batch_size, self.enc_in, self.L)  

        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last  # (b, L, c)
    
        return y
