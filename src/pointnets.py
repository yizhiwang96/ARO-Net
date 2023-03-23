import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import ResnetBlockFC, CResnetBlockConv1d


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, reduce=False):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.reduce = reduce
        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))
        if self.reduce:
            # Recude to  B x F
            net = self.pool(net, dim=1)

            c = self.fc_c(self.actvn(net))
        else:
            c = net
        return c


class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, reduce=True, size_aux=(48, 32), use_bn=True):
        super().__init__()
        self.c_dim = c_dim
        self.reduce = reduce
        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim, size_aux=size_aux, use_bn=use_bn)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim, size_aux=size_aux, use_bn=use_bn)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim, size_aux=size_aux, use_bn=use_bn)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim, size_aux=size_aux, use_bn=use_bn)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim, size_aux=size_aux, use_bn=use_bn)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        
        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)
        
        
        net = self.block_4(net)
        if self.reduce:
            net = self.pool(net, dim=1)

            c = self.fc_c(self.actvn(net))
        else:
            c = net
        return c


class ResnetPointnetCondBN(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, reduce=True, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.reduce = reduce
        self.fc_pos = nn.Conv1d(dim, 2*hidden_dim, 1)
        self.block_0 = CResnetBlockConv1d(c_dim=c_dim, size_in=2*hidden_dim, size_h=hidden_dim, size_out=hidden_dim, norm_method=norm_method)
        self.block_1 = CResnetBlockConv1d(c_dim=c_dim, size_in=2*hidden_dim, size_h=hidden_dim, size_out=hidden_dim, norm_method=norm_method)
        self.block_2 = CResnetBlockConv1d(c_dim=c_dim, size_in=2*hidden_dim, size_h=hidden_dim, size_out=hidden_dim, norm_method=norm_method)
        self.block_3 = CResnetBlockConv1d(c_dim=c_dim, size_in=2*hidden_dim, size_h=hidden_dim, size_out=hidden_dim, norm_method=norm_method)
        self.block_4 = CResnetBlockConv1d(c_dim=c_dim, size_in=2*hidden_dim, size_h=hidden_dim, size_out=hidden_dim, norm_method=norm_method)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p, c):
        batch_size, T, D = p.size()
        p = p.permute(0, 2, 1) # N, 4, N_point

        net = self.fc_pos(p)
        net = self.block_0(net, c)

        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())

        net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net, c)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        
        net = self.block_2(net, c)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net, c)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        
        net = self.block_4(net, c)
        if self.reduce:
            net = self.pool(net, dim=2)

            c = self.fc_c(self.actvn(net))
        else:
            c = net
        return c
