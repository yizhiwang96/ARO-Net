import torch
import torch.nn as nn

class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None, size_aux=(48, 32), use_bn=True):
        super().__init__()
        # Attributes
        self.use_bn = use_bn
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        self.n_anc = size_aux[0]
        self.n_local = size_aux[1]

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.n_anc * self.size_h)

    def perform_bn(self, net, idx_l=0):
        n_dim = net.shape[-1]
        net = net.view(-1, self.n_anc, self.n_local, n_dim).permute(0, 1, 3, 2).reshape(-1, self.n_anc * n_dim, self.n_local)
        net = self.bn(net)
        net = net.view(-1, self.n_anc, n_dim, self.n_local).permute(0, 1, 3, 2).reshape(-1, self.n_local, n_dim)
        return net

    def forward(self, x):

        net = self.fc_0(self.actvn(x)) # B * N * M, n_local, size_h
        if self.use_bn:
            net = self.perform_bn(net)
        dx = self.fc_1(self.actvn(net))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.
    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks 
    '''

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm'):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules

        self.bn_0 = CBatchNorm1d(
            c_dim, size_h, norm_method=norm_method)
        self.bn_1 = CBatchNorm1d(
            c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.
    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        elif norm_method == 'layer_norm':
            self.bn = nn.LayerNorm(f_dim, elementwise_affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):

        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        net = self.bn(x)

        out = gamma * net + beta

        return out
