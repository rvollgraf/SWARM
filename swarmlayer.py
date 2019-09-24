import math

import numpy as np
import torch
import torch.nn as nn

from pooling import Pooling, Mean, Causal


class SwarmConvLSTMCell(nn.Module):

    def __init__(self, n_in, n_out, n_dim, pooling, cache=False):
        """
        Create a SwarmConvLSTMCell. We use 1-by-1 convolutions to carry on entities individually. The entities are aligned
        in a 1d or 2d spatial structure. Note that, unless pooling is 'CAUSAL', this setup is indeed permutation-equivariant.
        Populations of different sizes (different number of entities) can be grouped in one batch were missing entities
        will be padded and masked out.
        :param n_in: input dimension of the entities
        :param n_out: output dimension of the entities
        :param n_dim: dimension of the spatial arrangement of the entities (1 or 2)
        :param pooling: pooling method 'MEAN' or 'CAUSAL'
        :param cache: cache the result of self.Wih(x) in self.x_cache
        """
        assert isinstance(pooling, Pooling)
        assert pooling.n_dim == n_dim
        assert pooling.n_in == n_out

        super().__init__()

        self.n_in = n_in
        self.n_out = n_out

        if n_dim==2:
            # output is 4 time n_out because it will be split into
            # input, output, and forget gates, and cell input
            self.Wih = nn.Conv2d( n_in, 4 * n_out, (1,1), bias=True)
            self.Whh = nn.Conv2d(n_out, 4 * n_out, (1,1), bias=False)
            self.Whp = nn.Conv2d(pooling.n_out, 4 * n_out, (1,1), bias=False)
        elif n_dim==1:
            self.Wih = nn.Conv1d( n_in, 4 * n_out, 1, bias=True)
            self.Whh = nn.Conv1d(n_out, 4 * n_out, 1, bias=False)
            self.Whp = nn.Conv1d(pooling.n_out, 4 * n_out, 1, bias=False)
        else:
            raise ValueError("dim {} not supported".format(n_dim))

        self.n_dim = n_dim

        self.pooling = pooling

        self.cache = cache
        self.x_cache = None


    def forward(self,x, mask=None, hc=None):
        """
        Forward process the SWARM cell
        :param x: input, size is (N,n_in,E1,E2,...)
        :param mask: {0,1}-mask, size is (N,E1,E2,...)
        :param hc: (hidden, cell) state of the previous iteration or None. If not None both their size is (N,n_out, E1,E2,...)
        :return: (hidden, cell) of this iteration
        """
        # x is (N,n_in,...)
        
        x_sz = x.size()
        N,C = x_sz[:2]
        assert C==self.n_in

        if hc is None:
            c = torch.zeros( (N,self.n_out,*x_sz[2:]), dtype=x.dtype, device=x.device)
            tmp = self.Wih(x)  # (N,4*n_out, H,W)
            self.x_cache = tmp
        else:
            h,c = hc
            pool = self.Whp (self.pooling(h,mask))
            tmp = (self.x_cache if self.cache else self.Wih(x)) + self.Whh(h)  + pool  # (N,4*n_out, H,W)

        tmp = tmp.view(N,4,self.n_out,*x_sz[2:])

        ig = torch.sigmoid( tmp[:,0])
        fg = torch.sigmoid( tmp[:,1])
        og = torch.sigmoid( tmp[:,2])
        d  =  torch.tanh( tmp[:,3])

        c = c*fg + d*ig
        h = og * torch.tanh(c)

        return h,c




class SwarmLayer(nn.Module):

    def __init__(self,
                 n_in,
                 n_out,
                 n_hidden,
                 n_iter,
                 n_dim=2,
                 dropout=0.0,
                 pooling='CAUSAL',
                 channel_first=True,
                 cache=False):
        """
        Create a SwarmLayer that repeatedly executes a SwarmCell for a given number of iterations
        :param n_in: number of dimensions of input entities
        :param n_out: number of dimensions of output entities
        :param n_hidden: number of dimensions of entities in intermediate iterations
        :param n_iter: number of iterations
        :param n_dim: spatial entity layout (1 or 2)-d
        :param dropout: dropout rate (applied to h, not c, between iterations)
        :param pooling: to be used in the SWARM cell 'CAUSAL' or 'MEAN'
        :param channel_first: entity dimension is dimension 1, right after batch dimension (default), otherwise it is last
        :param cache: perform the computation of self.cell.Wih(x) only once and cache it over the rest of the iterations
        """
        super().__init__()

        self.n_iter = n_iter

        if pooling=='MEAN':
            pooling = Mean(n_hidden, n_hidden, n_dim)
        elif pooling=='CAUSAL':
            pooling = Causal(n_hidden, n_hidden, n_dim)
        elif isinstance(pooling,Pooling):
            pass
        else:
            raise ValueError

        self.cell = SwarmConvLSTMCell(n_in, n_hidden, n_dim=n_dim, pooling=pooling, cache=cache)

        self.n_dim = n_dim
        if n_dim==2:
            # an output feed forward layer after. Because channel_first is default, is is implemented by a 1-by-1 conv.
            self.ffwd = nn.Conv2d(2 * n_hidden, n_out, (1,1), bias=True)
        elif n_dim==1:
            self.ffwd = nn.Conv1d(2 * n_hidden, n_out, 1, bias=True)
        else:
            raise ValueError("dim {} not supported".format(n_dim))


        if dropout>0:
            self.drop = nn.Dropout2d(dropout)
        else:
            self.drop = None
            
        self.channel_first = channel_first


    def forward(self, x, mask=None):
        """
        forward process the SwarmLayer
        :param x: input
        :param mask: entity mask
        :return:
        """

        # 1. permute channels dimension to the end if not channels_first
        if not self.channel_first:
            if self.n_dim==1:
                x = x.transpose(1,2)
            elif self.n_dim==2:
                x = x.transpose(1,2).transpose(2,3)

        # 2. iteratively execute SWARM cell
        hc = None
        for i in range(self.n_iter):

            hc = self.cell(x,mask,hc)

            # 2a. apply dropout on h if desired
            if self.drop is not None:
                h,c = hc
                h = self.drop(h)
                hc = (h,c)

        # 3. execute the output layer on the concatenation of h an c
        h,c = hc
        hc = torch.cat((h, c), dim=1)
        y = self.ffwd(hc)

        # 4. back-permute the channels dimension
        if not self.channel_first:
            if self.n_dim==1:
                y = y.transpose(1,2)
            elif self.n_dim==2:
                y = y.transpose(2,3).transpose(1,2)
        return y


