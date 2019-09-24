import numpy as np
import torch
import torch.nn as nn



class Pooling(nn.Module):

    def __init__(self, n_in, n_out, n_dim):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_dim = n_dim

        assert self.n_dim==1 or self.n_dim==2


    def forward(self, x, mask):
        # x is (N, n_in, E) or (N, n_in, E1, E2)
        # mask is (N, E) or (N, E1, E2)

        raise NotImplementedError("Pooling is only an abstract bas class")

class Mean( Pooling):

    def __init__(self, n_in, n_out, n_dim):
        super().__init__(n_in, n_out, n_dim)

        assert n_in==n_out


    def forward(self, x, mask=None):

        x_sz = x.size()

        if self.n_dim==1:
            pooling_dim = 2
        else:
            pooling_dim = (2,3)

        if mask is None:
            # 2. compute mean over spatial dimensions
            pool = x.mean(dim=pooling_dim, keepdim=True).expand(x_sz)
        else:
            # 2. compute masked mean over spatial dimensions
            mask = mask.view((x_sz[0], 1, *x_sz[2:])).float()
            pool = (x * mask).sum(dim=pooling_dim, keepdim=True).expand(x_sz)
            pool = pool / mask.sum(dim=pooling_dim, keepdim=True).expand(x_sz)
            pool = pool.view(x_sz)

        return pool



class Causal( Pooling):

    def __init__(self, n_in, n_out, n_dim):
        super().__init__(n_in, n_out, n_dim)

        assert n_in == n_out

    def forward(self, x, mask=None):

        if mask is not None:
            raise NotImplementedError("Causal pooling is not yet implemented for masked input!")

        x_sz = x.size()

        # 1. flatten all spatial dimensions
        pool = x.view((x_sz[0], self.n_in, -1))
        # 2. compute cumulative means of non-successort entities
        pool = torch.cumsum(pool, dim=2) / (torch.arange(np.prod(x_sz[2:]), device=pool.device).float() + 1.0).view(1, 1, -1)
        # 3. reshape to the original spatial layout
        pool = pool.view(x_sz)

        return pool

