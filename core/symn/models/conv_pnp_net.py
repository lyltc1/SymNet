import numpy as np
import torch
from torch import nn
from mmengine.model import normal_init, constant_init


class ConvPnPNet(nn.Module):
    def __init__(self, nIn, rot_dim=9, drop_prob=0.0, dropblock_size=5):
        # if output is R_allo_6d, then rot_dim=6;
        # if output is R_allo, then rot_dim=9;
        super().__init__()
        self.featdim = 128
        self.drop_prob = drop_prob
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=0.0, block_size=dropblock_size),
            start_value=0.0,
            stop_value=drop_prob,
            nr_steps=5000,
        )

        self.features = nn.ModuleList()
        for i in range(3):
            _in_channels = nIn if i == 0 else self.featdim
            self.features.append(nn.Conv2d(_in_channels, self.featdim, kernel_size=3, stride=2, padding=1, bias=False))
            self.features.append(nn.GroupNorm(32, self.featdim))
            self.features.append(nn.ReLU(inplace=True))

        self.fc1 = nn.Linear(self.featdim * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_r = nn.Linear(256, rot_dim)

        self.fc_t = nn.Linear(256, 3)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        # init ------------------------------------
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.GroupNorm):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_r, std=0.01)
        normal_init(self.fc_t, std=0.01)

    def forward(self, visib_mask_prob, amodal_mask_prob, binary_code_prob):
        """
        output:
            rot_param: allo_rot6d params
            t_param: SITE params
        """
        x = torch.cat([visib_mask_prob, amodal_mask_prob, binary_code_prob], dim=1)

        if self.drop_prob > 0:
            self.dropblock.step()  # increment number of iterations
            x = self.dropblock(x)

        for layer in self.features:
            x = layer(x)

        x = x.view(-1, self.featdim * 16 * 16)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))

        rot_param = self.fc_r(x)
        t_param = self.fc_t(x)
        return rot_param, t_param


class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]
        self.i += 1


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(
            input=mask[:, None, :, :],
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2,
        )

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)
