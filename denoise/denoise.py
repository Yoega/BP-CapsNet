import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

# mean filter
import torch
import torch.nn as nn
channel_n = 3
# define mean filter layer
class MeanFilter(nn.Module):
    def __init__(self, kernel_size):
        super(MeanFilter, self).__init__()
        self.kernel_size = kernel_size
        # initialize a convolutional layer
        self.conv = nn.Conv2d(in_channels=channel_n, out_channels=channel_n, kernel_size=kernel_size,
                              stride=1, padding=kernel_size // 2, bias=False, groups=channel_n)

        # initialize filter weights
        self.initialize_weights()

    def forward(self, x):
        return self.conv(x)

    def initialize_weights(self):
        # calculate mean filter kernel
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size) / (self.kernel_size * self.kernel_size)
        kernel = kernel.repeat(channel_n, 1, 1, 1)  # repeat the kernel for each channel

        # set convolutional layer weights
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False  # fixed weights, no gradient required


def GaussianBlur(batch_img, ksize, sigma=0):
    kernel = getGaussianKernel(ksize, sigma) 
    B, C, H, W = batch_img.shape # C：number of image channels, to be used by group convolution
    # generate the convolution kernel of group convolution
    kernel = kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)
    pad = (ksize - 1) // 2 
    # mode=relfect--->more suitable for calculating the weight of edge pixels
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
    weighted_pix = F.conv2d(batch_img_pad, weight=kernel, bias=None,
                           stride=1, padding=0, groups=C)
    return weighted_pix

@torch.no_grad()
def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
     
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center) # the lateral distance of the element from the center of the matrix
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2)) 
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum() # 归一化
    return kernel

