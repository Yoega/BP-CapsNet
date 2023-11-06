import numpy as np
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn


def denoise_selector(denoise, channel_n):
    """
    :param denoise: the name of denoising method
    :param channel_n: number of channels
    :return: method for noise reduction
    """
    if denoise == "svd":
        # # SVD module

        return SVD(15, channel_n)
    elif denoise == "mean":
        # mean value filter
        # default kernel size is 3
        return MeanFilter(3, channel_n)
    elif denoise == "gaussian":
        # Guassian Blur
        # default kernel size is 3
        return GaussianBlur(3)
    else:
        raise NotImplementedError


class SVD(nn.Module):
    """
        Use SVD to reduce noise.
        Number of selected eigenvalues (k in the paper) needs to be predefined.
    """
    def __init__(self, num_eig, channel_n):
        super(SVD, self).__init__()
        self.num_eig = num_eig  # number of selected eigenvalues
        self.channel_n = channel_n

    def forward(self, inputs):
        inputs = torch.reshape(inputs, [-1, 28 * self.channel_n, 28])
        U, sigma, VT = torch.linalg.svd(inputs)  # obtain matrix U and matrix VT, and eigenvalues
        inputs = torch.matmul(
            (torch.matmul(U[:, :, 0:self.num_eig], torch.diag_embed(sigma)[:, 0:self.num_eig, 0:self.num_eig])),
            VT[:, 0:self.num_eig, :])
        inputs = torch.reshape(inputs, [-1, self.channel_n, 28, 28])

        return inputs


class MeanFilter(nn.Module):
    """
        Use mean value filter to reduce noise.
        The kernel size needs to be predefined (default = 3).
    """
    def __init__(self, kernel_size, channel_n):
        super(MeanFilter, self).__init__()
        self.kernel_size = kernel_size
        self.channel_n = channel_n
        # initialize a conv layer
        self.conv = nn.Conv2d(in_channels=self.channel_n, out_channels=self.channel_n, kernel_size=kernel_size,
                              stride=1, padding=kernel_size // 2, bias=False, groups=self.channel_n)

        # initialize the filter weight
        self.initialize_weights()

    def forward(self, x):

        return self.conv(x)

    def initialize_weights(self):
        # compute the mean value filter kernel
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size) / (self.kernel_size * self.kernel_size)
        kernel = kernel.repeat(self.channel_n, 1, 1, 1)  # Repeat the kernel for each channel

        # set the conv weight
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False  # fix the weight, no grad


class GaussianBlur(nn.Module):
    """
        Use Gaussian filter to reduce noise.
        The kernel size needs to be predefined (default = 3).
    """
    def __init__(self, kernel_size):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, batch_img, sigma=0):
        kernel = self.getGaussianKernel(self.kernel_size, sigma)  # generate the weight
        B, C, H, W = batch_img.shape  # C：channel number of the image
        # generate the conv kernel of group convolution
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(C, 1, 1, 1)
        pad = (self.kernel_size - 1) // 2
        # mode=relfect is more suitable for computing the weight of edge pixels
        batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
        weighted_pix = F.conv2d(batch_img_pad, weight=kernel, bias=None,
                                stride=1, padding=0, groups=C)
        return weighted_pix

    @torch.no_grad()
    def getGaussianKernel(self, kernel_size, sigma=0):
        if sigma <= 0:
            # calculate the default sigma according to kernelsize, consistent with opencv
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        center = kernel_size // 2
        xs = (np.arange(kernel_size, dtype=np.float32) - center)  # the lateral distance of the element from the
        # center of the matrix
        kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))
        kernel = kernel1d[..., None] @ kernel1d[None, ...]
        kernel = torch.from_numpy(kernel)
        kernel = kernel / kernel.sum()  # normalize
        return kernel
