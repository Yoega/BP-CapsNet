"""
    Simple implementation of BP-CapsNet
"""

import torch
from torch import nn


def squash(x, dim=-1):
    """
    :param x: tensor needs to be squashed
    :param dim: the dimension needs to be squashed along
    :return: squashed x, with the same dimension of x
    """
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / squared_norm.sqrt()


class PrimaryCaps(nn.Module):
    """Primary capsule layer."""

    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        # Each conv unit stands for a single capsule.
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * num_conv_units,
                              kernel_size=kernel_size,
                              stride=stride)
        self.out_channels = out_channels

    def forward(self, x):
        # Shape of x: (batch_size, in_channels, height, weight)
        # Shape of out: out_capsules * (batch_size, out_channels, height, weight)
        out = self.conv(x)
        # Flatten out: (batch_size, out_capsules * height * weight, out_channels)
        batch_size = out.shape[0]
        return squash(out.contiguous().view(batch_size, -1, self.out_channels), dim=-1)


class RoutingMoudule(nn.Module):
    """The Routing module."""

    def __init__(self, in_dim, in_caps, out_caps, out_dim, num_routing):
        """
        Initialize the module.

        Args:
            in_dim: 		Dimensionality of each capsule vector.
            in_caps: 		Number of input capsules if digits layer.
            out_caps: 		Number of capsules in the capsule layer
            out_dim: 		Dimensionality, of the output capsule vector.
            num_routing:	Number of iterations during routing algorithm
        """
        super(RoutingMoudule, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.num_routing = num_routing
        self.W = nn.Parameter(0.01 * torch.randn(1, out_caps, in_caps, out_dim, in_dim),
                              requires_grad=True)

        self.pearson = Pearson()

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        # W @ x =
        # (1, out_caps, in_caps, out_dim, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, out_caps, in_caps, out_dims, 1)
        u_hat = torch.matmul(self.W, x)
        # (batch_size, out_caps, in_caps, out_dim)
        u_hat = u_hat.squeeze(-1)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        b = torch.zeros(batch_size, self.out_caps, self.in_caps, 1).to(device)
        v1, v2 = 0., 0.
        p = 0.
        for route_iter in range(self.num_routing - 1):
            # (batch_size, out_caps, in_caps, 1) -> Softmax along out_caps
            c = b.softmax(dim=1)

            # element-wise multiplication
            # (batch_size, out_caps, in_caps, 1) * (batch_size, in_caps, out_caps, out_dim) ->
            # (batch_size, out_caps, in_caps, out_dim) sum across in_caps ->
            # (batch_size, out_caps, out_dim)
            # s = (c * temp_u_hat).sum(dim=2)

            # s0: (batch_size, out_caps, in_caps, out_dim)
            s0 = c * temp_u_hat
            # s: (batch_size, out_caps, out_dim)
            s = s0.sum(dim=2)

            if route_iter == 0:
                # v1: (batch_size, out_caps, out_dim)
                v1 = squash(s)
            else:
                # v2: (batch_size, out_caps, out_dim)
                v2 = squash(s)
                # p: (batch_size, out_caps, 1)
                p = torch.norm(v2, dim=-1, keepdim=True)
                p = torch.unsqueeze(p, dim=2)

            # s: (batch_size, out_caps, out_dim)-->(batch_size, out_caps, 1, out_dim)
            s = torch.unsqueeze(s, dim=2)
            # s: (batch_size, out_caps, 1, out_dim)-->(batch_size, out_caps, in_caps, out_dim)
            s = s.repeat(1, 1, self.in_caps, 1)
            s = s - s0  # for self-exclusion

            # apply "squashing" non-linearity along out_dim
            # v: (batch_size, out_caps, in_caps, out_dim)
            v = squash(s)

            # pearson correlation coefficient
            # uv: (batch_size, out_caps, in_caps, 1)
            uv = self.pearson(temp_u_hat, v)

            # b: (batch_size, out_caps, in_caps, 1)
            b += uv

        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        # Bayes's theorem
        c = c * p
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along out_dim
        v = squash(s)

        return v, v1, v2


class BPCapsNet(nn.Module):
    """Basic implementation of BP-CapsNet."""

    def __init__(self, channel_n, class_n):
        """
        Initialize BP-CapsNet

        Args:
             channel_n: number of channels
             class_n: number of classes
        """
        super(BPCapsNet, self).__init__()

        self.class_n = class_n
        self.channel_n = channel_n

        # Conv2d layer
        self.conv = nn.Conv2d(self.channel_n, 256, 9)
        self.relu = nn.ReLU(inplace=True)

        # Primary capsule
        self.primary_caps = PrimaryCaps(num_conv_units=32,
                                        in_channels=256,
                                        out_channels=8,
                                        kernel_size=9,
                                        stride=2)

        # routing
        self.routing_module = RoutingMoudule(in_dim=8,
                                    in_caps=32 * 6 * 6,
                                    out_caps=self.class_n,
                                    out_dim=16,
                                    num_routing=3)

        # Reconstruction module
        self.decoder = nn.Sequential(
            nn.Linear(16 * self.class_n, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784*self.channel_n),
            nn.Sigmoid())

    def forward(self, x):
        device = x.device
        out = self.relu(self.conv(x))
        out = self.primary_caps(out)
        out, v1, v2 = self.routing_module(out)

        # Shape of logits: (batch_size, out_capsules)
        logits1 = torch.norm(v1, dim=-1)
        logits2 = torch.norm(v2, dim=-1)
        logits = torch.norm(out, dim=-1)
        pred = torch.eye(self.class_n).to(device).index_select(dim=0, index=torch.argmax(logits, dim=1))

        # Reconstruction
        batch_size = out.shape[0]
        reconstruction = self.decoder((v1 * pred.unsqueeze(2)).contiguous().view(batch_size, -1))

        return logits, reconstruction


class MarginLoss(nn.Module):
    """Combine margin loss & reconstruction loss of capsule network."""

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(MarginLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda

    def forward(self, logits, labels):
        # Shape of left / right / labels: (batch_size, num_classes)
        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)

        return margin_loss


class ReconsLoss(nn.Module):

    def __init__(self, theta):
        super(ReconsLoss, self).__init__()
        self.theta = theta
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, images, reconstructions):
        reconstruction_loss = self.theta * (self.mse(reconstructions.contiguous().view(images.shape), images))

        return reconstruction_loss


class Pearson(nn.Module):
    """
        Compute Pearson Coefficient
    """
    def forward(self, a, b):
        a = a - torch.mean(a, dim=-1, keepdim=True)
        b = b - torch.mean(b, dim=-1, keepdim=True)
        cost = torch.sum(a*b, dim=-1, keepdim=True) / (torch.norm(a, dim=-1, keepdim=True) * torch.norm(b, dim=-1, keepdim=True))
        return cost


