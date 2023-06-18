import torch
from torch import nn

# Available device
device = torch.device('cuda:1' if torch.cuda.is_available() else'cpu')
# 'cuda:2' if torch.cuda.is_available() else

def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)


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


class DigitCaps(nn.Module):
    """Digit capsule layer."""

    def __init__(self, in_dim, in_caps, out_caps, out_dim, num_routing):
        """
        Initialize the layer.

        Args:
            in_dim: 		Dimensionality of each capsule vector.
            in_caps: 		Number of input capsules if digits layer.
            out_caps: 		Number of capsules in the capsule layer
            out_dim: 		Dimensionality, of the output capsule vector.
            num_routing:	Number of iterations during routing algorithm
        """
        super(DigitCaps, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.num_routing = num_routing
        self.device = device
        self.W = nn.Parameter(0.01 * torch.randn(1, out_caps, in_caps, out_dim, in_dim),
                              requires_grad=True)

    def forward(self, x, y):    # y---->labels--->(batch_size, out_caps)
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

        b = torch.zeros(batch_size, self.out_caps, self.in_caps, 1).to(self.device)

        for route_iter in range(self.num_routing - 1):
            # (batch_size, out_caps, in_caps, 1) -> Softmax along out_caps
            c = b.softmax(dim=1)

            # element-wise multiplication
            # (batch_size, out_caps, in_caps, 1) * (batch_size, in_caps, out_caps, out_dim) ->
            # (batch_size, out_caps, in_caps, out_dim) sum across in_caps ->
            # (batch_size, out_caps, 1, out_dim)
            s = (c * temp_u_hat).sum(dim=2, keepdim=True)

            '''
            # compute routing_loss
            if route_iter == 0:
                loss_fun = CapsuleLoss()
                temp = torch.norm(squash(s).squeeze(dim=2), dim=-1)
                # (batch_size, out_caps)@((batch_size, out_caps, 1, out_dim))
                routing_loss = loss_fun(y, temp, state=False)
                # weighing routing_loss
                # routing_loss = routing_loss * 2
                correct = torch.sum(torch.argmax(y, dim=1) == torch.argmax(temp, dim=1)).item()
                print(correct/batch_size*1.0)
            '''

            '''
            # expand s in the dim-->in_caps(1->1152)->
            # summation vector minus each vector itself
            # (batch_size, out_caps, in_caps, out_dim)
            s = s.repeat([1, 1, self.in_caps, 1]) - temp_u_hat
            '''
            # [batch_size, out_caps, 1, out_dim]-->[batch_size, out_caps, in_caps, out_dim]
            # [batch_size, out_caps, in_caps, out_dim]
            s = s.repeat([1, 1, self.in_caps, 1]) - temp_u_hat
            # apply "squashing" non-linearity along out_dim
            # squash the vector along last dimension
            v = squash(s)

            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, out_caps, in_caps, out_dim) @ (batch_size, out_caps, out_dim, out_dim)
            # -> (batch_size, out_caps, in_caps, out_dim)-->(batch_size, out_caps, out_dim, 1)
            uv = temp_u_hat * v
            uv = uv.sum(dim=3, keepdim=True)
            b += uv

        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along out_dim
        v = squash(s)

        return v  # , routing_loss, correct


class CapsNet(nn.Module):
    """Basic implementation of capsule network layer."""

    def __init__(self):
        super(CapsNet, self).__init__()

        # attention layer
        self.attention_ = SELayer(3)

        # Conv2d layer
        self.conv = nn.Conv2d(3, 256, 9)
        self.relu = nn.ReLU(inplace=True)

        # Primary capsule
        self.primary_caps = PrimaryCaps(num_conv_units=32,
                                        in_channels=256,
                                        out_channels=8,
                                        kernel_size=9,
                                        stride=2)

        # attention layer
        self.attention = SELayer(256)

        # Digit capsule
        self.digit_caps = DigitCaps(in_dim=8,
                                    in_caps=32 * 6 * 6,
                                    out_caps=5,
                                    out_dim=16,
                                    num_routing=3)

        # Reconstruction layer
        self.decoder = nn.Sequential(
            nn.Linear(16 * 5, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3*28*28),
            nn.Sigmoid())

    def forward(self, x, y):
        out = self.relu(self.conv(x))
        # out = self.attention(out)
        out = self.primary_caps(out)

        # loss is the routing_loss
        # sec_correct to sum up the second_routing number
        out, loss, sec_correct = self.digit_caps(out, y)

        # Shape of logits: (batch_size, out_capsules)
        logits = torch.norm(out, dim=-1)
        pred = torch.eye(5).to(device).index_select(dim=0, index=torch.argmax(logits, dim=1))

        # Reconstruction
        batch_size = out.shape[0]
        reconstruction = self.decoder((out * pred.unsqueeze(2)).contiguous().view(batch_size, -1))

        return logits, reconstruction, loss, sec_correct


class CapsuleLoss(nn.Module):
    """Combine margin loss & reconstruction loss of capsule network."""

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.reconstruction_loss_scalar = 5e-4
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, labels, logits, images=None, reconstructions=None, state=True):
        # Shape of left / right / labels: (batch_size, num_classes)
        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)
        reconstruction_loss = self.mse(reconstructions.contiguous().view(images.shape), images)
        '''
        # Reconstruction loss
        if state:
            reconstruction_loss = self.mse(reconstructions.contiguous().view(images.shape), images)
        else:
            reconstruction_loss = 0  # when margin_loss only in routing
        
        '''



        # Combine two losses
        return margin_loss + self.reconstruction_loss_scalar * reconstruction_loss


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
