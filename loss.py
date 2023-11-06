"""
    Define the loss function.
    Margin loss and reconstruction loss for CapsNets.
    Cross entropy loss for resnets.
"""

import torch
import torch.nn as nn


class Criterion(nn.Module):
    def __init__(self, model_flag):
        super(Criterion, self).__init__()
        if model_flag[:3] == "res":
            self.criterion = nn.CrossEntropyLoss()
        elif model_flag[:2] == "bp":
            self.criterion = CapsuleLoss(
                reconstruction_loss_scalar=0.5)  # adjust reconstruction loss weight for BP-Capsnet
        else:
            self.criterion = CapsuleLoss()

    def forward(self, inputs, targets, outputs, reconstruction):
        device = outputs.device
        self.criterion = self.criterion.to(device)
        if reconstruction is None:
            loss = self.criterion(outputs, targets)
        else:
            loss = self.criterion(inputs, targets, outputs, reconstruction)

        return loss


class CapsuleLoss(nn.Module):
    """Combine margin loss & reconstruction loss of capsule network."""

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5, reconstruction_loss_scalar=5e-4):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.reconstruction_loss_scalar = reconstruction_loss_scalar  # assign weight
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, logits, reconstructions):
        # Shape of left / right / labels: (batch_size, num_classes)
        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)
        reconstruction_loss = self.mse(reconstructions.contiguous().view(images.shape), images)

        # Combine two losses
        return margin_loss + self.reconstruction_loss_scalar * reconstruction_loss
