import torch
from torch.nn import functional
from torch import nn
from utils.generic_utils import sequence_mask


class L1LossMasked(nn.Module):
    def __init__(self):
        super(L1LossMasked, self).__init__()

    def forward(self, input, target, length):
        """
        Args:
            input: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = sequence_mask(
            sequence_length=length, max_len=target.size(1)).unsqueeze(2).float()
        mask = mask.expand_as(input)
        loss = functional.l1_loss(
            input * mask, target * mask, reduction="sum")
        loss = loss / mask.sum()
        return loss


class MSELossMasked(nn.Module):
    def __init__(self):
        super(MSELossMasked, self).__init__()

    def forward(self, input, target, length):
        """
        Args:
            input: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = sequence_mask(
            sequence_length=length, max_len=target.size(1)).unsqueeze(2).float()
        mask = mask.expand_as(input)
        loss = functional.mse_loss(
            input * mask, target * mask, reduction="sum")
        loss = loss / mask.sum()
        return loss


class JitterLayer(nn.Module):
    """Jitters the input around and calculates a difference for two images"""
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        """
        Args:
            input: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
        """
        def get_opposite(a, b):
            if a == 0 and b == 0:
                return 0, 0
            elif a == 1 and b == 0:
                return 0, 1
            elif a == 0 and b == 1:
                return 1, 0
            else:
                raise ValueError

        # shifts in Left Right Top Bottom order
        shifts = ((0, 0, 0, 0),  # No op
                  (0, 0, 1, 0),  # shift down
                  (0, 0, 0, 1),  # shift up
                  (1, 0, 0, 0),  # shift right
                  (0, 1, 0, 0),  # shift left
                  (1, 0, 1, 0),  # shift down right
                  (0, 1, 0, 1),  # shift up left
                  (0, 1, 1, 0),  # shift down left
                  (1, 0, 0, 1)  # shift up right
        )

        shifted_diffs = []
        for L, R, T, B in shifts:
            # get opposing shifts for target
            L_T, R_T = get_opposite(L, R)
            T_T, B_T = get_opposite(T, B)

            shifted_input = nn.ConstantPad2d((L, R, T, B), 0)(input)
            shifted_target = nn.ConstantPad2d((L_T, R_T, T_T, B_T), 0)(target)
            shifted_diff = torch.abs(shifted_input - shifted_target)

            # cut off padding again
            shifted_diff = shifted_diff[:, L:-R, T:-B]
            shifted_diffs.append(shifted_diff)

        diffs_concatenated = torch.cat(shifted_diffs, dim=-1)

        center_and_min_avg = torch.mean(diffs_concatenated[:, :, 0],
                                        smin(diffs_concatenated, 32))

        return center_and_min_avg


def smin(t, k):
    """Smooth minimum function."""
    res = torch.sum(torch.exp(-k * t), dim=-1)
    return -1 * torch.div(torch.log(res), k)


class JitterLoss(nn.Module):
    def __init__(self, masked=True):
        super().__init__()
        self.jitter = JitterLayer()
        self.masked = masked

    def forward(self, input, target, length):
        """
        Args:
            input: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value masked by the length.
        """

        jittered = self.jitter(input, target)
        if self.masked:
            mask = sequence_mask(
                sequence_length=length, max_len=target.size(1)).unsqueeze(
                2).float()
            mask = mask.expand_as(input)
            jittered = mask * jittered
        loss = torch.mean(jittered)
        return loss
