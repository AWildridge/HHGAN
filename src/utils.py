import torch
from torch.autograd.variable import Variable


def ones_target(size):
    """Tensor containing ones, with shape = size"""
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available():
        return data.cuda()
    return data


def zeros_target(size):
    """Tensor containing zeros, with shape = size"""
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available():
        return data.cuda()
    return data

