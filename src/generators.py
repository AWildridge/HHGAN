import torch
from torch import nn
from torch.autograd.variable import Variable

from utils import ones_target


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self, n_features_in, n_features_out):
        super(GeneratorNet, self).__init__()
        self.n_features = n_features_in
        self.n_out = n_features_out

        self.hidden0 = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, self.n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

    def train(self, discriminator, fake_data):
        # Reset gradients
        self.optimizer.zero_grad()

        prediction = discriminator(fake_data)
        # Calculate error and backpropagate
        error = self.loss(prediction, ones_target(prediction.size(0)))
        error.backward()

        # Update weights with gradients
        self.optimizer.step()
        return error


def noise(size, n_features):
    """
    Generates 1D vectors of gaussian sampled random values. Number of vectors is equal to "size".

    :param size: The batch size.
    :param n_features: The number of random values to sample. The length of the 1D vector
    """
    return Variable(torch.randn(size, n_features))
