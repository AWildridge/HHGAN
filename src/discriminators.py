import torch
from torch import nn

from utils import ones_target
from utils import zeros_target


class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self, n_features_in):
        super(DiscriminatorNet, self).__init__()
        self.n_features = n_features_in
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(self.n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

    def train(self, real_data, fake_data):
        # Reset gradients
        self.optimizer.zero_grad()

        # Train on Real Data
        prediction_real = self(real_data)
        # Calculate error and backpropagate
        error_real = self.loss(prediction_real, ones_target(real_data.size(0)))
        error_real.backward()

        # Train on Fake Data
        prediction_fake = self(fake_data)
        # Calculate error and backpropagate
        error_fake = self.loss(prediction_fake, zeros_target(real_data.size(0)))
        error_fake.backward()

        # Update weights with gradients
        optimizer.step()

        return error_real + error_fake, prediction_real, prediction_fake
