import numpy as np
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

from dihiggs_dataset import DiHiggsSignalMCDataset
from discriminators import DiscriminatorNet
from generators import GeneratorNet
from generators import noise


# Number of features being used. For now: pt, eta, phi for all 4 b quarks
NUM_FEATURES = 12

# Load the data
data = DiHiggsSignalMCDataset("/home/aj/CMS_Research/HH_4b/13TeV_Data/MC", download=False, generator_level=True,
                              normalize=False)
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
num_batches = len(data_loader)

# Create the discriminator and generator
discriminator = DiscriminatorNet(NUM_FEATURES)
generator = GeneratorNet(NUM_FEATURES, NUM_FEATURES)

# Set-up the learning algorithm and loss function
discriminator.optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
generator.optimizer = optim.Adam(generator.parameters(), lr=0.0002)
# Binary Cross-Entropy loss function
# https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression
discriminator.loss = generator.loss = nn.BCELoss()

# Every set number of epochs we will test generator to see how well the generator is performing
NUM_TEST_SAMPLES = 100
test_noise = noise(NUM_TEST_SAMPLES, NUM_FEATURES)

# Total number of epochs to train
NUM_EPOCHS = 1

# Training happens here
for epoch in range(NUM_EPOCHS):
    for n_batch, real_batch in enumerate(data_loader):
        N = real_batch.size(0)
        real_data = Variable(real_batch)

        # Generate fake data and detach
        # (so gradients are not calculated for generator)
        fake_data = generator(noise(N, NUM_FEATURES)).detach()

        # Train Discriminator
        d_error, d_prediction_real, d_prediction_fake = discriminator.train(real_data, fake_data)

        # Generate fake data
        fake_data = generator(noise(N, NUM_FEATURES))
        # Train Generator
        g_error = generator.train(discriminator, fake_data)

        # Display Progress every few batches
        if n_batch % 100 == 0:
            test_events = generator(test_noise)
            test_events = test_events.data

            # TODO: Implement logging of progress

