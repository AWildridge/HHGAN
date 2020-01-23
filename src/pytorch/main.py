from array import array
import numpy as np
from ROOT import TCanvas, TGraph
import torch
from torch import nn, optim
from torch.autograd.variable import Variable

from dihiggs_dataset import DiHiggsSignalMCDataset
from discriminators import DiscriminatorNet
from generators import GeneratorNet
from generators import noise


# Batch size - Number of events to be done per weight update (via batch Stochastic gradient descent)
BATCH_SIZE = 100

# Load the data
data = DiHiggsSignalMCDataset("/home/aj/CMS_Research/HH_4b/13TeV_Data/MC", download=False, generator_level=False,
                              normalize=True)
data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

num_batches = len(data_loader)
# Number of features being used. For now: pt, eta, phi, mass, and b-tag score for all 4 b quarks
NUM_FEATURES = data.n_features


# Create the discriminator and generator
discriminator = DiscriminatorNet(NUM_FEATURES)
generator = GeneratorNet(NUM_FEATURES, NUM_FEATURES)

# Use GPU if available
if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()

# Set-up the learning algorithm and loss function
discriminator.optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
generator.optimizer = optim.Adam(generator.parameters(), lr=0.0002)
# Binary Cross-Entropy loss function
# https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression
discriminator.loss = generator.loss = nn.BCELoss()

# Every set number of epochs we will test generator to see how well the generator is performing
NUM_TEST_SAMPLES = 200000
test_noise = noise(NUM_TEST_SAMPLES, NUM_FEATURES)

# Total number of epochs to train
NUM_EPOCHS = 100

# Lists to record error
discriminator_errors = array('d')
generator_errors = array('d')
batch_ids = array('d')
num_minibatches = 0
num_tests = 0

# Make the canvases for the GIFs...
c_pt = TCanvas("c_pt", "c_pt", 700, 700)
c_eta = TCanvas("c_eta", "c_eta", 700, 700)
c_phi = TCanvas("c_phi", "c_phi", 700, 700)
c_mass = TCanvas("c_mass", "c_mass", 700, 700)
c_btag = TCanvas("c_btag", "c_btag", 700, 700)

make_gif = False

# Training happens here
for epoch in range(NUM_EPOCHS):
    print("Epoch #" + str(epoch))
    for minibatch_id, real_batch in enumerate(data_loader):
        num_events_in_batch = real_batch.size(0)
        real_data = Variable(real_batch)

        # Generate fake data and detach
        # (so gradients are not calculated for generator)
        fake_data = generator(noise(num_events_in_batch, NUM_FEATURES)).detach()

        # Train Discriminator
        disc_error, disc_prediction_real, disc_prediction_fake = discriminator.train(real_data, fake_data)

        # Generate fake data
        fake_data = generator(noise(num_events_in_batch, NUM_FEATURES))
        # Train Generator
        gen_error = generator.train(discriminator, fake_data)

        discriminator_errors.append(disc_error)
        generator_errors.append(gen_error)
        batch_ids.append(num_minibatches)
        num_minibatches += 1

        if make_gif:
            # Display Progress every few batches
            if num_minibatches % 1000 == 0:
                print("Minibatch #" + str(num_minibatches))
                test_events = generator(test_noise)
                test_events = test_events.data

                # Make and fill the histograms...
                h_pt = TH1F("h_pt", "GAN Generated Jet p_T; p_T; Entries", 20, -1, 1)
                h_eta = TH1F("h_eta", "GAN Generated Jet \eta; \eta; Entries", 20, -1, 1)
                h_phi = TH1F("h_phi", "GAN Generated Jet \phi; \phi; Entries", 20, -1, 1)
                h_mass = TH1F("h_mass", "GAN Generated Jet mass; m; Entries", 20, -1, 1)
                h_btag = TH1F("h_btag", "GAN Generated Jet b-tag scores; b-tag scores; Entries", 20, -1, 1)

                for i in range(len(test_events)):
                    for j in range(5):
                        h_pt.Fill(test_events[i][0+(j*5)])
                        h_eta.Fill(test_events[i][1+(j*5)])
                        h_phi.Fill(test_events[i][2+(j*5)])
                        h_mass.Fill(test_events[i][3+(j*5)])
                        h_btag.Fill(test_events[i][4+(j*5)])

                # Draw the gifs...
                c_pt.cd()
                h_pt.Draw()
                c_pt.Update()
                c_pt.SaveAs("/output/ptgif/gan_pt"+str(num_tests)+".png")

                c_eta.cd()
                h_eta.Draw()
                c_eta.Update()
                c_eta.SaveAs("/output/etagif/gan_eta"+str(num_tests)+".png")

                c_phi.cd()
                h_phi.Draw()
                c_phi.Update()
                c_phi.SaveAs("/output/phigif/gan_phi"+str(num_tests)+".png")

                c_mass.cd()
                h_mass.Draw()
                c_mass.Update()
                c_mass.SaveAs("/output/massgif/gan_mass"+str(num_tests)+".png")

                c_btag.cd()
                h_btag.Draw()
                c_btag.Update()
                c_btag.SaveAs("/output/btaggif/gan_btag"+str(num_tests)+".png")

                num_tests += 1

torch.save(generator.state_dict(), "gan_generator_1k_epochs.pt")
torch.save(discriminator.state_dict(), "gan_discriminator_1k_epochs.pt")
torch.save(discriminator_errors, "discriminator_errors.pt")
torch.save(generator_errors, "generator_errors.pt")
