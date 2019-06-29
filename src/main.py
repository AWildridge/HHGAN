import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

from dihiggs_dataset import DiHiggsSignalMCDataset
from discriminators import DiscriminatorNet
from generators import GeneratorNet

data = DiHiggsSignalMCDataset("/home/aj/CMS_Research/HH_4b/13TeV_Data/MC", download=False, generator_level=True,
                              normalize=True)
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
num_batches = len(data_loader)

discriminator = DiscriminatorNet()
generator = GeneratorNet()
