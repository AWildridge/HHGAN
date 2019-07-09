import numpy as np
import os
import ROOT
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import uproot


class DiHiggsSignalMCDataset(torch.utils.data.Dataset):
    """The DiHiggs signal Monte Carlo (MC) dataset used for the PyTorch DataLoader

    Args:
        :param root (string): Root directory of the signal MC dataset.
        :param split (string, optional): The dataset split, supports ``train`` and ``val``
        :param download(bool, optional): If true, downloads the dataset using XRootD (http://xrootd.org/) and puts it in
            root directory, If dataset is already downloaded, it is not downloaded again.
        :param generator_level (bool, optional): If true, determine the pt, eta, phi, and mass of the b-jets from the
            generator level. If false, determine the pt, eta, phi, and mass from reconstruction level.
        :param normalize (bool, optional): If true, sets the 4 bottom quarks with transverse
            momentum, eta, phi, and mass to be between -1 and 1.

    Attributes:
        root: The root directory of the dataset.
        events: The 'Events' TTree in the ROOT file.
        b_quarks_pt: The transverse momentum for all of the bottom quarks originating from a Higgs boson
        b_quarks_eta: The pseudorapidity (https://en.wikipedia.org/wiki/Pseudorapidity) of the bottom quarks originating
            from a Higgs boson
        b_quarks_phi: The azimuthal angle of the bottom quarks originating from a Higgs boson
    """

    def __init__(self, root, split='train', download=False, generator_level=True, normalize=True):
        root = self.root = os.path.expanduser(root)

        # Download the HH MC signal data if it doesn't exist already.
        if download:
            # Opens via XRootD protocol
            self.events = uproot.open("root://cmsxrootd.fnal.gov///store/mc/RunIIFall17NanoAODv5/GluGluToHHTo4B_node_"
                                      "SM_13TeV-madgraph_correctedcfg/NANOAODSIM/PU2017_12Apr2018_Nano1June2019_102X_"
                                      "mc2017_realistic_v7-v1/40000/22D6CC16-CF5C-AE43-81F8-C3E8BD66A35E.root")
            self.events = self.events['Events']
        else:
            self.events = uproot.open(root + "/HH_Signal_MC.root")
            self.events = self.events['Events']

        if generator_level:
            # Determine flags to identify b's and anti-b's from Higgs bosons
            is_b_quark_mask = abs(self.events.array('GenPart_pdgId')) == 5
            mother_of_b_quarks_indices = self.events.array('GenPart_genPartIdxMother')[is_b_quark_mask]
            mother_is_higgs_mask = self.events.array('GenPart_pdgId')[mother_of_b_quarks_indices] == 25

            self.b_quarks_pt = self.events.array('GenPart_pt')[is_b_quark_mask][mother_is_higgs_mask]
            self.b_quarks_eta = self.events.array('GenPart_eta')[is_b_quark_mask][mother_is_higgs_mask]
            self.b_quarks_phi = self.events.array('GenPart_phi')[is_b_quark_mask][mother_is_higgs_mask]
        else:
            # Do from RECO level here
            raise NotImplementedError

        assert (len(self.b_quarks_eta) == len(self.b_quarks_phi) == len(self.b_quarks_pt)),\
            "Number of events is unequal in pt, eta, and phi"

        # Make sure we are only looking HH->bbbb
        num_b_quarks = np.array([len(self.b_quarks_pt[e]) for e in range(len(self.b_quarks_pt))])
        self.b_quarks_pt = self.b_quarks_pt[num_b_quarks == 4]
        self.b_quarks_eta = self.b_quarks_eta[num_b_quarks == 4]
        self.b_quarks_phi = self.b_quarks_phi[num_b_quarks == 4]

        if normalize:
            mean_pt = np.mean(self.b_quarks_pt)
            mean_eta = np.mean(self.b_quarks_eta)
            mean_phi = np.mean(self.b_quarks_phi)

            pt_range = np.amax(self.b_quarks_pt) - np.amin(self.b_quarks_pt)
            eta_range = np.amax(self.b_quarks_eta) - np.amin(self.b_quarks_eta)
            phi_range = np.amax(self.b_quarks_phi) - np.amin(self.b_quarks_phi)

            # This ensures that all data is between -1 to 1 to help GAN with gradients/learning
            self.b_quarks_pt = (self.b_quarks_pt - mean_pt) / (pt_range / 2)
            self.b_quarks_eta = (self.b_quarks_eta - mean_eta) / (eta_range / 2)
            self.b_quarks_phi = (self.b_quarks_phi - mean_phi) / (phi_range / 2)

        self.events_array = np.array([[self.b_quarks_pt[i][0], self.b_quarks_eta[i][0], self.b_quarks_phi[i][0],
                                       self.b_quarks_pt[i][1], self.b_quarks_eta[i][1], self.b_quarks_phi[i][1],
                                       self.b_quarks_pt[i][2], self.b_quarks_eta[i][2], self.b_quarks_phi[i][2],
                                       self.b_quarks_pt[i][3], self.b_quarks_eta[i][3], self.b_quarks_phi[i][3]]
                                      for i in range(len(self.b_quarks_pt))])

    def __len__(self):
        return len(self.b_quarks_phi)

    def __getitem__(self, index):
        """ Returns the properties of the bottom quarks associated with a single di-Higgs event
        :param index: The index of the event
        :return: The pt, phi, eta, and mass of the b and anti-b quarks in the event
        """
        return self.events_array[index]
