import numpy as np
import os
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
        :param normalize (bool, optional): If true, sets the features to all be between -1 and 1.

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
        self.generator_level = generator_level

        # Download the HH MC signal data if it doesn't exist already.
        if download:
            # Opens via XRootD protocol
            self.events = uproot.open("root://cmsxrootd.fnal.gov///store/mc/RunIIFall17NanoAODv5/GluGluToHHTo4B_node_"
                                      "SM_13TeV-madgraph_correctedcfg/NANOAODSIM/PU2017_12Apr2018_Nano1June2019_102X_"
                                      "mc2017_realistic_v7-v1/40000/22D6CC16-CF5C-AE43-81F8-C3E8BD66A35E.root")
            self.events = self.events['Events']
        else:
            self.events = uproot.open(root + "/Signal.root")
            self.events = self.events['Events']

        if generator_level:
            self.n_features = 16
            # Determine flags to identify b's and anti-b's from Higgs bosons
            is_b_quark_mask = abs(self.events.array('GenPart_pdgId')) == 5
            mother_of_b_quarks_indices = self.events.array('GenPart_genPartIdxMother')[is_b_quark_mask]
            mother_is_higgs_mask = self.events.array('GenPart_pdgId')[mother_of_b_quarks_indices] == 25

            self.b_quarks_pt = self.events.array('GenPart_pt')[is_b_quark_mask][mother_is_higgs_mask]
            self.b_quarks_eta = self.events.array('GenPart_eta')[is_b_quark_mask][mother_is_higgs_mask]
            self.b_quarks_phi = self.events.array('GenPart_phi')[is_b_quark_mask][mother_is_higgs_mask]
            self.b_quarks_mass = self.events.array('GenPart_mass')[is_b_quark_mask][mother_is_higgs_mask]

            assert (len(self.b_quarks_eta) == len(self.b_quarks_phi) == len(self.b_quarks_pt)), \
                "Number of events is unequal in pt, eta, and phi"

            # Make sure we are only looking HH->bbbb
            num_b_quarks = np.array([len(self.b_quarks_pt[e]) for e in range(len(self.b_quarks_pt))])
            self.b_quarks_pt = self.b_quarks_pt[num_b_quarks == 4]
            self.b_quarks_eta = self.b_quarks_eta[num_b_quarks == 4]
            self.b_quarks_phi = self.b_quarks_phi[num_b_quarks == 4]
            self.b_quarks_mass = self.b_quarks_mass[num_b_quarks == 4]

            if normalize:
                self.min_pt = np.amin(self.b_quarks_pt)
                self.min_eta = np.amin(self.b_quarks_eta)
                self.min_phi = np.amin(self.b_quarks_phi)
                self.min_mass = np.amin(self.b_quarks_mass)

                self.pt_range = np.amax(self.b_quarks_pt) - np.amin(self.b_quarks_pt)
                self.eta_range = np.amax(self.b_quarks_eta) - np.amin(self.b_quarks_eta)
                self.phi_range = np.amax(self.b_quarks_phi) - np.amin(self.b_quarks_phi)
                self.mass_range = np.amax(self.b_quarks_mass) - np.amin(self.b_quarks_mass)

                # This ensures that all data is between -1 to 1 to help GAN with gradients/learning
                self.b_quarks_pt = (self.b_quarks_pt - self.min_pt - (self.pt_range / 2)) / (self.pt_range / 2)
                self.b_quarks_eta = (self.b_quarks_eta - self.min_eta - (self.eta_range / 2)) / (self.eta_range / 2)
                self.b_quarks_phi = (self.b_quarks_phi - self.min_phi - (self.phi_range / 2)) / (self.phi_range / 2)
                self.b_quarks_mass = (self.b_quarks_mass - self.min_mass - (self.mass_range / 2)) / (
                            self.mass_range / 2)

            self.events_array = np.array(
                [[self.b_quarks_pt[i][0], self.b_quarks_eta[i][0], self.b_quarks_phi[i][0], self.b_quarks_mass[i][0],
                  self.b_quarks_pt[i][1], self.b_quarks_eta[i][1], self.b_quarks_phi[i][1], self.b_quarks_mass[i][1],
                  self.b_quarks_pt[i][2], self.b_quarks_eta[i][2], self.b_quarks_phi[i][2], self.b_quarks_mass[i][2],
                  self.b_quarks_pt[i][3], self.b_quarks_eta[i][3], self.b_quarks_phi[i][3], self.b_quarks_mass[i][3], ]
                 for i in range(len(self.b_quarks_pt))])
        else:
            self.n_features = 25
            self.jet_btags = self.events.array('Jet_btagDeepB')
            self.jet_pts = self.events.array('Jet_pt')
            self.jet_etas = self.events.array('Jet_eta')
            self.jet_phis = self.events.array('Jet_phi')
            self.jet_masses = self.events.array('Jet_mass')

            # Get rid of all b-tagged jets with score -2
            negative_two_mask = self.jet_btags != -2
            self.jet_btags = self.jet_btags[negative_two_mask]
            # Find events with >=3 b-tagged jets
            is_b = self.jet_btags > 0.226
            b_only_jets = self.jet_btags[is_b]
            has_3_btagged_jets = np.array([len(b_only_jets[i]) >= 3 for i in range(len(b_only_jets))])
            # Find events with >=3 b-tagged jets and 5 jets
            self.jet_btags = self.jet_btags[has_3_btagged_jets]
            has_atleast_5_jets = np.array([len(self.jet_btags[i]) >= 5 for i in range(len(self.jet_btags))])

            # Apply all of the masks on btags scores, pts, etas, phis, and masses
            self.jet_btags = self.jet_btags[has_atleast_5_jets]
            self.jet_pts = self.jet_pts[negative_two_mask][has_3_btagged_jets][has_atleast_5_jets]
            self.jet_etas = self.jet_etas[negative_two_mask][has_3_btagged_jets][has_atleast_5_jets]
            self.jet_phis = self.jet_phis[negative_two_mask][has_3_btagged_jets][has_atleast_5_jets]
            self.jet_masses = self.jet_masses[negative_two_mask][has_3_btagged_jets][has_atleast_5_jets]

            # Sort the jets based on b-tag score so that top 5 can be grabbed
            sorted_indices = [self.jet_btags[i].argsort()[::-1] for i in range(len(self.jet_btags))]
            self.jet_btags = [self.jet_btags[i][sorted_indices[i]][:5] for i in range(len(sorted_indices))]
            self.jet_pts = [self.jet_pts[i][sorted_indices[i]][:5] for i in range(len(sorted_indices))]
            self.jet_etas = [self.jet_etas[i][sorted_indices[i]][:5] for i in range(len(sorted_indices))]
            self.jet_phis = [self.jet_phis[i][sorted_indices[i]][:5] for i in range(len(sorted_indices))]
            self.jet_masses = [self.jet_masses[i][sorted_indices[i]][:5] for i in range(len(sorted_indices))]

            if normalize:
                self.min_btags = np.amin(self.jet_btags)
                self.min_pt = np.amin(self.jet_pts)
                self.min_eta = np.amin(self.jet_etas)
                self.min_phi = np.amin(self.jet_phis)
                self.min_mass = np.amin(self.jet_masses)

                self.btag_range = np.amax(self.jet_btags) - np.amin(self.jet_btags)
                self.pt_range = np.amax(self.jet_pts) - np.amin(self.jet_pts)
                self.eta_range = np.amax(self.jet_etas) - np.amin(self.jet_etas)
                self.phi_range = np.amax(self.jet_phis) - np.amin(self.jet_phis)
                self.mass_range = np.amax(self.jet_masses) - np.amin(self.jet_masses)

                # This ensures that all data is between -1 to 1 to help GAN with gradients/learning
                self.jet_btags = (self.jet_btags - self.min_btags - (self.btag_range / 2)) / (self.btag_range / 2)
                self.jet_pts = (self.jet_pts - self.min_pt - (self.pt_range / 2)) / (self.pt_range / 2)
                self.jet_etas = (self.jet_etas - self.min_eta - (self.eta_range / 2)) / (self.eta_range / 2)
                self.jet_phis = (self.jet_phis - self.min_phi) - (self.phi_range / 2) / (self.phi_range / 2)
                self.jet_masses = (self.jet_masses - self.min_mass - (self.mass_range / 2)) / (self.mass_range / 2)
            self.events_array = np.array([[self.jet_pts[i][0], self.jet_etas[i][0], self.jet_phis[i][0],
                                           self.jet_masses[i][0], self.jet_btags[i][0],
                                           self.jet_pts[i][1], self.jet_etas[i][1], self.jet_phis[i][1],
                                           self.jet_masses[i][1], self.jet_btags[i][1],
                                           self.jet_pts[i][2], self.jet_etas[i][2], self.jet_phis[i][2],
                                           self.jet_masses[i][2], self.jet_btags[i][2],
                                           self.jet_pts[i][3], self.jet_etas[i][3], self.jet_phis[i][3],
                                           self.jet_masses[i][3], self.jet_btags[i][3],
                                           self.jet_pts[i][4], self.jet_etas[i][4], self.jet_phis[i][4],
                                           self.jet_masses[i][4], self.jet_btags[i][4]]
                                          for i in range(len(self.jet_pts))])

    def __len__(self):
        if self.generator_level:
            return len(self.b_quarks_phi)
        else:
            return len(self.jet_pts)

    def __getitem__(self, index):
        """ Returns the properties of the bottom quarks associated with a single di-Higgs event
        :param index: The index of the event
        :return: The pt, phi, eta, and mass of the b and anti-b quarks in the event
        """
        return self.events_array[index]
