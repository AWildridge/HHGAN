import argparse
import ast
import logging
import math
import os
from array import array

import numpy as np
import scipy
from scipy import stats
from scipy.special import comb

import ROOT
from dihiggs_dataset import DiHiggsSignalMCDataset
from ROOT import TH1F, TCanvas, TGraph, TLorentzVector

PT_INDEX = 0 # index of transverse momentum in event data
ETA_INDEX = 1 # index of pseudorapidity in event data
PHI_INDEX = 2 # index of phi in event data
MASS_INDEX = 3 # index of mass in event data
BTAG_INDEX = 4 # index of b-tag score in event data
FEATURES_PER_JET = 5
JETS_PER_EVENT = 5

# Below lists are used for plotting purposes
FEATURE_NAME_LIST = ['pts', 'etas', 'phis', 'masses', 'btags']
MOMENTUMS = ['pxs', 'pys', 'pzs']
DELTA_RS = ['dr12', 'dr13', 'dr14', 'dr15', 'dr23', 'dr24', 'dr25', 'dr34', 'dr35', 'dr45']
DELTA_R_HIGGS = ['dr12']
HIGGS_FEATURES = [['lead_pts', 'lead_masses'], ['subl_pts', 'subl_masses'], ['dihiggs_etas', 'dihiggs_masses']]


def scale_data(gen_datas, dataset):
    ''' Re-scales the data to be back to within its regular range. 

    Args:
        :param gen_datas: The generative-model-based dataset (GAN)
        :param dataset: The Monte Carlo-based dataset
    '''
    num_events = len(gen_datas[0][:,0::5])
    num_jets = int(dataset.n_features / FEATURES_PER_JET)

    all_pts = np.array([], dtype=np.float).reshape(0, num_events * num_jets)
    all_etas = np.array([], dtype=np.float).reshape(0, num_events * num_jets)
    all_phis = np.array([], dtype=np.float).reshape(0, num_events * num_jets)
    all_masses = np.array([], dtype=np.float).reshape(0, num_events * num_jets)
    all_btags = np.array([], dtype=np.float).reshape(0, num_events * num_jets)
    
    for events in gen_datas:
        pts = (events[:,PT_INDEX::num_jets] * dataset.pt_range) + dataset.min_pt
        etas = (events[:,ETA_INDEX::num_jets] * dataset.eta_range) + dataset.min_eta
        phis = (events[:,PHI_INDEX::num_jets] * dataset.phi_range) + dataset.min_phi
        masses = (events[:,MASS_INDEX::num_jets] * dataset.mass_range) + dataset.min_mass
        btags = (events[:,BTAG_INDEX::num_jets] * dataset.btag_range) + dataset.min_btags

        pts = np.reshape(pts, num_events * num_jets)
        etas = np.reshape(etas, num_events * num_jets)
        phis = np.reshape(phis, num_events * num_jets)
        masses = np.reshape(masses, num_events * num_jets)
        btags = np.reshape(btags, num_events * num_jets)

        all_pts = np.vstack((all_pts, pts))
        all_etas = np.vstack((all_etas, etas))
        all_phis = np.vstack((all_phis, phis))
        all_masses = np.vstack((all_masses, masses))
        all_btags = np.vstack((all_btags, btags))

    return all_pts, all_etas, all_phis, all_masses, all_btags


def draw_hists(hist_list, canvas, legend, title_prefix, folder=None):
    ''' Function for drawing the histograms. Grabs the maximum y-value from all the histograms in
    the list. Saves the histogram plot with name title_prefix in the folder specified. Legend is also drawn.

    Args:
        :param hist_list: The list of TH1 histograms
        :param canvas: The TCanvas to draw on
        :param legend: The TLegend for the different histograms
        :param title_prefix: The name of the file. Saved with a '.png' and '.root' extension
        :param folder: The folder to save the histogram plot in. Default is 'None'
    '''
    max_y = 0
    legend.SetFillColor(0)
    legend.SetLineColor(0)

    for hist in hist_list:
        if hist.GetSumOfWeights() > 0:
            hist.Scale(1 / hist.GetSumOfWeights())
            if hist.GetBinContent(hist.GetMaximumBin()) > max_y:
                max_y = hist.GetBinContent(hist.GetMaximumBin())

    for i in range(len(hist_list)):
        hist = hist_list[i]
        hist.SetLineColor(4 - (1 * i))
        hist.SetLineWidth(2)
        hist.SetMaximum(max_y + 0.1)
        if hist.GetSumOfWeights() > 0:
            if i == 0:
                hist.DrawNormalized("HIST")
            else:
                hist.DrawNormalized("HIST SAME")
        if i == len(hist_list) - 1:
            legend.AddEntry(hist, "MC Data")
        else:
            legend.AddEntry(hist, names[i] + ' Batches')
    legend.Draw()
    if not folder is None:
        title_prefix = test_path + '/' + folder + '/' + title_prefix
    canvas.SaveAs(title_prefix + '.pdf')
    canvas.SaveAs(title_prefix + '.root')


def generate_hist_params(name_prefix, title_prefix, feature, name_index):
    hist_params = {}
    hist_params['nbins'] = 20
    hist_params['min_x'] = 0
    hist_params['max_x'] = 700
    hist_params['name'] = name_prefix + feature
    hist_params['title'] = title_prefix
    if not name_index is None:
        hist_params['name'] += '_' + names[name_index]
    if 'dr' in feature:
        particle_nums = feature.split('dr')[1]
        hist_params['title'] += particle_nums[0] + ' and ' + particle_nums[1] + '; ; Events'
        hist_params['min_x'] = 0
        hist_params['max_x'] = 10
    elif 'lead' in feature or 'subl' in feature or 'dihiggs' in feature:
        feature_name = feature.split('_')[1]
        hist_params['title'] += feature_name.capitalize() + '; '
        if feature_name == 'pts':
            hist_params['title'] += 'Pt [GeV]; Events'
            hist_params['min_x'] = 0
            hist_params['max_x'] = 700
        elif feature_name == 'masses':
            hist_params['min_x'] = 0
            if 'lead' in feature:
                hist_params['title'] += 'm_{H,lead} [GeV]; Events'
            elif 'subl' in feature:
                hist_params['title'] += 'm_{H,subl} [GeV]; Events'
            elif 'dihiggs' in feature:
                hist_params['title'] += 'm_{HH} [GeV]; Events'
                hist_params['max_x'] = 1000
        elif feature_name == 'etas':
            hist_params['title'] += '\eta; Events'
            hist_params['min_x'] = -3
            hist_params['max_x'] = 3
    else:
        hist_params['title'] += feature.capitalize() + '; '
        if feature == 'pts':
            hist_params['title'] += 'Pt [GeV]; Events'
            hist_params['min_x'] = 0
            hist_params['max_x'] = 700
        elif feature == 'etas':
            hist_params['title'] += '\eta; Events'
            hist_params['min_x'] = -3
            hist_params['max_x'] = 3
        elif feature == 'phis':
            hist_params['title'] += '\phi; Events'
            hist_params['min_x'] = -math.pi
            hist_params['max_x'] = math.pi
        elif feature == 'masses':
            hist_params['title'] += 'm_{jet} [GeV]; Events'
            hist_params['min_x'] = 0
            hist_params['max_x'] = 500
        elif feature == 'btags':
            hist_params['title'] += 'b-tag score; Events'
            hist_params['min_x'] = 0
            hist_params['max_x'] = 1
        elif feature == 'pxs':
            hist_params['title'] += 'Px [GeV]; Events'
            hist_params['min_x'] = -700
            hist_params['max_x'] = 700
        elif feature == 'pys':
            hist_params['title'] += 'Py [GeV]; Events'
            hist_params['min_x'] = -700
            hist_params['max_x'] = 700
        elif feature == 'pzs':
            hist_params['title'] += 'Pz [GeV]; Events'
            hist_params['min_x'] = -700
            hist_params['max_x'] = 700
    return hist_params


def plot_dists(gen_features, mc_features, feature_list, gan_generic_title, mc_generic_title):
    input_level_hist_list = [[] for i in range(len(feature_list))]
    c = TCanvas("", "", 700, 700)

    # Used for separating all the different plots being generated. Split into 4 different folders.
    for feature_index in range(len(feature_list)):
        # The name of the distribution, e.g., pt, eta, phi, di-Higgs mass, Higgs momentum, delta R, etc.
        feature_name = feature_list[feature_index]

        # Fill all the histograms from the GAN generated data
        for i in range(len(gen_features[feature_index])):
            sample_features = gen_features[feature_index][i]
            hist_params = generate_hist_params('h_generated_', gan_generic_title, feature_name, i)
            hist = TH1F(hist_params['name'], hist_params['title'], hist_params['nbins'], hist_params['min_x'], hist_params['max_x'])
            for feature in sample_features:
                hist.Fill(feature)
            input_level_hist_list[feature_index].append(hist)

        # Fill histogram from MC data
        for i in range(len(mc_features[feature_index])):
            sample_features = mc_features[feature_index][i]
            hist_params = generate_hist_params('h_mc_', mc_generic_title, feature_name, None)
            hist = TH1F(hist_params['name'], hist_params['title'], hist_params['nbins'], hist_params['min_x'], hist_params['max_x'])
            for feature in sample_features:
                hist.Fill(feature, 0.2) # TODO: This helps with plotting but not with STATISTICS!!!
            input_level_hist_list[feature_index].append(hist)
        
        # Find the folder name and correct location for the legend
        leg = ROOT.TLegend(0.55, 0.5, 0.89, 0.7)
        folder_name = ''
        if feature_list == FEATURE_NAME_LIST:
            folder_name = 'InputLevel'
            if feature_index == 1 or feature_index == 2:
                leg = ROOT.TLegend(0.35, 0.15, 0.75, 0.25)
            elif feature_index == 4:
                leg = ROOT.TLegend(0.5, 0.5, 0.8, 0.7)
        elif feature_list == MOMENTUMS:
            folder_name = 'PXYZ'
        elif feature_list == DELTA_RS:
            folder_name = 'Jet_DeltaR'
        elif feature_list == DELTA_R_HIGGS or feature_list in HIGGS_FEATURES:
            folder_name = 'Higgs'
        # finally draw the histograms for this feature. Save in the specified folder_name.
        draw_hists(input_level_hist_list[feature_index], c, leg, feature_name, folder=folder_name)


def calc_pxyz(all_pts, all_etas, all_phis):
    '''Calculates the x, y, and z coordinates of the 4-momentum.
    
    Args:
        :param all_pts: The collection of all transverse momentums being analyzed. Structure is K x M x 5
                        where K is the number of collections of events and M is the number of events. There
                        are 5 jets per event
        :param all_etas: The collection of all pseudorapidities. See transverse momentum for identical shape
        :param all_phis: The collection of all phis. See transverse momentum for identical shape

    Return:
        The x, y, and z coordinates of the 4-momentum
    '''
    all_pxs = all_pts * np.cos(all_phis)
    all_pys = all_pts * np.sin(all_phis)
    all_pzs = all_pts * np.sinh(all_etas)
    return all_pxs, all_pys, all_pzs


def calc_drs(all_etas, all_phis):
    ''' Calclates all of the different combinations of angular distances between a collection of all of the phis and etas.
    There are 10 combinations per event and does them per event for all collections of events. The distance metric being
    used is the L2 norm. 

    Args:
        :param all_etas: The collection of all pseudorapidites being analyzed. Structure is K x M x 5
                        where K is the number of collections of events and M is the number of events. There
                        are 5 jets per event.
        :param all_phis: The collection of all phis being analyzed. See pseudorapidities for  structure.
    
    Return:
        The L2 norm between the two (eta, phi) pairs'''
    num_jets = 5
    all_drs = np.array([], dtype=np.float).reshape(0, len(all_etas), int(len(all_etas[0])/5))
    first_indices, second_indices = np.triu_indices(num_jets, m=num_jets)
    for k in range(len(first_indices)):
        i = first_indices[k]
        j = second_indices[k]
        if i == j:
            continue
        drs = np.array([np.sqrt(np.power(all_etas[:,i::5] - all_etas[:,j::5], 2) + np.power(all_phis[:,i::5] - all_phis[:,j::5], 2))])
        all_drs = np.concatenate((all_drs[:], drs), axis=0)
    return all_drs


def reconstruct_higgs(all_pts, all_etas, all_phis, all_masses, all_btags):
    gan_pts = all_pts.reshape(len(all_pts), int(len(all_pts[0]) / JETS_PER_EVENT), JETS_PER_EVENT)
    gan_etas = all_etas.reshape(len(all_etas), int(len(all_etas[0]) / JETS_PER_EVENT), JETS_PER_EVENT)
    gan_phis = all_phis.reshape(len(all_phis), int(len(all_phis[0]) / JETS_PER_EVENT), JETS_PER_EVENT)
    gan_masses = all_masses.reshape(len(all_masses), int(len(all_masses[0]) / JETS_PER_EVENT), JETS_PER_EVENT)
    gan_btags = all_btags.reshape(len(all_btags), int(len(all_btags[0]) / JETS_PER_EVENT), JETS_PER_EVENT)
    
    mask = (gan_pts > 30) & (np.absolute(gan_etas) < 2.4)
    gan_pts = np.where(mask, gan_pts, np.nan)
    gan_etas = np.where(mask, gan_etas, np.nan)
    gan_phis = np.where(mask, gan_phis, np.nan)
    gan_masses = np.where(mask, gan_masses, np.nan)
    gan_btags = np.where(mask, gan_btags, np.nan)
    
    is_b = gan_btags > 0.185
    has_4bs = np.count_nonzero(is_b, axis=2) >= 4
    nEventsWith4bs = np.count_nonzero(has_4bs, axis = 1)
    #print("Number of events with 4 b-tagged jets is " + str(nEventsWith4bs) + ' out of a total of ' + str(nEvents) + ' events.')
    #print('Percentage = ' + str(nEventsWith4bs/nEvents))
    
    gan_4b_pts = [gan_pts[i,has_4bs[i]] for i in range(len(has_4bs))]
    gan_4b_etas = [gan_etas[i,has_4bs[i]] for i in range(len(has_4bs))]
    gan_4b_phis = [gan_phis[i,has_4bs[i]] for i in range(len(has_4bs))]
    gan_4b_masses = [gan_masses[i,has_4bs[i]] for i in range(len(has_4bs))]
    gan_4b_btags = [gan_btags[i,has_4bs[i]] for i in range(len(has_4bs))]
    
    lead_higgs_list = np.empty((len(has_4bs),),dtype=object)
    subl_higgs_list = np.empty((len(has_4bs),),dtype=object)
    dihiggs_list = np.empty((len(has_4bs),),dtype=object)
    delta_r_higgs_list = np.empty((len(has_4bs),),dtype=object)
    
    for i in range(len(has_4bs)):
        sorted_indices = np.array([gan_4b_btags[i][j].argsort()[::-1] for j in range(len(gan_4b_btags[i]))])
        gan_4b_pts[i] = np.array([gan_4b_pts[i][j][sorted_indices[j]][1:] for j in range(len(sorted_indices))])
        gan_4b_etas[i] = np.array([gan_4b_etas[i][j][sorted_indices[j]][1:] for j in range(len(sorted_indices))])
        gan_4b_phis[i] = np.array([gan_4b_phis[i][j][sorted_indices[j]][1:] for j in range(len(sorted_indices))])
        gan_4b_masses[i] = np.array([gan_4b_masses[i][j][sorted_indices[j]][1:] for j in range(len(sorted_indices))])
    
        lead_higgs_bosons = []
        subl_higgs_bosons = []
        delta_r_higgs = []
        dihiggses = []
        for j in range(len(gan_4b_pts[i])):
            jet1 = TLorentzVector()
            jet2 = TLorentzVector()
            jet3 = TLorentzVector()
            jet4 = TLorentzVector()
            jet1.SetPtEtaPhiM(gan_4b_pts[i][j][0], gan_4b_etas[i][j][0], gan_4b_phis[i][j][0], gan_4b_masses[i][j][0])
            jet2.SetPtEtaPhiM(gan_4b_pts[i][j][1], gan_4b_etas[i][j][1], gan_4b_phis[i][j][1], gan_4b_masses[i][j][1])
            jet3.SetPtEtaPhiM(gan_4b_pts[i][j][2], gan_4b_etas[i][j][2], gan_4b_phis[i][j][2], gan_4b_masses[i][j][2])
            jet4.SetPtEtaPhiM(gan_4b_pts[i][j][3], gan_4b_etas[i][j][3], gan_4b_phis[i][j][3], gan_4b_masses[i][j][3])

            higgs1 = TLorentzVector()
            higgs2 = TLorentzVector()
            mass_disc = 0
            jets_12_34_disc = abs((jet1 + jet2).M() - (jet3 + jet4).M())
            jets_13_24_disc = abs((jet1 + jet3).M() - (jet2 + jet4).M())
            jets_14_23_disc = abs((jet1 + jet4).M() - (jet2 + jet3).M())
            if mass_disc == 0 or jets_12_34_disc < mass_disc:
                mass_disc = jets_12_34_disc
                higgs1 = jet1 + jet2
                higgs2 = jet3 + jet4
            if jets_13_24_disc < mass_disc:
                mass_disc = jets_13_24_disc
                higgs1 = jet1 + jet3
                higgs2 = jet2 + jet4
            if jets_14_23_disc < mass_disc:
                mass_disc = jets_14_23_disc
                higgs1 = jet1 + jet4
                higgs2 = jet2 + jet3

            if higgs1.M() > higgs2.M():
                lead_higgs_bosons.append(higgs1)
                subl_higgs_bosons.append(higgs2)
            else:
                lead_higgs_bosons.append(higgs2)
                subl_higgs_bosons.append(higgs1)
            dihiggs = higgs1 + higgs2
            delta_r_higgs.append(higgs1.DeltaR(higgs2))
            dihiggses.append(dihiggs)
        
        lead_higgs_list[i] = lead_higgs_bosons
        subl_higgs_list[i] = subl_higgs_bosons
        dihiggs_list[i] = dihiggses
        delta_r_higgs_list[i] = delta_r_higgs
    return lead_higgs_list, subl_higgs_list, dihiggs_list, delta_r_higgs_list


parser = argparse.ArgumentParser(description='Processes generated output from HH-GAN')
parser.add_argument('--mcdir', type=str, default='/home/aj/CMS_Research/HH_4b/13TeV_Data/MC', help='The directory where the HH Monte Carlo is stored (default: AJ\'s directory)')
parser.add_argument('--testdir', type=str, default='.', help='The directory to find the generated output from the HH-GAN (default: \'.\')')
parser.add_argument('--plot', default=True, type=ast.literal_eval, help='Set whether to plot histograms or not (default: True)')
parser.add_argument('--epochs', nargs='*', default=None, help='Specify which epochs to generate the plots from (default: Last, middle, and 5th)')
args = parser.parse_args()

is_plotting = args.plot
plotting_epochs = args.epochs
test_path = args.testdir
mc_path = args.mcdir
print("Plotting: " + str(is_plotting))
print("The epochs to plot: " + str(plotting_epochs))
print("The test directory: " + str(test_path))
print("The MC directory: " + str(mc_path))

mc_data = DiHiggsSignalMCDataset(mc_path, download=False, generator_level=False, normalize=True)

try:
    os.makedirs(os.path.join(test_path, 'InputLevel'))
    os.makedirs(os.path.join(test_path, 'PXYZ'))
    os.makedirs(os.path.join(test_path, 'Jet_DeltaR'))
    os.makedirs(os.path.join(test_path, 'Higgs'))
except OSError:
    logging.warning("Output folders already exist. May overwrite some output files.")

all_files = os.listdir(test_path)
gen_datas = [filename for filename in all_files if filename.endswith(".npy")]
names = np.array([gen_data.split('_')[0] for gen_data in gen_datas])
names = np.array([name[:-3] + 'k' if len(name) >= 4 else '0k' for name in names]) # only grab thousands prefix: 90000 -> 90k

epochs = np.array([int(name[:-1]) for name in names])
epoch_order = epochs.argsort()
epochs = epochs[epoch_order]

plot_data_idxs = []
if plotting_epochs is None:
    plot_data_idxs = [names.tolist().index(str(epochs[len(gen_datas) - 1]) + 'k'), names.tolist().index(str(epochs[int(len(gen_datas) / 2)]) + 'k'), names.tolist().index(str(epochs[4]) + 'k')]
else:
    plot_data_idxs = [gen_datas.index(filename) for filename in gen_datas if (filename.split('_')[0] in plotting_epochs)]

gen_events = [np.load(test_path + '/' + gen_data) for gen_data in gen_datas]

generated_pts, generated_etas, generated_phis, generated_masses, generated_btags = scale_data(gen_events, mc_data)
mc_pts, mc_etas, mc_phis, mc_masses, mc_btags = scale_data([mc_data], mc_data)

pxs, pys, pzs = calc_pxyz(generated_pts, generated_etas, generated_phis)
mc_pxs, mc_pys, mc_pzs = calc_pxyz(mc_pts, mc_etas, mc_phis)    

drs = calc_drs(generated_etas, generated_phis)
mc_drs = calc_drs(mc_etas, mc_phis)

gen_lead_higgs_bosons, gen_subl_higgs_bosons, gen_dihiggs, gen_deltaR_higgs = reconstruct_higgs(generated_pts, generated_etas, generated_phis, generated_masses, generated_btags)
mc_lead_higgs_bosons, mc_subl_higgs_bosons, mc_dihiggs, mc_deltaR_higgs = reconstruct_higgs(mc_pts, mc_etas, mc_phis, mc_masses, mc_btags)

if is_plotting:
    generated_features = [generated_pts[plot_data_idxs], generated_etas[plot_data_idxs], generated_phis[plot_data_idxs], generated_masses[plot_data_idxs], generated_btags[plot_data_idxs]]
    mc_features = [mc_pts, mc_etas, mc_phis, mc_masses, mc_btags]
    plot_dists(generated_features, mc_features, FEATURE_NAME_LIST, 'GAN Generated Jet ', 'MC Jet ')

    plot_dists([pxs[plot_data_idxs], pys[plot_data_idxs], pzs[plot_data_idxs]], [mc_pxs, mc_pys, mc_pzs], MOMENTUMS, 'GAN Generated Jet ', 'MC Jet ')

    plot_dists(drs[:,plot_data_idxs], mc_drs, DELTA_RS, 'Angular Distance Between Jets ', 'Angular Distance Between Jets ')

    plot_dists([gen_deltaR_higgs[plot_data_idxs]], [mc_deltaR_higgs], DELTA_R_HIGGS, 'Angular Distance Between Higgs ', 'Angular Distance Between Higgs ')

    higgs_datas = [gen_lead_higgs_bosons[plot_data_idxs], gen_subl_higgs_bosons[plot_data_idxs], gen_dihiggs[plot_data_idxs]]
    mc_higgs_datas = [mc_lead_higgs_bosons, mc_subl_higgs_bosons, mc_dihiggs]
    higgs_titles = ['Leading Higgs ', 'Subleading Higgs ', 'Di-Higgs ']

    for i in range(len(higgs_datas)):
        higgs_data = higgs_datas[i]
        mc_higgs_data = mc_higgs_datas[i]
        
        gen_masses = [[higgs_data[j][k].M() for k in range(len(higgs_data[j]))] for j in range(len(higgs_data))]
        mc_masses = [[mc_higgs_data[j][k].M() for k in range(len(mc_higgs_data[j]))] for j in range(len(mc_higgs_data))]
        if i == 2: # plot eta for di-higgs instead of pt
            gen_etas = [[higgs_data[j][k].Eta() for k in range(len(higgs_data[j]))] for j in range(len(higgs_data))]
            mc_etas = [[mc_higgs_data[j][k].Eta() for k in range(len(mc_higgs_data[j]))] for j in range(len(mc_higgs_data))]
            plot_dists([gen_etas, gen_masses], [mc_etas, mc_masses], HIGGS_FEATURES[i], 'GAN Generated ' + higgs_titles[i], 'MC ' + higgs_titles[i])
        else: # plot pt and mass
            gen_pts = [[higgs_data[j][k].Pt() for k in range(len(higgs_data[j]))] for j in range(len(higgs_data))]
            mc_pts = [[mc_higgs_data[j][k].Pt() for k in range(len(mc_higgs_data[j]))] for j in range(len(mc_higgs_data))]
            plot_dists([gen_pts, gen_masses], [mc_pts, mc_masses], HIGGS_FEATURES[i], 'GAN Generated ' + higgs_titles[i], 'MC ' + higgs_titles[i])

# Calculate and plot all of the Kolmogorov-Smirnov distances
epochs = epochs.astype("float64")

pts_ks_values = np.array([scipy.stats.ks_2samp(generated_pts[i], mc_pts[0])[0] for i in range(len(generated_pts))], dtype="float64")[epoch_order]
etas_ks_values = np.array([scipy.stats.ks_2samp(generated_etas[i], mc_etas[0])[0] for i in range(len(generated_etas))], dtype="float64")[epoch_order]
phis_ks_values = np.array([scipy.stats.ks_2samp(generated_phis[i], mc_phis[0])[0] for i in range(len(generated_phis))], dtype="float64")[epoch_order]
masses_ks_values = np.array([scipy.stats.ks_2samp(generated_masses[i], mc_masses[0])[0] for i in range(len(generated_masses))], dtype="float64")[epoch_order]
btags_ks_values = np.array([scipy.stats.ks_2samp(generated_btags[i], mc_btags[0])[0] for i in range(len(generated_btags))], dtype="float64")[epoch_order]

c_input = TCanvas("", "", 1400, 1400)
gr_pts_ks = TGraph(len(pts_ks_values), epochs, pts_ks_values)
gr_etas_ks = TGraph(len(etas_ks_values), epochs, etas_ks_values)
gr_phis_ks = TGraph(len(phis_ks_values), epochs, phis_ks_values)
gr_masses_ks = TGraph(len(masses_ks_values), epochs, masses_ks_values)
gr_btags_ks = TGraph(len(btags_ks_values), epochs, btags_ks_values)
input_lvl_legend = ROOT.TLegend(0.35, 0.15, 0.75, 0.35)
gr_pts_ks.SetTitle('K-S Distance for b-jet\'s pt,#eta, #phi, mass, and cMVA score')
gr_pts_ks.GetXaxis().SetTitle('Epoch # (x1000)')
gr_pts_ks.GetYaxis().SetTitle('K-S Distance')
gr_pts_ks.SetMinimum(0.001)
gr_pts_ks.SetLineColor(1)
gr_etas_ks.SetLineColor(2)
gr_phis_ks.SetLineColor(3)
gr_masses_ks.SetLineColor(4)
gr_btags_ks.SetLineColor(5)
gr_pts_ks.SetLineWidth(2)
gr_etas_ks.SetLineWidth(2)
gr_phis_ks.SetLineWidth(2)
gr_masses_ks.SetLineWidth(2)
gr_btags_ks.SetLineWidth(2)
input_lvl_legend.SetFillColor(0)
input_lvl_legend.SetLineColor(0)
input_lvl_legend.AddEntry(gr_pts_ks, "p_{T} K-S")
input_lvl_legend.AddEntry(gr_etas_ks, "#eta K-S")
input_lvl_legend.AddEntry(gr_phis_ks, "#phi K-S")
input_lvl_legend.AddEntry(gr_masses_ks, "m_{jet} K-S")
input_lvl_legend.AddEntry(gr_btags_ks, "cMVA b-tag K-S")
gr_pts_ks.Draw("AC")
gr_etas_ks.Draw("C SAME")
gr_phis_ks.Draw("C SAME")
gr_masses_ks.Draw("C SAME")
gr_btags_ks.Draw("C SAME")
input_lvl_legend.Draw()
c_input.SetLogy(True)
c_input.SaveAs(test_path + "/input_level_KS.pdf")
c_input.SaveAs(test_path + "/input_level_KS.root")

pxs_ks_values = np.array([scipy.stats.ks_2samp(pxs[i], mc_pxs[0])[0] for i in range(len(pxs))], dtype="float64")[epoch_order]
pys_ks_values = np.array([scipy.stats.ks_2samp(pys[i], mc_pys[0])[0] for i in range(len(pys))], dtype="float64")[epoch_order]
pzs_ks_values = np.array([scipy.stats.ks_2samp(pzs[i], mc_pzs[0])[0] for i in range(len(pzs))], dtype="float64")[epoch_order]

c_pxyz = TCanvas("", "", 1400, 1400)
gr_pxs_ks = TGraph(len(pxs_ks_values), epochs, pxs_ks_values)
gr_pys_ks = TGraph(len(pys_ks_values), epochs, pys_ks_values)
gr_pzs_ks = TGraph(len(pzs_ks_values), epochs, pzs_ks_values)
pxyz_legend = ROOT.TLegend(0.35, 0.15, 0.75, 0.35)
gr_pxs_ks.SetTitle('K-S Distance for b-jet\'s px, py, and pz')
gr_pxs_ks.GetXaxis().SetTitle('Epoch # (x1000)')
gr_pxs_ks.GetYaxis().SetTitle('K-S Distance')
gr_pxs_ks.SetMinimum(0.001)
gr_pxs_ks.SetLineColor(1)
gr_pys_ks.SetLineColor(2)
gr_pzs_ks.SetLineColor(3)
gr_pxs_ks.SetLineWidth(2)
gr_pys_ks.SetLineWidth(2)
gr_pzs_ks.SetLineWidth(2)
pxyz_legend.SetFillColor(0)
pxyz_legend.SetLineColor(0)
pxyz_legend.AddEntry(gr_pxs_ks, "Px K-S")
pxyz_legend.AddEntry(gr_pys_ks, "Py K-S")
pxyz_legend.AddEntry(gr_pzs_ks, "Pz K-S")
gr_pxs_ks.Draw("AC")
gr_pys_ks.Draw("C SAME")
gr_pzs_ks.Draw("C SAME")
pxyz_legend.Draw()
c_pxyz.SetLogy(True)
c_pxyz.SaveAs(test_path + "/pxyz_KS.pdf")
c_pxyz.SaveAs(test_path + "/pxyz_KS.root")

dr12_ks_values = np.array([scipy.stats.ks_2samp(drs[0][i], mc_drs[0][0])[0] for i in range(len(drs[0]))], dtype="float64")[epoch_order]
dr13_ks_values = np.array([scipy.stats.ks_2samp(drs[1][i], mc_drs[1][0])[0] for i in range(len(drs[1]))], dtype="float64")[epoch_order]
dr14_ks_values = np.array([scipy.stats.ks_2samp(drs[2][i], mc_drs[2][0])[0] for i in range(len(drs[2]))], dtype="float64")[epoch_order]
dr15_ks_values = np.array([scipy.stats.ks_2samp(drs[3][i], mc_drs[3][0])[0] for i in range(len(drs[3]))], dtype="float64")[epoch_order]
dr23_ks_values = np.array([scipy.stats.ks_2samp(drs[4][i], mc_drs[4][0])[0] for i in range(len(drs[4]))], dtype="float64")[epoch_order]
dr24_ks_values = np.array([scipy.stats.ks_2samp(drs[5][i], mc_drs[5][0])[0] for i in range(len(drs[5]))], dtype="float64")[epoch_order]
dr25_ks_values = np.array([scipy.stats.ks_2samp(drs[6][i], mc_drs[6][0])[0] for i in range(len(drs[6]))], dtype="float64")[epoch_order]
dr34_ks_values = np.array([scipy.stats.ks_2samp(drs[7][i], mc_drs[7][0])[0] for i in range(len(drs[7]))], dtype="float64")[epoch_order]
dr35_ks_values = np.array([scipy.stats.ks_2samp(drs[8][i], mc_drs[8][0])[0] for i in range(len(drs[8]))], dtype="float64")[epoch_order]
dr45_ks_values = np.array([scipy.stats.ks_2samp(drs[9][i], mc_drs[9][0])[0] for i in range(len(drs[9]))], dtype="float64")[epoch_order]

c_drs = TCanvas("", "", 1400, 1400)
gr_dr12_ks = TGraph(len(dr12_ks_values), epochs, dr12_ks_values)
gr_dr13_ks = TGraph(len(dr13_ks_values), epochs, dr13_ks_values)
gr_dr14_ks = TGraph(len(dr14_ks_values), epochs, dr14_ks_values)
gr_dr15_ks = TGraph(len(dr15_ks_values), epochs, dr15_ks_values)
gr_dr23_ks = TGraph(len(dr23_ks_values), epochs, dr23_ks_values)
gr_dr24_ks = TGraph(len(dr24_ks_values), epochs, dr24_ks_values)
gr_dr25_ks = TGraph(len(dr25_ks_values), epochs, dr25_ks_values)
gr_dr34_ks = TGraph(len(dr34_ks_values), epochs, dr34_ks_values)
gr_dr35_ks = TGraph(len(dr35_ks_values), epochs, dr35_ks_values)
gr_dr45_ks = TGraph(len(dr45_ks_values), epochs, dr45_ks_values)
drs_legend = ROOT.TLegend(0.3, 0.15, 0.85, 0.4)
gr_dr12_ks.SetTitle('K-S Distance for #DeltaR between Jets')
gr_dr12_ks.GetXaxis().SetTitle('Epoch # (x1000)')
gr_dr12_ks.GetYaxis().SetTitle('K-S Distance')
gr_dr12_ks.SetMinimum(0.001)
gr_dr12_ks.SetLineColor(1)
gr_dr13_ks.SetLineColor(2)
gr_dr14_ks.SetLineColor(3)
gr_dr15_ks.SetLineColor(4)
gr_dr23_ks.SetLineColor(5)
gr_dr24_ks.SetLineColor(6)
gr_dr25_ks.SetLineColor(7)
gr_dr34_ks.SetLineColor(8)
gr_dr35_ks.SetLineColor(9)
gr_dr45_ks.SetLineColor(11)
gr_dr12_ks.SetLineWidth(2)
gr_dr13_ks.SetLineWidth(2)
gr_dr14_ks.SetLineWidth(2)
gr_dr15_ks.SetLineWidth(2)
gr_dr23_ks.SetLineWidth(2)
gr_dr24_ks.SetLineWidth(2)
gr_dr25_ks.SetLineWidth(2)
gr_dr34_ks.SetLineWidth(2)
gr_dr35_ks.SetLineWidth(2)
gr_dr45_ks.SetLineWidth(2)
drs_legend.SetFillColor(0)
drs_legend.SetLineColor(0)
drs_legend.AddEntry(gr_dr12_ks, "#DeltaR Jets 1 and 2 K-S")
drs_legend.AddEntry(gr_dr13_ks, "#DeltaR Jets 1 and 3 K-S")
drs_legend.AddEntry(gr_dr14_ks, "#DeltaR Jets 1 and 4 K-S")
drs_legend.AddEntry(gr_dr15_ks, "#DeltaR Jets 1 and 5 K-S")
drs_legend.AddEntry(gr_dr23_ks, "#DeltaR Jets 2 and 3 K-S")
drs_legend.AddEntry(gr_dr24_ks, "#DeltaR Jets 2 and 4 K-S")
drs_legend.AddEntry(gr_dr25_ks, "#DeltaR Jets 2 and 5 K-S")
drs_legend.AddEntry(gr_dr34_ks, "#DeltaR Jets 3 and 4 K-S")
drs_legend.AddEntry(gr_dr35_ks, "#DeltaR Jets 3 and 5 K-S")
drs_legend.AddEntry(gr_dr45_ks, "#DeltaR Jets 4 and 5 K-S")
gr_dr12_ks.Draw("AC")
gr_dr13_ks.Draw("C SAME")
gr_dr14_ks.Draw("C SAME")
gr_dr15_ks.Draw("C SAME")
gr_dr23_ks.Draw("C SAME")
gr_dr24_ks.Draw("C SAME")
gr_dr25_ks.Draw("C SAME")
gr_dr34_ks.Draw("C SAME")
gr_dr35_ks.Draw("C SAME")
gr_dr45_ks.Draw("C SAME")
drs_legend.Draw()
c_drs.SetLogy(True)
c_drs.SaveAs(test_path + "/drs_KS.pdf")
c_drs.SaveAs(test_path + "/drs_KS.root")

gen_lead_higgs_masses = [[gen_lead_higgs_bosons[j][k].M() for k in range(len(gen_lead_higgs_bosons[j]))] for j in range(len(gen_lead_higgs_bosons))]
gen_subl_higgs_masses = [[gen_subl_higgs_bosons[j][k].M() for k in range(len(gen_subl_higgs_bosons[j]))] for j in range(len(gen_subl_higgs_bosons))]
gen_dihiggs_masses = [[gen_dihiggs[j][k].M() for k in range(len(gen_dihiggs[j]))] for j in range(len(gen_dihiggs))]
gen_lead_higgs_pts = [[gen_lead_higgs_bosons[j][k].Pt() for k in range(len(gen_lead_higgs_bosons[j]))] for j in range(len(gen_lead_higgs_bosons))]
gen_subl_higgs_pts = [[gen_subl_higgs_bosons[j][k].Pt() for k in range(len(gen_subl_higgs_bosons[j]))] for j in range(len(gen_subl_higgs_bosons))]
gen_dihiggs_etas = [[gen_dihiggs[j][k].Eta() for k in range(len(gen_dihiggs[j]))] for j in range(len(gen_dihiggs))]

mc_lead_higgs_masses = [[mc_lead_higgs_bosons[j][k].M() for k in range(len(mc_lead_higgs_bosons[j]))] for j in range(len(mc_lead_higgs_bosons))]
mc_subl_higgs_masses = [[mc_subl_higgs_bosons[j][k].M() for k in range(len(mc_subl_higgs_bosons[j]))] for j in range(len(mc_subl_higgs_bosons))]
mc_dihiggs_masses = [[mc_dihiggs[j][k].M() for k in range(len(mc_dihiggs[j]))] for j in range(len(mc_dihiggs))]
mc_lead_higgs_pts = [[mc_lead_higgs_bosons[j][k].Pt() for k in range(len(mc_lead_higgs_bosons[j]))] for j in range(len(mc_lead_higgs_bosons))]
mc_subl_higgs_pts = [[mc_subl_higgs_bosons[j][k].Pt() for k in range(len(mc_subl_higgs_bosons[j]))] for j in range(len(mc_subl_higgs_bosons))]
mc_dihiggs_etas = [[mc_dihiggs[j][k].Eta() for k in range(len(mc_dihiggs[j]))] for j in range(len(mc_dihiggs))]

higgs_dr_ks_values = np.array([scipy.stats.ks_2samp(gen_deltaR_higgs[i], mc_deltaR_higgs[0])[0] if len(gen_deltaR_higgs[i]) > 0 else 1.0 for i in range(len(gen_deltaR_higgs))], dtype="float64")[epoch_order]
higgs_lead_m_ks_values = np.array([scipy.stats.ks_2samp(gen_lead_higgs_masses[i], mc_lead_higgs_masses[0])[0] if len(gen_deltaR_higgs[i]) > 0 else 1.0 for i in range(len(gen_lead_higgs_masses))], dtype="float64")[epoch_order]
higgs_subl_m_ks_values = np.array([scipy.stats.ks_2samp(gen_subl_higgs_masses[i], mc_subl_higgs_masses[0])[0] if len(gen_deltaR_higgs[i]) > 0 else 1.0 for i in range(len(gen_subl_higgs_masses))], dtype="float64")[epoch_order]
higgs_lead_pt_ks_values = np.array([scipy.stats.ks_2samp(gen_lead_higgs_pts[i], mc_lead_higgs_pts[0])[0] if len(gen_deltaR_higgs[i]) > 0 else 1.0 for i in range(len(gen_lead_higgs_pts))], dtype="float64")[epoch_order]
higgs_subl_pt_ks_values = np.array([scipy.stats.ks_2samp(gen_subl_higgs_pts[i], mc_subl_higgs_pts[0])[0] if len(gen_deltaR_higgs[i]) > 0 else 1.0 for i in range(len(gen_subl_higgs_pts))], dtype="float64")[epoch_order]
dihiggs_m_ks_values = np.array([scipy.stats.ks_2samp(gen_dihiggs_masses[i], mc_dihiggs_masses[0])[0] if len(gen_deltaR_higgs[i]) > 0 else 1.0 for i in range(len(gen_dihiggs_masses))], dtype="float64")[epoch_order]
dihiggs_eta_ks_values = np.array([scipy.stats.ks_2samp(gen_dihiggs_etas[i], mc_dihiggs_etas[0])[0] if len(gen_deltaR_higgs[i]) > 0 else 1.0 for i in range(len(gen_dihiggs_etas))], dtype="float64")[epoch_order]

c_higgs = TCanvas("", "", 1400, 1400)
gr_higgs_dr_ks = TGraph(len(higgs_dr_ks_values), epochs, higgs_dr_ks_values)
gr_lead_m_ks = TGraph(len(higgs_lead_m_ks_values), epochs, higgs_lead_m_ks_values)
gr_subl_m_ks = TGraph(len(higgs_subl_m_ks_values), epochs, higgs_subl_m_ks_values)
gr_lead_pt_ks = TGraph(len(higgs_lead_pt_ks_values), epochs, higgs_lead_pt_ks_values)
gr_subl_pt_ks = TGraph(len(higgs_subl_pt_ks_values), epochs, higgs_subl_pt_ks_values)
gr_HH_m_ks = TGraph(len(dihiggs_m_ks_values), epochs, dihiggs_m_ks_values)
gr_HH_eta_ks = TGraph(len(dihiggs_eta_ks_values), epochs, dihiggs_eta_ks_values)
higgs_legend = ROOT.TLegend(0.3, 0.15, 0.85, 0.4)
gr_higgs_dr_ks.SetTitle('K-S Distance for Higgs Variables')
gr_higgs_dr_ks.GetXaxis().SetTitle('Epoch # (x1000)')
gr_higgs_dr_ks.GetYaxis().SetTitle('K-S Distance')
gr_higgs_dr_ks.SetMinimum(0.001)
gr_higgs_dr_ks.SetLineColor(1)
gr_lead_m_ks.SetLineColor(2)
gr_subl_m_ks.SetLineColor(3)
gr_lead_pt_ks.SetLineColor(4)
gr_subl_pt_ks.SetLineColor(5)
gr_HH_m_ks.SetLineColor(6)
gr_HH_eta_ks.SetLineColor(7)
gr_higgs_dr_ks.SetLineWidth(2)
gr_lead_m_ks.SetLineWidth(2)
gr_subl_m_ks.SetLineWidth(2)
gr_lead_pt_ks.SetLineWidth(2)
gr_subl_pt_ks.SetLineWidth(2)
gr_HH_m_ks.SetLineWidth(2)
gr_HH_eta_ks.SetLineWidth(2)
higgs_legend.SetFillColor(0)
higgs_legend.SetLineColor(0)
higgs_legend.AddEntry(gr_higgs_dr_ks, "#DeltaR_{H1, H2} K-S")
higgs_legend.AddEntry(gr_lead_m_ks, "m_{H, lead} K-S")
higgs_legend.AddEntry(gr_subl_m_ks, "m_{H, subl} K-S")
higgs_legend.AddEntry(gr_lead_pt_ks, "pt_{lead} K-S")
higgs_legend.AddEntry(gr_subl_pt_ks, "pt_{subl} K-S")
higgs_legend.AddEntry(gr_HH_m_ks, "m_{HH} K-S")
higgs_legend.AddEntry(gr_HH_eta_ks, "#eta_{HH} K-S")
gr_higgs_dr_ks.Draw("AC")
gr_lead_m_ks.Draw("C SAME")
gr_subl_m_ks.Draw("C SAME")
gr_lead_pt_ks.Draw("C SAME")
gr_subl_pt_ks.Draw("C SAME")
gr_HH_m_ks.Draw("C SAME")
gr_HH_eta_ks.Draw("C SAME")
higgs_legend.Draw()
c_higgs.SetLogy(True)
c_higgs.SaveAs(test_path + "/higgs_KS.pdf")
c_higgs.SaveAs(test_path + "/higgs_KS.root")