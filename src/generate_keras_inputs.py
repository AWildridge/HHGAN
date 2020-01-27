from pytorch_code.dihiggs_dataset import DiHiggsSignalMCDataset

import numpy as np

mc_data = DiHiggsSignalMCDataset('D:\CMS_Research\HH\HHGAN\data', download=False, generator_level=False, normalize=True)
np.save('NHETraining', mc_data.events_array)