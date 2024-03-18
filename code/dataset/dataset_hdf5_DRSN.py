import os
import numpy as np
from torch.utils.data import Dataset
import h5py

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from data.NEU_POWDER.POWDER_HDF5 import device_mapping as device_mapping_powder
from data.torchsig_HackRF.TORCHSIG_HDF5 import device_mapping as device_mapping_torchsig

class POWDER_Dataset_HDF5(Dataset):
    def __init__(self, hdf5_file, standards = ["4G", ], days = ["Day_1", ], devices = ["bes", ], recordings = ["s1", ]):
        self.file_path = hdf5_file
        self.hdf5_file = h5py.File(self.file_path, 'r')
        self.standards = standards
        self.days = days
        self.devices = devices
        self.recordings = recordings
        self.data, self.length = [], []
        for standard in standards:
            for day in days:
                for device in devices:
                    for recording in recordings: 
                        name = "{}_{}_{}_{}".format(standard, day, device, recording)
                        self.data.append((self.hdf5_file[name], standard, day, device, recording))
                        self.length.append(len(self.hdf5_file[name]))
                        
    def DataDealing(self, sig):
        re, im = np.real(sig), np.imag(sig)
        return np.array([re, im])
              
    def __len__(self):
        return sum(self.length)
        
    def __getitem__(self, index):
        for i, l in enumerate(self.length):
            temp = index - l
            if temp <=-1:
                part = i
                break
            else:index = temp
        sig_pack, standard, day, device, recording = self.data[part]
        sig = sig_pack[index]
        return self.DataDealing(sig), device_mapping_powder[device]

class TORCHSIG_Dataset_HDF5(Dataset):
    def __init__(self, hdf5_file, modulations = ["ook", ], devices = ["device_0", ]):
        self.file_path = hdf5_file
        self.hdf5_file = h5py.File(self.file_path, 'r')
        self.modulations = modulations
        self.devices = devices
        self.data, self.length = [], []
        for modulation in modulations:
            for device in devices:
                name = "{}_{}".format(modulation, device, )
                self.data.append((self.hdf5_file[name], modulation, device, ))
                self.length.append(len(self.hdf5_file[name]))
                        
    def DataDealing(self, sig):
        re, im = np.real(sig), np.imag(sig)
        return np.array([re, im])
              
    def __len__(self):
        return sum(self.length)
        
    def __getitem__(self, index):
        for i, l in enumerate(self.length):
            temp = index - l
            if temp <=-1:
                part = i
                break
            else:index = temp
        sig_pack, modulation, device = self.data[part]
        sig = sig_pack[index]
        return self.DataDealing(sig), device_mapping_torchsig[device]

'''
这个是将列表转化为torch dataset的工具，一般不使用，放着就行
'''
class MyDataset(Dataset):
    def __init__(self, data):
        self.data=data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data[idx][0].to("cuda:0")
        label = self.data[idx][1].to("cuda:0")
        return feature, label