from tqdm import tqdm
import h5py
import numpy as np
import struct 
import os

modulation_list = ["qam-16", "qam-32", "qam-64", "qam-256", "qam-1024", 
                 "qam_cross-32", "qam_cross-128", "qam_cross-512", 
                 "psk-2", "psk-4", "psk-8", "psk-16", "psk-32", "psk-64", 
                 "pam-4", "pam-8", "pam-16", "pam-32", "pam-64",
                 "ook", "ask-4", "ask-8", "ask-16", "ask-32", "ask-64",]
device_list = ["device_0", "device_1", "device_2", "device_3", "device_4", "device_5", "device_6",]
device_mapping = {"device_0": 0, "device_1": 1, "device_2": 2, "device_3": 3, "device_4": 4, "device_5": 5, "device_6": 6, }
name_list = [modulation+"_"+device for modulation in modulation_list for device in device_list]

modulation_list_example = ["qam-16", "qam-64", "psk-2", "psk-4", "pam-4", "pam-8", "ook", "ask-4"]


if __name__ == '__main__':

    dataset_name = "TORCHSIG_DATASET_small"
    pickle_dict = dict()
    # for name in tqdm(name_list):
    for device in tqdm(device_list, desc="device"):
        for modulation in tqdm(modulation_list, desc="modulation"):
        # for modulation in tqdm(modulation_list_example, desc="modulation"):
            '''
            Hyper settings
            '''
            name = modulation+"_"+device
            root = "/home/zhangyezhuo/modulation_attack/data/torchsig_HackRF/snr_10/"
            filepath = root+'{0}/{1}'.format(device, modulation)
            sample_rate = 16e6
                
            point_per_sample = 256
            num_samples = 2000
            
            save_path = '/home/zhangyezhuo/modulation_attack/data/torchsig_HackRF/'
            save_name = dataset_name+'.hdf5'

            '''
            Data loading
            '''
            # bin read
            binfile = open(filepath, 'rb')  
            size = os.path.getsize(filepath) 
            data_bin = []
            data_int = []

            for i in range(int(size/8*2)):# 8bit 2route
                data = binfile.read(4)  # float=4×8bit
                num = struct.unpack('f', data)
                data_bin.append(data)
                data_int.append(num[0])
            binfile.close()

            # IQ divide
            re = np.array(data_int[0::2])
            im = np.array(data_int[1::2])
            re = re / np.abs(np.max(re))  # normalize
            im = im / np.abs(np.max(im))
            sig = re +1j*im
            
            data = list()       
              
            if not (num_samples is None):
                for i in range(num_samples):
                    tmp_sig = sig[int(point_per_sample*i):int(point_per_sample*(i+1))]
                    data.append(tmp_sig)
            else:
                for i in range(int(len(sig)/point_per_sample)):
                    tmp_sig = sig[int(point_per_sample*i):int(point_per_sample*(i+1))]
                    data.append(tmp_sig)

            '''
            Write a hdf5 file
            '''
            with h5py.File(save_path+save_name, 'a') as f:
                f[name] = data