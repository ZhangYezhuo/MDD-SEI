from tqdm import tqdm
import h5py
import numpy as np
import json

def json_to_dict(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        return "File not found"
    except json.JSONDecodeError:
        return "JSON analysis error"

name_list = ["4G_Day_1_bes_s1", "4G_Day_1_bes_s2",  "4G_Day_1_bes_s3", "4G_Day_1_bes_s4", "4G_Day_1_bes_s5", 
             "4G_Day_1_browning_s1", "4G_Day_1_browning_s2", "4G_Day_1_browning_s3", "4G_Day_1_browning_s4", "4G_Day_1_browning_s5",
             "4G_Day_1_honors_s1", "4G_Day_1_honors_s2", "4G_Day_1_honors_s3", "4G_Day_1_honors_s4", "4G_Day_1_honors_s5",
             "4G_Day_1_meb_s1", "4G_Day_1_meb_s2", "4G_Day_1_meb_s3", "4G_Day_1_meb_s4", "4G_Day_1_meb_s5",
             "4G_Day_2_bes_s1", "4G_Day_2_bes_s2", "4G_Day_2_bes_s3", "4G_Day_2_bes_s4", "4G_Day_2_bes_s5",
             "4G_Day_2_browning_s1", "4G_Day_2_browning_s2", "4G_Day_2_browning_s3", "4G_Day_2_browning_s4", "4G_Day_2_browning_s5",
             "4G_Day_2_honors_s1", "4G_Day_2_honors_s2", "4G_Day_2_honors_s3", "4G_Day_2_honors_s4", "4G_Day_2_honors_s5",
             "4G_Day_2_meb_s1", "4G_Day_2_meb_s2", "4G_Day_2_meb_s3", "4G_Day_2_meb_s4", "4G_Day_2_meb_s5", 
             
             "5G_Day_1_bes_s1", "5G_Day_1_bes_s2",  "5G_Day_1_bes_s3", "5G_Day_1_bes_s4", "5G_Day_1_bes_s5", 
             "5G_Day_1_browning_s1", "5G_Day_1_browning_s2", "5G_Day_1_browning_s3", "5G_Day_1_browning_s4", "5G_Day_1_browning_s5",
             "5G_Day_1_honors_s1", "5G_Day_1_honors_s2", "5G_Day_1_honors_s3", "5G_Day_1_honors_s4", "5G_Day_1_honors_s5",
             "5G_Day_1_meb_s1", "5G_Day_1_meb_s2", "5G_Day_1_meb_s3", "5G_Day_1_meb_s4", "5G_Day_1_meb_s5",
             "5G_Day_2_bes_s1", "5G_Day_2_bes_s2", "5G_Day_2_bes_s3", "5G_Day_2_bes_s4", "5G_Day_2_bes_s5",
             "5G_Day_2_browning_s1", "5G_Day_2_browning_s2", "5G_Day_2_browning_s3", "5G_Day_2_browning_s4", "5G_Day_2_browning_s5",
             "5G_Day_2_honors_s1", "5G_Day_2_honors_s2", "5G_Day_2_honors_s3", "5G_Day_2_honors_s4", "5G_Day_2_honors_s5",
             "5G_Day_2_meb_s1", "5G_Day_2_meb_s2", "5G_Day_2_meb_s3", "5G_Day_2_meb_s4", "5G_Day_2_meb_s5",

             "Wifi_Day_1_bes_s1", "Wifi_Day_1_bes_s2",  "Wifi_Day_1_bes_s3", "Wifi_Day_1_bes_s4", "Wifi_Day_1_bes_s5", 
             "Wifi_Day_1_browning_s1", "Wifi_Day_1_browning_s2", "Wifi_Day_1_browning_s3", "Wifi_Day_1_browning_s4", "Wifi_Day_1_browning_s5",
             "Wifi_Day_1_honors_s1", "Wifi_Day_1_honors_s2", "Wifi_Day_1_honors_s3", "Wifi_Day_1_honors_s4", "Wifi_Day_1_honors_s5",
             "Wifi_Day_1_meb_s1", "Wifi_Day_1_meb_s2", "Wifi_Day_1_meb_s3", "Wifi_Day_1_meb_s4", "Wifi_Day_1_meb_s5",
             "Wifi_Day_2_bes_s1", "Wifi_Day_2_bes_s2", "Wifi_Day_2_bes_s3", "Wifi_Day_2_bes_s4", "Wifi_Day_2_bes_s5",
             "Wifi_Day_2_browning_s1", "Wifi_Day_2_browning_s2", "Wifi_Day_2_browning_s3", "Wifi_Day_2_browning_s4", "Wifi_Day_2_browning_s5",
             "Wifi_Day_2_honors_s1", "Wifi_Day_2_honors_s2", "Wifi_Day_2_honors_s3", "Wifi_Day_2_honors_s4", "Wifi_Day_2_honors_s5",
             "Wifi_Day_2_meb_s1", "Wifi_Day_2_meb_s2", "Wifi_Day_2_meb_s3", "Wifi_Day_2_meb_s4", "Wifi_Day_2_meb_s5",
    ]

standard_list = ["4G", "5G", "Wifi"]
day_list = ["Day_1", "Day_2"]
device_list = ["bes", "browning", "honors", "meb"]
recording_list = ["s1", "s2", "s3", "s4", "s5"]

device_mapping = {"bes": 0, "browning": 1, "honors": 2, "meb": 3}

if __name__ == '__main__':

    dataset_name = "POWDER_DATASET"
    pickle_dict = dict()
    for name in tqdm(name_list):
        '''
        Hyper settings
        '''
        name = name
        data_path = 'D:/NEU-POWDER/GlobecomPOWDER/{0}.bin'.format(name)
        metadata_path = 'D:/NEU-POWDER/GlobecomPOWDER/{0}.json'.format(name)
        metadata = json_to_dict(metadata_path)
        save_path = 'D:/NEU-POWDER/'
        save_name = dataset_name+'.hdf5'

        point_per_sample = 512
        num_samples = None

        '''
        Data loading
        '''
        data_complex = np.fromfile(data_path, dtype=np.complex64)
        data = list()
         
        if  not (num_samples is None):
            for i in range(num_samples):
                data_in = data_complex[i*point_per_sample:(i+1)*point_per_sample]
                data.append(data_in)
        else:
            for i in range(int(len(data_complex)/point_per_sample)):
                data_in = data_complex[i*point_per_sample:(i+1)*point_per_sample]
                data.append(data_in)

        '''
        Write a hdf5 file
        '''
        with h5py.File(save_path+save_name, 'a') as f:
            f[name] = data  