import struct
import os
import numpy as np
from tqdm import tqdm
from _00_modulations import classes, mapping_dict, ask_family, pam_family, psk_family, qam_family, fsk_family, ofdm_family

if __name__ == '__main__':

    root="/home/zhangyezhuo/modulation_attack/data/torchsig_HackRF/snr_10/"
    for mode in tqdm(["ook", ]):
        mode = mapping_dict[mode]
        for device in ["device_0", "device_1", "device_2", "device_3", "device_4", "device_5", "device_6"]:
            filepath = root+'{0}/{1}'.format(device, mode)
            sample_rate = 16e6  

            # bin read
            binfile = open(filepath, 'rb') 
            size = os.path.getsize(filepath)
            data_bin = []
            data_int = []
            for i in range(int(size/8*2)):# 8bit 2channel
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
            
            # cut
            
            point = 256 # sample length
            number = 1000 # sample number
            
            toward_path = "/home/zhangyezhuo/modulation_attack/data/torchsig_HackRF/snr_10_ook/"
            path=toward_path+'/{0}/{1}_{2}/'.format(device, device, mode)
            if not os.path.exists(path):
                os.makedirs(path)
                           
            for i in tqdm(range(number), desc="{0}_{1}: ".format(device, mode)):
                tmp_re = re[int(point*i):int(point*(i+1))]
                tmp_im = im[int(point*i):int(point*(i+1))]
                np.save(path+'{0}.npy'.format(str(i).rjust(len(str(number)), "0")), np.append(tmp_re, tmp_im))
                
                