import numpy as np
import os 
from tqdm import tqdm
from _00_modulations import classes, mapping_dict, ask_family, pam_family, psk_family, qam_family, fsk_family, ofdm_family

def make_pulse(root, toward, c):
    if not os.path.exists(toward):
        os.makedirs(toward)
    pulses=os.listdir(root)
    pulses=[root+p for p in pulses]
    output=np.load(pulses[0])
    for i in range(1,len(pulses)):
        output=np.concatenate((output,np.load(pulses[i])))
    np.save(toward+"{}.npy".format(c), output)

if __name__ == '__main__':
    toward="E:/torchsig_mine/dataset_digital_long/"
    for c in tqdm(ask_family+pam_family+psk_family+qam_family):
        c=mapping_dict[c]
        make_pulse("E:/torchsig_mine/dataset_digital/{}/".format(c), toward=toward, c=c)
