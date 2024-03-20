# MDD-SEI
Specific Emitter Identification Handling Modulation Variation with Margin Disparity Discrepancy

A project employing the **Margin Disparity Discrepancy (MDD)** method for **Domain Adaptation** in a scenario with **changing modulation schemes** within the **Specific Emitter Identification (SEI)**.

if our codes helped your reasearch, please consider citing the corresponding submission

```citatioin
@article{zhang2024specific,
        title={Specific Emitter Identification Handling Modulation Variation with Margin Disparity Discrepancy},
        author={Yezhuo Zhang and Zinan Zhou and Xuanpeng Li},
        year={2024},
        journal={arXiv preprint arXiv:2403.11531},
}
```

- SEI with modulation variation
![MDD_structure](modulation_changing.jpg)

- MDD structure for SEI
![MDD_structure](MDD_structure.jpg)

To related SEI with modulation variation works: 
- IEEE Transactions on Information Forensics and Security 2023

    [Variable-Modulation Specific Emitter Identification With Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/9956857)

- IEEE the 23rd International Conference on Communication Technology 2023

    [Few-Shot Domain Adaption-Based Specific Emitter Identification Under Varying Modulation](https://ieeexplore.ieee.org/abstract/document/10419733)

To the DA method employed in our work: 
- Proceedings of the 36th International Conference on Machine Learning 2019

    [Bridging Theory and Algorithm for Domain Adaptation](https://proceedings.mlr.press/v97/zhang19i.html?ref=https://codemonkey)

To the source code we referenced: 
- Transfer-Learning-Library

    [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library)

## Requirements

```
pip install -r requirements.txt
```

## Data preparation

The data in the article is not readily available for publication. Instead, we provide two open-source datasets that can be used for experimentation within the existing framework.

- Datasets for RF Fingerprinting on the POWDER Platform [NEU_POWDER](https://www.genesys-lab.org/powder)

- A PyTorch Signal Processing Machine Learning Toolkit [TorchSig](https://torchsig.com/)

If you choose ```NEU_POWDER```, you need to 
- download the dataset from [NEU_POWDER](https://www.genesys-lab.org/powder) 
- cd into ```data/NEU_POWDER``` and run ```POWDER_HDF5.py``` to make a ```.hdf5``` dataset.

If you choose ```TorchSig```, note that this is a simulated dataset. If you use the data with hardware impairments, you can directly create an ```.hdf5`````` dataset. If you use the ideal data, please follow the steps below.
- download the code from [TorchSig](https://torchsig.com/) and generate clear data.
- cd into ```data/torchsig_HackRF/_make_dataset_from_torchsig``` and run ```_00_modulations.py```, ```_01_from_seperate_to_continue.py``` and ```_02_add_carrier.py```.
- **transmit real signal yourself**
- generate samples from the emitted signal with ```_03_cut_from_sampled.py```.
- cd into ```data/torchsig_HackRF``` and run ```TORCHSIG_HDF5.py``` to make a ```.hdf5``` dataset.

## Training
cd into ```code/script``` and do

raw DRSN model:     
```python
bash DRSN.sh
```

DRSN with MDD:  
```python
bash MDD_DRSN.sh
```

## Contact

If you have any problem with our code or any suggestions, including discussion on SEI, please feel free to contact

- Yezhuo Zhang (zhang_yezhuo@seu.edu.cn | zhang_yezhuo@outlook.com)
- Zinan Zhou (zhouzinan919@seu.edu.cn)
- Xuanpeng Li (li_xuanpeng@seu.edu.cn)

