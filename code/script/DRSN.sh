# for TORCHSIG TRAIN
CUDA_VISIBLE_DEVICES=3 python DRSN.py -a resnet18 -e 90 --seed 114514 -l ./logs/DRSN/TORCHSIG_TEST/ -ph train\
 --hdf5_file "/home/zhangyezhuo/modulation_attack/data/torchsig_HackRF/TORCHSIG_DATASET_small.hdf5" \
 -devtrain 'device_0' 'device_1' 'device_2' 'device_3' 'device_4' 'device_5' 'device_6' \
 -devtest 'device_0' 'device_1' 'device_2' 'device_3' 'device_4' 'device_5' 'device_6' \
 -modtrain "ask-4" -modtest "ask-16" --inner_train false  --cru_name test --class_names 'TORCHSIG_device_list' --warm_iter 10 

# for POWDER TRAIN
CUDA_VISIBLE_DEVICES=3 python DRSN.py -a resnet18 -e 90 --seed 114514 -l ./logs/DRSN/POWDER_TEST/ -ph train\
 --hdf5_file "/home/zhangyezhuo/modulation_attack/data/NEU_POWDER/POWDER_DATASET.hdf5" \
 -devtrain bes browning honors meb \
 -devtest bes browning honors meb \
 --inner_train false -rectrain s1 -rectest s1 -stdtrain '4G' -stdtest '4G' -daytrain 'Day_1' -daytest 'Day_2'  --cru_name test --class_names 'POWDER_device_list' --warm_iter 10 