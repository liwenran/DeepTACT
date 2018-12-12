#!/bin/bash

git clone https://github.com/davek44/Basset.git

#data preparation
python sequences.py augment
python Basset/src/seq_hdf5.py -r -c -v num_val -t num_test sequences.fa label.txt train.h5

#training
CUDA_VISIBLE_DEVICES=0 th Basset/src/basset_train.lua -job Basset/data/models/pretrained_params.txt -cuda -cudnn -save model.th train.h5

#testing
CUDA_VISIBLE_DEVICES=0 th Basset/src/basset_test.lua -cuda -cudnn model.th test.h5 test_out

