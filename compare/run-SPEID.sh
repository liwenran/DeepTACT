#!/bin/bash

git clone https://github.com/ma-compbio/SPEID

#data preparation
#Note: the filepath in the code need to be updated to your own data
python SPEID/pairwise/load_data_pairs.py

#training
python SPEID/pairwise/basic_training.py
python SPEID/pairwise/frozen_model/build_model.py

#testing
python SPEID/pairwise/predict.py

