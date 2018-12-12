#!/bin/bash

git clone https://github.com/Dongwon-Lee/lsgkm.git

#data preparation
python sequences.py non-augment

#training
gkmtrain sequences.pos.fa sequences.neg.fa model

#testing
gkmpredict sequences.test.fa model.txt label.test.txt

