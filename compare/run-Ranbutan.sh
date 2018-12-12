#!/bin/bash

git clone https://github.com/jmschrei/rambutan 
cd rambutan
python setup.py install

#training
#Note: the filepath in the code need to be updated to your own data
python
>>from rambutan import Rambutan
>>model = Rambutan('sequences.fa', 'dnase.bedgraph')

#testing
python
>>from rambutan import Rambutan
>>y_pred = model.predict('sequences.test.fa', 'dnase.test.bedgraph')

