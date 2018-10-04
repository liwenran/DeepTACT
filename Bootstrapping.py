#!/usr/bin/env python
import sys
import os, re
import random
import datetime
import numpy as np
import hickle as hkl
from sklearn import metrics

"""
Bootstrapping.py

@author: liwenran
"""

###################### Input #######################
if len(sys.argv)<4:
	print '[USAGE] python Bootstrapping.py cell interaction_type num_DNase_experiments'
	print 'For example, python Bootstrapping.py demo P-E 3'
	sys.exit()
CELL = sys.argv[1]
TYPE = sys.argv[2]
NUM_REP = int(sys.argv[3])
if TYPE == 'P-P':
	filename1 = 'promoter1'
	filename2 = 'promoter2'
	RESIZED_LEN = 1000 #promoter
elif TYPE == 'P-E':
	filename1 = 'enhancer'
	filename2 = 'promoter'
	RESIZED_LEN = 2000 #enhancer
else:
	print '[USAGE] python Bootstrapping.py cell interaction_type num_DNase_experiments'
	print 'For example, python Bootstrapping.py demo P-E 3'
	sys.exit()

################# Initialization ###################
NUM_SEQ = 4
NUM_ENSEMBL = 20


############################# Bootstrapping #############################
def bagging():
    os.system('mkdir -p '+CELL+'/'+TYPE+'/bagData')
    ## load data: sequence
    shape1 = (-1, 1, RESIZED_LEN, NUM_SEQ)
    shape2 = (-1, 1, 1000, NUM_SEQ)
    region1 = np.load(CELL+'/'+TYPE+'/'+filename1+'_Seq.npz')
    region2 = np.load(CELL+'/'+TYPE+'/'+filename2+'_Seq.npz')
    Tlabel = region1['label']
    Tregion1_seq = region1['sequence'].reshape(shape1).transpose(0, 1, 3, 2)
    Tregion2_seq = region2['sequence'].reshape(shape2).transpose(0, 1, 3, 2)

    ## load data: DNase
    shape1 = (-1, 1, NUM_REP, RESIZED_LEN)
    shape2 = (-1, 1, NUM_REP, 1000)
    region1 = np.load(CELL+'/'+TYPE+'/'+filename1+'_DNase.npz')
    region2 = np.load(CELL+'/'+TYPE+'/'+filename2+'_DNase.npz')
    Tregion1_expr = region1['expr'].reshape(shape1)
    Tregion2_expr = region2['expr'].reshape(shape2)

    NUM = Tlabel.shape[0]
    for t in range(0, NUM_ENSEMBL):
        print t
        """bootstrap"""
        index = [random.choice(range(NUM)) for i in range(NUM)]
        hkl.dump(index, CELL+'/'+TYPE+'/bagData/index_'+str(t)+'.hkl')
        label = Tlabel[index]
        np.savez(CELL+'/'+TYPE+'/bagData/label_'+str(t)+'.npz', label = label)
        np.savez(CELL+'/'+TYPE+'/bagData/'+filename1+'_Seq_'+str(t)+'.npz', sequence = Tregion1_seq[index], label = label)
        np.savez(CELL+'/'+TYPE+'/bagData/'+filename2+'_Seq_'+str(t)+'.npz', sequence = Tregion2_seq[index], label = label)
        np.savez(CELL+'/'+TYPE+'/bagData/'+filename1+'_DNase_'+str(t)+'.npz', expr = Tregion1_expr[index], label = label)
        np.savez(CELL+'/'+TYPE+'/bagData/'+filename2+'_DNase_'+str(t)+'.npz', expr = Tregion2_expr[index], label = label)

#MAIN
bagging()

