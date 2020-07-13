import os,sys
from Bio import SeqIO
import pandas as pd
import numpy as np

"""
DataPrepare.py

@author: liwenran
"""

###################### Input #######################
if len(sys.argv)<3:
	print '[USAGE] python DataPrepare.py cell interaction_type'
	print 'For example, python DataPrepare.py Mon P-E'
	sys.exit()
CELL = sys.argv[1]
TYPE = sys.argv[2]
if TYPE == 'P-P':
	RESIZED_LEN = 1000 #promoter
elif TYPE == 'P-E':
	RESIZED_LEN = 2000 #enhancer
else:
	print '[USAGE] python DataPrepare.py cell interaction_type'
	print 'For example, python DataPrepare.py Mon P-E'
	sys.exit()


def split():
	pairs = pd.read_csv(CELL+'/'+TYPE+'/pairs.csv')
	n_sample = pairs.shape[0]
	rand_index = range(0, n_sample)
	np.random.seed(n_sample)
	np.random.shuffle(rand_index)
	n_sample_train = n_sample - n_sample // 10
	pairs_train = pairs.iloc[rand_index[:n_sample_train]]
	pairs_test = pairs.iloc[rand_index[n_sample_train:]]
	#imbalanced testing set
	pairs_test_pos = pairs_test[pairs_test['label'] == 1]
	pairs_test_neg = pairs_test[pairs_test['label'] == 0]
	num_pos = pairs_test_pos.shape[0]
	num_neg = pairs_test_neg.shape[0]
	np.random.seed(num_neg)
	rand_index = range(0, num_neg)
	pairs_test_neg = pairs_test_neg.iloc[rand_index[:num_pos*5]]
	pairs_test = pd.concat([pairs_test_pos, pairs_test_neg])
	#save
	pairs_train.to_csv(CELL+'/'+TYPE+'/pairs_train.csv', index = False)
	pairs_test.to_csv(CELL+'/'+TYPE+'/pairs_test.csv', index = False)

def resize_location(original_location, resize_len):
    original_len = int(original_location[1]) - int(original_location[0])
    len_diff = abs(resize_len - original_len)
    rand_int = np.random.randint(0, len_diff + 1)
    if resize_len < original_len: rand_int = - rand_int
    resize_start = int(original_location[0]) - rand_int
    resize_end = resize_start + resize_len
    return (str(resize_start), str(resize_end))

def augment():
	RESAMPLE_TIME = 20
	PROMOTER_LEN = 1000
	fout = open(CELL+'/'+TYPE+'/pairs_train_augment.csv','w')
	file = open(CELL+'/'+TYPE+'/pairs_train.csv')
	for line in file:
		line = line.strip().split(',')
		if line[-1] != '1':
			fout.write(','.join(line)+'\n')
			continue
		for j in range(0, RESAMPLE_TIME):
			original_location = (line[1], line[2])
			resized_location = resize_location(original_location, RESIZED_LEN)
			fout.write(','.join([line[0],resized_location[0],resized_location[1],line[3]])+',')
			original_location = (line[5], line[6])
			resized_location = resize_location(original_location, PROMOTER_LEN)
			fout.write(','.join([line[0],resized_location[0],resized_location[1],line[3]])+',1,\n')
	file.close()
	fout.close()

def one_hot(sequence_dict, chrom, start, end):
    seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 1, 0, 0],
                'C':[0, 0, 1, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 1, 0, 0],
                'c':[0, 0, 1, 0], 't':[0, 0, 0, 1]}
    temp = []
    seq = str(sequence_dict[chrom].seq[start:end])
    for c in seq:
        temp.extend(seq_dict.get(c, [0, 0, 0, 0]))
    return temp

def encoding(sequence_dict, filename):
    file = open(CELL+'/'+TYPE+'/'+filename)
    file.readline()
    seqs_1 = []
    seqs_2 = []
    label = []
    for line in file:
        line = line.strip().split(',')
        seqs_1.append(one_hot(sequence_dict, line[0], int(line[1]), int(line[2])))
        seqs_2.append(one_hot(sequence_dict, line[4], int(line[5]), int(line[6])))
        label.append(line[-1])
    if TYPE == 'P-P':
        np.savez(CELL+'/'+TYPE+'/promoter1_Seq.npz', label = np.array(label), sequence = np.array(seqs_1))
        np.savez(CELL+'/'+TYPE+'/promoter2_Seq.npz', label = np.array(label), sequence = np.array(seqs_2))
    else:
        np.savez(CELL+'/'+TYPE+'/enhancer_Seq.npz', label = np.array(label), sequence = np.array(seqs_1))
        np.savez(CELL+'/'+TYPE+'/promoter_Seq.npz', label = np.array(label), sequence = np.array(seqs_2))

def main():
	"""Split for training and testing data"""
	split()
	"""Augment training data"""
	augment()
	"""One-hot encoding"""
	sequence_dict = SeqIO.to_dict(SeqIO.parse(open('hg19.fa'), 'fasta'))
	encoding(sequence_dict, 'pairs_train_augment.csv')
	"""DNase data process"""
	#Please use the tool, openanno: http://bioinfo.au.tsinghua.edu.cn/openness/anno/
	#The input for openanno is the position of regions formatted in a '.bed' file (i.e. column 1-3 and 5-7 in CELL/TYPE/pairs.cvs)
	#To derive the input for DeepTACT, please select "Per-base pair annotation" option when using openanno
	#The output of openanno is the basepair-level DNase scores of the given promoters or enhancers in .gz file, then load and save it into .npz format as did for sequence data

"""RUN"""
main()
