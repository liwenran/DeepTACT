#!/usr/bin/env python
#keras version: keras-1.2.0

import sys
import os, re
import random
import datetime
import numpy as np
import hickle as hkl
from sklearn import metrics

import keras
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Reshape, Merge, Permute
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.models import load_model
from keras.engine.topology import Layer, InputSpec
from keras import initializations

"""
DeepTACT.py

Training DeepTACT for P-P/P-E interactions

@author: liwenran
"""

######################## GPU Settings #########################
gpu_use = raw_input('Use gpu: ')
gpu_cnmem = raw_input('CNMeM: ')
os.environ['THEANO_FLAGS'] = "warn.round=False,device=gpu"+gpu_use+",lib.cnmem="+gpu_cnmem


########################### Input #############################
if len(sys.argv)<3:
	print '[USAGE] python DeepTACT.py cell interaction_type num_DNase_experiments'
	print 'For example, python DeepTACT.py demo P-E 3'
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
	print '[USAGE] python DeepTACT.py cell interaction_type num_DNase_experiments'
	print 'For example, python DeepTACT.py demo P-E 3'
	sys.exit()


######################## Initialization #######################
NUM_SEQ = 4
NUM_ENSEMBL = 20


########################### Training ##########################
# Attention GRU network
class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializations.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        M = K.tanh(x)
        alpha = K.dot(M,self.W)#.dimshuffle(0,2,1)

        ai = K.exp(alpha)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return K.tanh(weighted_input.sum(axis=1))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

def model_def():
    drop_rate = 0.5 
    conv_enhancer_seq = Sequential()
    conv_enhancer_seq.add(Convolution2D(1024, NUM_SEQ, 40, activation = 'relu', border_mode = 'valid',
                                        dim_ordering = 'th', input_shape = (1, NUM_SEQ, RESIZED_LEN)))
    conv_enhancer_seq.add(MaxPooling2D(pool_size = (1, 20), border_mode = 'valid', dim_ordering = 'th'))
    conv_enhancer_seq.add(Reshape((1024, (RESIZED_LEN-40+1)/20)))
    conv_promoter_seq = Sequential()
    conv_promoter_seq.add(Convolution2D(1024, NUM_SEQ, 40, activation = 'relu', border_mode = 'valid',
                                        dim_ordering = 'th', input_shape = (1, NUM_SEQ, 1000)))
    conv_promoter_seq.add(MaxPooling2D(pool_size = (1, 20), border_mode = 'valid', dim_ordering = 'th'))
    conv_promoter_seq.add(Reshape((1024, 48)))
    merged_seq = Sequential()
    merged_seq.add(Merge([conv_enhancer_seq, conv_promoter_seq], mode = 'concat'))
    #   
    conv_enhancer_DNase = Sequential()
    conv_enhancer_DNase.add(Convolution2D(1024, NUM_REP, 40, activation = 'relu', border_mode = 'valid',
                                          dim_ordering = 'th', input_shape = (1, NUM_REP, RESIZED_LEN)))
    conv_enhancer_DNase.add(MaxPooling2D(pool_size = (1, 20), border_mode = 'valid', dim_ordering = 'th'))
    conv_enhancer_DNase.add(Reshape((1024, (RESIZED_LEN-40+1)/20)))
    conv_promoter_DNase = Sequential()
    conv_promoter_DNase.add(Convolution2D(1024, NUM_REP, 40, activation = 'relu', border_mode = 'valid',
                                          dim_ordering = 'th', input_shape = (1, NUM_REP, 1000)))
    conv_promoter_DNase.add(MaxPooling2D(pool_size = (1, 20), border_mode = 'valid', dim_ordering = 'th'))
    conv_promoter_DNase.add(Reshape((1024, 48)))
    merged_DNase = Sequential()
    merged_DNase.add(Merge([conv_enhancer_DNase, conv_promoter_DNase], mode = 'concat'))
    #   
    merged = Sequential()
    merged.add(Merge([merged_seq, merged_DNase], mode = 'concat', concat_axis = -2))
    merged.add(Permute((2, 1)))
    merged.add(BatchNormalization())
    merged.add(Dropout(drop_rate))
    merged.add(Bidirectional(LSTM(100, return_sequences = True), merge_mode = 'concat'))
    merged.add(AttLayer())
    merged.add(BatchNormalization())
    merged.add(Dropout(drop_rate))
    model = Sequential()
    model.add(merged)
    model.add(Dense(925))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

def f1(y_true, y_pred):
    TP = K.sum(K.equal(y_true, 1) & K.equal(K.round(y_pred), 1))
    FP = K.sum(K.equal(y_true, 0) & K.equal(K.round(y_pred), 1))
    FN = K.sum(K.equal(y_true, 1) & K.equal(K.round(y_pred), 0))
    TN = K.sum(K.equal(y_true, 0) & K.equal(K.round(y_pred), 0))
    P = TP / (TP + FP + K.epsilon())
    R = TP / (TP + FN + K.epsilon())
    F1 = 2 * P * R / (P + R + K.epsilon())
    return F1

def bagging(t):
    ## load data: sequence
    region1 = np.load(CELL+'/'+TYPE+'/bagData/'+filename1+'_Seq_'+str(t)+'.npz')
    region2 = np.load(CELL+'/'+TYPE+'/bagData/'+filename2+'_Seq_'+str(t)+'.npz')
    label = region1['label']
    region1_seq = region1['sequence']
    region2_seq = region2['sequence']
	
    ## load data: DNase
    region1 = np.load(CELL+'/'+TYPE+'/bagData/'+filename1+'_DNase_'+str(t)+'.npz')
    region2 = np.load(CELL+'/'+TYPE+'/bagData/'+filename2+'_DNase_'+str(t)+'.npz')
    region1_expr = region1['expr']
    region2_expr = region2['expr']

    model = model_def()
    print 'compiling...'
    model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizers.Adam(lr = 0.00001),
                  metrics = ['acc', f1])
    filename = CELL+'/'+TYPE+'/models/best_model_' + str(t) + '.h5'
    modelCheckpoint = ModelCheckpoint(filename, monitor = 'val_acc', save_best_only = True, mode = 'max')
    print 'fitting...'
    model.fit([region1_seq, region2_seq, region1_expr, region2_expr], label, nb_epoch = 40, batch_size = 100,
              validation_split = 0.1, callbacks = [modelCheckpoint])

def train():
    for t in range(NUM_ENSEMBL):
        print t
        bagging(t)


########################### Evaluation ##########################
def bag_pred(label, bag_pred, bag_score):
    vote_pred = np.zeros(bag_pred.shape[1])
    vote_score = np.zeros(bag_score.shape[1])
    for i in range(bag_pred.shape[1]):
        vote_pred[i] = stats.mode(bag_pred[:,i]).mode
        vote_score[i]= np.mean(bag_score[:,i])
    f1 = metrics.f1_score(label, vote_pred)
    auprc = metrics.average_precision_score(label, vote_score)
    return f1, auprc

def evaluate():
    region1_seq, region2_seq, region1_expr, region2_expr, label = load_test_data()#the same format as training data
    bag_pred = np.zeros((NUM_ENSEMBL,label.shape[0]))
    bag_score = np.zeros((NUM_ENSEMBL,label.shape[0]))
    for t in range(NUM_ENSEMBL):
        model.load_weights('./models/best_model_'+str(t)+'.h5')
        score = model.predict([region1_seq, region2_seq, region1_expr, region2_expr], batch_size = 100)
        bag_pred[t,:] = (score > 0.5).astype(int).reshape(-1)
        bag_score[t,:] = score.reshape(-1)
    f1, auprc = bag_pred(label, bag_pred, bag_score)
    return f1, auprc


############################ MAIN ###############################
os.system('mkdir -p '+CELL+'/'+TYPE+'/models')
train()


