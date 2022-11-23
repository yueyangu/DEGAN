import tensorflow as tf
import os
import random
import time
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from keras.models import Model
from keras.layers import Input, Reshape
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.utils import to_time_series_dataset

from utils import *

def get_generative(G_in, out_dim=500, glr=6e-3):
    '''
    build Generator
    '''
    x = Dense(256,kernel_regularizer=tf.keras.regularizers.l1(0.0001))(G_in)
    x = Activation('tanh')(x)
    G_out = Dense(out_dim, activation='tanh',kernel_regularizer=tf.keras.regularizers.l1(0.0001))(x)
    G = Model(G_in, G_out)
    opt = Adam(learning_rate=glr)
    G.compile(loss='binary_crossentropy', optimizer=opt)
    return G, G_out

def get_discriminative(D_in, dlr=5e-4, drate=.25, n_channels=16, conv_sz=5, leak=.2):
    '''
    build Discriminator
    '''
    x = Reshape((-1, 1))(D_in)
    x = Conv1D(n_channels, conv_sz, activation='relu',kernel_regularizer=tf.keras.regularizers.l1(0.0001))(x)
    x = Dropout(drate)(x)
    x = Flatten()(x)
    x = Dense(n_channels,kernel_regularizer=tf.keras.regularizers.l1(0.0001))(x)
    D_out = Dense(2, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l1(0.0001))(x)   #################################################################################
    D = Model(D_in, D_out)
    dopt = Adam(learning_rate=dlr)
    D.compile(loss='binary_crossentropy', optimizer=dopt)
    return D, D_out

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
        
def make_gan(GAN_in, G, D):
    '''
    build GAN
    '''
    set_trainability(D, False)
    # print(GAN_in.dtype)
    # print(GAN_in.shape)
    x = G(GAN_in)
    GAN_out = D(x)
    GAN = Model(GAN_in, GAN_out)
    GAN.compile(loss='binary_crossentropy', optimizer=G.optimizer)
    return GAN, GAN_out

# fixed noise
def sample_data_and_gen(G,fixed_noise,X_train,num_input, noise_dim=561, n_samples=10000):
    '''
    combine the training data and fake data to feed into D
    '''
    XT = X_train
    XN = G.predict(fixed_noise)
    X = np.concatenate((XT, XN))
    y = np.zeros((num_input+n_samples, 2))
    y[:num_input,1] = 1      # number of training samples
    y[num_input:,0] = 1   ###############################################################
    X,y = shuffle(X,y)
    return X, y

def sample_noise(G, noise_dim=561, n_samples=10000):
    '''
    sample random noise for G
    '''
    X = np.random.uniform(0, 1, size=[n_samples, noise_dim])
    y = np.zeros((n_samples, 2))
    y[:,1] = 1
    return X, y

def train_val(channel,t,MP, X_train,X_val, glr,dlr, epochs, window_length, n_samples=10000,noise_dim=50, ts_dim=500, v_freq=50,foldernum=1,verbose=True):    
    G_in = Input(shape=[noise_dim])
    G, G_out = get_generative(G_in, out_dim=ts_dim, glr=glr)
    D_in = Input(shape=[ts_dim])
    D, D_out = get_discriminative(D_in,dlr=dlr)
    GAN_in =  Input([noise_dim])   
    GAN, GAN_out = make_gan(GAN_in, G, D)
#     can be used to show the model structure
#     G.summary()
#     D.summary()
#     GAN.summary()
    
    d_loss = []
    g_loss = []
    num_ano_list =[]
    e_range = range(epochs)
    
    #####  fixed noise  (Gaussian)
    num_input = X_train.shape[0]
    fixed_noise = np.random.normal(0, 1, size=[num_input, noise_dim])

    if verbose:
        e_range = tqdm(e_range)
 
    for epoch in e_range:
#         noise = np.random.normal(0, 1, size=[num_input, noise_dim])
        X, y = sample_data_and_gen(G,fixed_noise,X_train,num_input= num_input, n_samples=n_samples, noise_dim=noise_dim)
        set_trainability(D, True)
        d_loss.append(D.train_on_batch(X, y))

        X, y = sample_noise(G, n_samples=n_samples, noise_dim=noise_dim)
        set_trainability(D, False)
        g_loss.append(GAN.train_on_batch(X, y))
     
        if verbose and (epoch + 1) % v_freq == 0:
            y_pred =np.round(D.predict(X_val.iloc[:,:window_length])[:,0])
            num_anomaly=count_label(y_pred,2)[1]  # the total number of 1 in X_test['pred']
#             print('Epoch {} - {}'.format(epoch+1, num_anomaly))
            num_ano_list.append(num_anomaly)
#             D.save("Cluster_Discriminator/{}/epoch{}_discriminator_{}.h5".format(foldernum,epoch,cluster))
            if num_anomaly < t:
                print('model saved, best epoch = {}'.format(epoch+1))
                D.save("trained_Discriminator/{}/{}_MP{}_discriminator.h5".format(foldernum,channel,MP))
                break
    return d_loss, g_loss, num_ano_list


