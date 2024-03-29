'''
This module contains all the functions related to the generative adversarial network
(GAN). Functions range from building the generator and the discriminator model to 
combining the two, setting their trainability, generating sample random noise, training 
and saving the trained discriminator models.

The file contains the following functions:

Functions 
---------
get_generative:
    Build the generator model.
    
get_discriminative:
    Build the discriminator model.
    
set_trainability:
    Freeze or unfreeze the weights of the model and it's layers.
    
make_gan:
    Build the generative adversarial network (GAN) model by combining
    the generator and the discriminator model.
    
sample_data_and_gen:
    Combine the real data and the fake noise generated by the generator
    to pass on to the discriminator.

sample_noise:
    Produce sample random noise for the generator.
    
train_val:
    Train the generator and discriminator, and save the model using early stopping and a threshold `t`.

'''

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
from keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

from DEGAN.utils import *

def get_generative(GAN_param_dic, G_in):
    '''
    Build the generator model.
    
    Parameters
    ----------
    G_in: tensor
        Input tensor fed to the generator model.
        
    GAN_param_dic: dict
        Dictionary containing the various parameters of the Generative Adversarial Network.
        For example, learning rate, epochs and threshold.
            
    Returns
    -------
    G: model
        The generator model.

    G_out: tensor
        Output tensor derived from the generator model.
    
    '''
    
    #################################################################################
    # Main Structure of the generator
    x = Dense(GAN_param_dic["latent_dim"],
              kernel_regularizer = tf.keras.regularizers.l1(GAN_param_dic["regularization_coeff"]))(G_in) 
    x = Activation('tanh')(x)
    G_out = Dense(GAN_param_dic["out_dim"],activation='tanh',
                  kernel_regularizer = tf.keras.regularizers.l1(GAN_param_dic["regularization_coeff"]))(x)
    #################################################################################
    
    G = Model(G_in, G_out)
    opt = Adam(learning_rate=GAN_param_dic["glr"])
    G.compile(loss='binary_crossentropy', optimizer=opt)
    return G, G_out

def get_discriminative(GAN_param_dic, D_in):
    '''
    Build the discriminator model.
    
    Parameters
    ----------
    D_in: tensor
        Input tensor fed to the discriminator model.
        
    GAN_param_dic: dict
        Dictionary containing the various parameters of the Generative Adversarial Network.
        For example, learning rate, epochs and threshold.
    
    Returns
    -------
    D: model
        The discriminator model.

    D_out: tensor
        Output tensor derived from the discriminator model.
    
    '''
    
    #################################################################################
    # Main Structure of the discriminator
    x = Reshape((-1, 1))(D_in)
    x = Conv1D(GAN_param_dic["n_channels"], GAN_param_dic["conv_sz"], activation='relu',
               kernel_regularizer=tf.keras.regularizers.l1(GAN_param_dic["regularization_coeff"]))(x)
    x = Dropout(GAN_param_dic["drate"])(x)
    x = Flatten()(x)
    x = Dense(GAN_param_dic["n_channels"],
              kernel_regularizer=tf.keras.regularizers.l1(GAN_param_dic["regularization_coeff"]))(x)
    D_out = Dense(2, activation='sigmoid',
                  kernel_regularizer=tf.keras.regularizers.l1(GAN_param_dic["regularization_coeff"]))(x)  
    #################################################################################
    
    D = Model(D_in, D_out)
    dopt = Adam(learning_rate=GAN_param_dic["dlr"])
    D.compile(loss='binary_crossentropy', optimizer=dopt)
    return D, D_out

def set_trainability(model, trainable=False):
    '''
    Freeze or unfreeze the weights of the model and it's layers.
    
    Parameters
    ----------
    model: model
        Model whose weights are to be frozen/ unfrozen.
        
    trainable: boolean, optional
        Indicate whether the weights are to be frozen (False) or not (True) (default value is False).
            
    '''
    
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def make_gan(GAN_param_dic, GAN_in, G, D):
    '''
    Build the generative adversarial network (GAN) model by combining
    the generator and the discriminator model.
    
    Parameters
    ----------
    GAN_in: tensor
        Input tensor fed to the Generative Adversarial Network model.
        
    G: model
        The generator model.
         
    D: model
        The discriminator model.
        
    GAN_param_dic: dict
        Dictionary containing the various parameters of the Generative Adversarial Network.
        For example, learning rate, epochs and threshold.
           
    Returns
    -------
    GAN: model
        The generative adversarial network model consisting of the generator
        and the discriminator.

    GAN_out: tensor
        Output tensor derived from the Generative Adversarial Network model.
    
    '''
    
    set_trainability(D, False)
    x = G(GAN_in)
    GAN_out = D(x)
    GAN = Model(GAN_in, GAN_out)
    GAN.compile(loss='binary_crossentropy', optimizer=G.optimizer)
    return GAN, GAN_out

def sample_data_and_gen(GAN_param_dic, input_df_dic, G):
    '''
    Combine the real data and the fake data generated from noise by the generator,
    and shuffle after merging.
    
    Parameters
    ----------
    G: model
        The generator model.
     
    GAN_param_dic: dict
        Dictionary containing the various parameters of the Generative Adversarial Network.
        For example, learning rate, epochs and threshold.
    
    input_df_dic: dict
        Dictionary containing the input dataset used for processing.
        For example, training, validation and testing dataset.
        
    Returns
    -------
    X_data: ndarray
        Combination of fake and real data consisting of `num_input` + `n_samples` samples 
        some of which are generated from noise by the generator and  some of which are
        real training data.

    y_target: ndarray
        Dependent or target variable array which shows whether the sample is real or fake
        For example [0, 1] signals fake data whereas [1, 0] signals real data.
    
    '''
    XT = input_df_dic["train_df"]
    XN = G.predict(np.random.normal(0, 1, size=[input_df_dic["train_df"].shape[0], GAN_param_dic["noise_dim"]]))
    X_data = np.concatenate((XT, XN))
    y_target = np.zeros((input_df_dic["train_df"].shape[0]+input_df_dic["train_df"].shape[0], 2))
    y_target[:input_df_dic["train_df"].shape[0],1] = 1      # number of training samples
    y_target[input_df_dic["train_df"].shape[0]:,0] = 1   
    X_data,y_target = shuffle(X_data,y_target)
    return X_data, y_target

def sample_noise(GAN_param_dic, input_df_dic):
    '''
    Build the generator model.
    
    Parameters
    ----------
    GAN_param_dic: dict
        Dictionary containing the various parameters of the Generative Adversarial Network.
        For example, learning rate, epochs and threshold.
    
    input_df_dic: dict
        Dictionary containing the input dataset used for processing.
        For example, training, validation and testing dataset.
        
    Returns
    -------
    X_noise: ndarray
        Fake data consisting of `n_samples` samples generated from the uniform distribution .

    y_target: ndarray
        Dependent or target variable array which shows whether the sample is real or fake
        For example [0, 1] signals fake data whereas [1, 0] signals real data.
    
    '''
    
    X_noise = np.random.uniform(0, 1, size=[input_df_dic["train_df"].shape[0], GAN_param_dic["noise_dim"]])
    y_target = np.zeros((input_df_dic["train_df"].shape[0], 2))
    y_target[:,1] = 1
    return X_noise, y_target

def train_val(input_df_dic, GAN_param_dic, window_length, verbose=True, showStructure=False):   
    '''
    Build the generator model.
    
    Parameters
    ----------

    GAN_param_dic: dict
        Dictionary containing the various parameters of the Generative Adversarial Network.
        For example, learning rate, epochs and threshold.
    
    input_df_dic: dict
        Dictionary containing the input dataset used for processing.
        For example, training, validation and testing dataset.
        
    window_length: int
        Length of the window used for the time series.
    
    val_criterion: string 
        Validation criterion, can be "abosolute" (default value) or "elbow".
        "absolute" uses absolute value as threshold, i.e., GAN_param_dic["ano_thr"] is set as the absolute anomaly number threshold on validation set, e.g., 10.
        "elbow" uses elbow method to decide stopping, i.e.,  GAN_param_dic["ano_thr"] is set as a relative error threshold, e.g., 0.01.
        
    verbose: boolean
        Used for showing the progress or logging the training process (default value is True).
        
    Returns
    -------
    D: model
        Trained discriminator model used for evaluation of the testing inspection.
        
    d_loss: float
        The discriminator loss.

    g_loss: float
        The generator loss.
         
    num_ano_list: list
        List of all anomalies detected by the model after val_freq epochs.
    
    '''
    
    G_in = Input(shape=[GAN_param_dic["noise_dim"]])
    G, G_out = get_generative(GAN_param_dic, G_in)
    D_in = Input(shape=[GAN_param_dic["out_dim"]])
    D, D_out = get_discriminative(GAN_param_dic, D_in)
    GAN_in =  Input([GAN_param_dic["noise_dim"]])   
    GAN, GAN_out = make_gan(GAN_param_dic, GAN_in, G, D)
    
    # Can be used to show the model structure
    if showStructure:
        G.summary()
        D.summary()
        GAN.summary()

    d_loss = []
    g_loss = []
    num_ano_list =[]
    e_range = range(GAN_param_dic["total_epochs"])
    
    if verbose:
        e_range = tqdm(e_range)
 
    for epoch in e_range:
        X_data, y_target = sample_data_and_gen(GAN_param_dic, input_df_dic, G)
        set_trainability(D, True)
        d_loss.append(D.train_on_batch(X_data, y_target))

        X_noise, y_target = sample_noise(GAN_param_dic, input_df_dic)
        set_trainability(D, False)
        g_loss.append(GAN.train_on_batch(X_noise, y_target))
     
        if verbose and (epoch + 1) % GAN_param_dic["val_freq"] == 0:
            y_pred =np.round(D.predict(input_df_dic["val_df"].iloc[:,:window_length])[:,0])
            num_anomaly=count_label(y_pred,2)[1]  # the total number of 1 in X_test['pred']
            num_ano_list.append(num_anomaly)
            if GAN_param_dic['val_criterion'] ==  'absolute':
                if num_anomaly < GAN_param_dic["ano_thr"]:
                    print('model saved, best epoch = {}'.format(epoch+1))
                    D.save("trained_Discriminator/{}/trainedDiscriminator.h5".format(input_df_dic["folder_num"]))
                    break
            
            if GAN_param_dic['val_criterion'] == 'elbow':
                if len(num_ano_list)>1:
                    r_err = abs((num_ano_list[-1] - num_ano_list[-2])*1.00/ num_ano_list[-2])
                    if r_err< GAN_param_dic["ano_thr"]:
                        print('model saved, best epoch = {}'.format(epoch+1))
                        D.save("trained_Discriminator/{}/trainedDiscriminator.h5".format(input_df_dic["folder_num"]))
                        break
    return D, d_loss, g_loss, num_ano_list
