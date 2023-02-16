'''
Training and validating the Generative Adversarial Network on the time series and extracting
the discriminator model to be used for testing.

The file contains the following functions:

Functions 
---------
GAN_train_val:
    Training and validating the Generative Adversarial Network on the time series and extracting
    the discriminator model to be used for testing.

'''

import time
import pandas as pd
from DEGAN.Conv1D_GAN import *
from DEGAN.utils import *
from DEGAN.evaluate import *
from DEGAN.test import *
import numpy

def GAN_train_val(window_length, input_df_dic, GAN_param_dic):
    '''
    Training and validating the Generative Adversarial Network on the time series and extracting
    the discriminator model to be used for testing.

    Parameters
    ----------
    window_length: int
        Length of the window used in the sliding window method for subsequence extractions.
        
    GAN_param_dic: dict
        Dictionary containing the various parameters of the Generative Adversarial Network.
        For example, learning rate, epochs and threshold.
    
    input_df_dic: dict
        Dictionary containing the input dataset used for processing.
        For example, training, validation and testing dataset.
        
    Returns
    -------
    model: model
        Trained discriminator model used for evaluation of the testing inspection.
        
    '''    
    
    model, *_ = train_val(input_df_dic, GAN_param_dic, window_length)
    
    return model
