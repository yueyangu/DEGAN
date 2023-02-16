'''
Testing the time series data using the trained discriminator from the generative adversarial network, 
and calculating the average metrics (Precision, Recall, F1) across the testing inspection. 

The file contains the following functions:

Functions 
---------
GAN_test:
    Testing the time series data using the trained discriminator from the generative adversarial network, 
    and calculating the average metrics (Precision, Recall, F1) across the testing inspection. 

'''

import time
import pandas as pd
from DEGAN.utils import *
from DEGAN.evaluate import *

def GAN_test(window_length, input_file_dic, post_process_dic, model):
    '''
    Testing the time series data using the trained discriminator from the generative adversarial network, 
    and calculating the average metrics (Precision, Recall, F1) across the testing inspection. 

    Parameters
    ----------
    window_length: int
        Length of the window used in the sliding window method for subsequence extractions.
        
    input_df_dic: dict
        Dictionary containing the input dataset used for processing.
        For example, training, validation and testing dataset.
        
    post_process_dic: dict
        Dictionary containing the ground truth list, the parameters used for getting the 
        anomaly index list, and the tolerance range used for evaluating the model's performance.
        
    model: model
        Trained discriminator model used for evaluation of the testing inspection.
        
    Returns
    -------
    KDE_scores: dataframe
        Kernel density estimation scores of all the indices in the time series.

    anomaly_index_list: list
        Index list of the anomalies detected by the trained model.
    
    predictedAnomalies: ndarray
        Predicted anomalies generated after applying kernel density estimation on all the predictions 
        of the model.
        
    metrics: ndarray
        Evaluation (Precision, Recall and F1) scores for the testing time series using the specified 
        tolerance ranges. 
    
    '''    
    
    KDE_scores, anomaly_index_list, predictedAnomalies = evaluate_D(post_process_dic, input_file_dic, window_length, model)
            
    metrics = calculate_scores(post_process_dic["ground_truth_list"], predictedAnomalies, post_process_dic["tolerance"])
    metrics = metrics.reshape(1,-1)

    print(metrics)
    return KDE_scores, anomaly_index_list, predictedAnomalies, metrics
