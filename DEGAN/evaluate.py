'''
Use the discriminator model to evaluate the testing inspection making use of the saved 
discriminator model. Apply Kernel Density Estimation to generate the peaks (anomalies) 
from the anomaly index list.

The file contains the following functions:

Functions 
---------
evaluate_D:
    Evaluate the predictions on the testing inspection using the discriminator model and 
    apply kernel density estimation to generate the anomalies (peaks) from the anomaly index
    list.

'''

import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
from numpy import array, linspace

def evaluate_D(post_process_dic, input_df_dic, window_length, model):
    '''
    Use the discriminator model to evaluate the predictions on the testing inspection.
    
    Parameters
    ----------
    input_df_dic: dict
        Dictionary containing the input dataset used for processing.
        For example, training, validation and testing dataset.
        
    post_process_dic: dict
        Dictionary containing the ground truth list, the parameters used for getting the 
        anomaly index list, and the tolerance range used for evaluating the model's performance.
    
    window_length: int
        Length of the sliding window used to process the time series data.
        
    model: model
        Trained discriminator model used for evaluation of the testing inspection.
        
    Returns
    -------
    KDE_scores: dataframe
        Kernel density scores of all the points in the time series.

    anomaly_index_list: list
        Indices where an anomaly has been detected by the trained model.
    
    peaks: ndarray
        Predicted anomalies generated after applying kernel density estimation on all the predictions 
        of the model.
    
    '''
    pred=np.zeros(input_df_dic["test_df"].shape[0])
    input_df_dic["test_df"]['y_pred']=pred
    
    input_df_dic["test_df"]['y_pred']=np.round(model.predict(input_df_dic["test_df"].iloc[:,1:(window_length+1)])[:,0])  
    # [0,1] taken as 0, [1,0] taken as 1

    anomaly = input_df_dic["test_df"].loc[input_df_dic["test_df"]['y_pred']==1]['Unnamed: 0']
    anomaly_index_list = anomaly.values.tolist()                                                    
    anomaly_index_list = [i+0.5*window_length for i in anomaly_index_list]

    a = np.array(anomaly_index_list).reshape(-1, 1)    
    try:
        kde = KernelDensity(kernel='gaussian', bandwidth=post_process_dic["kde_bandwidth"]).fit(a)
        s = linspace(0,input_df_dic["num_test"],num=input_df_dic["num_test"])
        KDE_scores = kde.score_samples(s.reshape(-1,1))
        KDE_scores = np.power(10,KDE_scores)
        scaler = MinMaxScaler()
        KDE_scores = scaler.fit_transform(KDE_scores.reshape(-1,1)) 
        # See explanations of the below parameters in post_process_dic in demo.ipynb
        peaks, _ = find_peaks(KDE_scores.flatten(),
                              height=np.histogram(KDE_scores,bins=post_process_dic["peak_bins"])[1][post_process_dic["peak_quant"]], distance=post_process_dic["peak_dist"])  
        KDE_scores = pd.DataFrame(KDE_scores)
    except:
        anomaly_index_list = []
        KDE_scores = pd.DataFrame()
        peaks=[]

    return KDE_scores, anomaly_index_list, peaks

