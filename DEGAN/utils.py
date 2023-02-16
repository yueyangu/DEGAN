'''
Helper functions that perform part of the computations for other functions. Used 
for improving the re-usability of code and making the code more modular.

The file contains the following functions:

Functions 
---------
    
count_label:
    Count the number of data points in each cluster.
    
evaluate:
    Calculate the precision, recall and F1 scores of the model.
    
calculate_scores:
    Return an array of evaluation scores under certain tolerance ranges.
    
'''

import numpy as np
import pandas as pd

def count_label(cluster_index,cluster_num):
    '''
    Count the number of data points in each cluster.
    
    Parameters
    ----------
    cluster_index: list
        Index list of the cluster that a particular data point belongs to.
        
    cluster_num: list
        The total number of clusters in the data.
        
    Returns
    -------
    count: list
        A list that returns the number of data points in each cluster ordered by it's index number,
        For example: count_label([0,1,0,2,2], 3) returns [2,1,2].
    
    '''
    
    count = np.zeros(cluster_num)
    for label in range(cluster_num):
        for i in range(len(cluster_index)):
            if (cluster_index[i] == label):
                count[label] += 1
    return count

def evaluate(defect_list,peaks_list,threshold):
    '''
    Calculate the precision, recall and F1 scores of the model.
    
    Parameters
    ----------
    defect_list: list
        The ground truth defect index list in the time series.
        
    peaks_list: list
        The anomaly index list of the defects in the time series as predicted by the model.
        
    threshold: int
        The tolerance range of the analysis.
        
    Returns
    -------
    tp_fp_dict: list
        A list that returns the number of the detected anomalies, number of the ground
        truth defects, true positives, false positives, false negatives, recall and precision
        in that order respectively.
    
    '''

    tp_fp_dict=[]
    dd=peaks_list
    tp=0
    fn=0
    fp=0

    gt_count=len(defect_list)
   
    for j in defect_list:
        dis=np.absolute(np.array(dd)-j)
        #Tracer()()
        if dis.size>0 and min(dis)<threshold:
            tp+=1
        else:
            fn+=1

    if gt_count==0:
            fp+=len(dd) # all detected peaks will be false positives
    else:
        for k in dd:
            dis=np.absolute(np.array(np.array(defect_list)-k))
            if dis.size>0 and min(dis)>threshold:
                fp+=1

    if tp+fp!=0 and tp+fn!=0:
        tp_fp_dict=[len(dd),gt_count,tp,fp,fn,round(tp/(tp+fn),2),round(tp/(tp+fp),2)]
    elif tp+fp==0:
        tp_fp_dict=[len(dd),gt_count,tp,fp,fn,round(tp/(tp+fn),2),'na']
    elif tp+fn==0:
        tp_fp_dict=[len(dd),gt_count,tp,fp,fn,'na',round(tp/(tp+fp),2)]
    else:
        tp_fp_dict=[len(dd),gt_count,tp,fp,fn,'na',round(tp/(tp+fp),2)]
    return tp_fp_dict

def calculate_scores(defect_index_list,anomaly_index_list, thresholds):
    '''
    Return an array of evaluation scores under certain tolerance ranges.
    
    Parameters
    ----------
    defect_index_list: list
        The ground truth index list of the defects in the time series.
        
    anomaly_index_list: list
        The anomaly index list of the defects in the time series as predicted by the model.
        
    thresholds: list
        Tolerance ranges to use for the evaluation (Precision, Recall and F1) of the 
        testing time series.
        
    Returns
    -------
    array: list
        The recall, precision and F1 scores of the given model under certain tolerance ranges 
        as specified in the list of thresholds.
    
    '''
    
    array = np.zeros(3*len(thresholds))
    index=0
    for i in thresholds:
        dic= evaluate(defect_index_list,anomaly_index_list,i)
        if dic[-2] == 'na':
            dic[-2] = 999 
            # To transform the 'na' results into calculable results. If this value is shown in the results, 
            # it means that thre is no anomoly predictions.
        if dic[-1] == 'na':
            dic[-1] = 999 
            # To transform the 'na' results into calculable results. If this value is shown in the results, 
            # it means that thre is no ground truth defects in the label data.
        array[index]=dic[-2]  #recall
        index+=1
        array[index]=dic[-1]  #precision
        index+=1
        if dic[-2]==0 or dic[-1]==0:  #F1
            array[index]=0
        else:
            array[index]=2.0/(1.0/dic[-2]+1.0/dic[-1])  
        index+=1
    return array