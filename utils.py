import numpy as np
import pandas as pd
from Data_processing import *

def df_normalize(df):
    '''zero-mean normalization of dataframe: conducted by row'''
    df_t=df.T
    for column in df_t.columns:
        df_t[column] = df_t[column] - df_t[column].mean()
    df=df_t.T
    return df

def count_label(y,cluster_num):
    '''
    count the number of data points in each cluster
    The clustering labels are: 0,1,2,...
    for example: count_label([0,1,0,2,2], 3) returns [2,1,2]
    '''
    count = np.zeros(cluster_num)
    for label in range(cluster_num):
        for i in range(len(y)):
            if (y[i] == label):
                count[label] += 1
    return count

def evaluate(defect_list,peaks_list,threshold):
    '''
    //------ Authorship: Dr. Farrokh Jazizadeh -----//
    
    This function is to calculate recall, precision.
    
    defect_list: a list of ground_truth defect locations
    peaks_list: a list of anomaly predictions
    threshold: the tolerance range of analysis, e.g. 50,100,150,200
    '''
    tp_fp_dict=[]
    dd=peaks_list
    tp=0
    tp_2=0  # we no longer use this
    fn=0
    fp=0

    gt_count=len(defect_list)
   
    for j in defect_list:
        dis=np.absolute(np.array(dd)-j)
#             Tracer()()
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
            if dis.size>0 and min(dis)<threshold:
                tp_2+=1

    if tp+fp!=0 and tp+fn!=0:
        tp_fp_dict=[len(dd),gt_count,tp,fp,fn,round(tp/(tp+fn),2),round(tp/(tp+fp),2)]
    elif tp+fp==0:
        tp_fp_dict=[len(dd),gt_count,tp,fp,fn,round(tp/(tp+fn),2),'na']
    elif tp+fn==0:
        tp_fp_dict=[len(dd),gt_count,tp,fp,fn,'na',round(tp/(tp+fp),2)]
    else:
        tp_fp_dict=[len(dd),gt_count,tp,fp,fn,'na',round(tp/(tp+fp),2)]
    return tp_fp_dict

def calculate_scores(defect_index_list,anomaly_index_list):

    '''
    return an array of evaluation scores under certain tolerance ranges
    such as one row of the following table:
    _____________________________________________________________________
    threshold      50      |       100     |     150      |      200
    scores    Rec  Pre  F1 |  Rec  Pre  F1 | Rec  Pre  F1 | Rec  Pre  F1 
             0.11  0.22                   ..........
    _____________________________________________________________________
    '''
    thresholds = [50,100,150,200]
    array = np.zeros(3*len(thresholds))
    index=0
    for i in thresholds:
        dic= evaluate(defect_index_list,anomaly_index_list,i)
        if dic[-2] == 'na':
            dic[-2] = 999
        if dic[-1] == 'na':
            dic[-1] = 999
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

def get_ground_truth_list(ins, MP_list):
    '''
    return the ground truth defect list for certain inspection and certain mileposts
    (ALL channels of defects are included!)
    '''
    df = pd.read_csv('Data/label data with geo acc index.csv')
    date = Get_InsDate(ins)
    for MP in MP_list:
        if MP == MP_list[0]:
            d = df[(df['SurveyDate'] == date) & (df['Mile'] == MP)]
        else:
            d = pd.concat([d, df[(df['SurveyDate'] == date) & (df['Mile']== MP)]])
    ground_truth_list = d['acc_index'].values.tolist()
    return ground_truth_list     

def get_ground_truth_list_one_channel(ins, MP_list,channel):
    '''
    return the ground truth defect list for certain inspection and certain mileposts
    (ONLY ONE specified channel of defects are included!)
    '''
    df = pd.read_csv('Data/label data with geo acc index.csv')
    date = Get_InsDate(ins)
    for MP in MP_list:
        if MP == MP_list[0]:
            d = df[(df['SurveyDate'] == date) & (df['Defect'] == channel) & (df['Mile'] == MP)]
        else:
            d = pd.concat([d, df[(df['SurveyDate'] == date) & (df['Defect'] == channel) & (df['Mile']== MP)]])
    ground_truth_list = d['acc_index'].values.tolist()
    return ground_truth_list  

