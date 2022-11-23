import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
from numpy import array, linspace

def evaluate_D(channel, kde_bandwidth,MP,X_test, window_length,foldernum):
    '''
    Test D on testing set, return original prediction list 
    '''
    pred=np.zeros(X_test.shape[0])
    X_test['y_pred']=pred
    
    model_path = "trained_Discriminator/{}/{}_MP{}_discriminator.h5".format(foldernum,channel,MP)
    model = load_model(model_path,compile = False)
    X_test['y_pred']=np.round(model.predict(X_test.iloc[:,1:(window_length+1)])[:,0])  # [0,1] taken as 0, [1,0] taken as 1

    anomaly = X_test.loc[X_test['y_pred']==1]['Unnamed: 0']
    anomaly_index_list = anomaly.values.tolist()                                                    
    anomaly_index_list = [i+0.5*window_length for i in anomaly_index_list]

    num_test = 60000 # larger than the maximum index of the origianl data
    a = np.array(anomaly_index_list).reshape(-1, 1)    
    try:
        kde = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(a)
        s = linspace(0,num_test,num=num_test)
        e = kde.score_samples(s.reshape(-1,1))
        e = np.power(10,e)
        scaler = MinMaxScaler()
        e = scaler.fit_transform(e.reshape(-1,1)) 
        peaks, _ = find_peaks(e.flatten(),height=np.histogram(e,bins=21)[1][11],distance=50)  #FJK
        e = pd.DataFrame(e)
    except:
        peaks=[]

    return peaks

