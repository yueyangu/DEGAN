import time
import pandas as pd
from Conv1D_GAN import *
from utils import *
from evaluate import *


def GAN_train_test(kde_bandwidth,folder,t,uniChannel, MP_list, window_length, glr,dlr, total_epochs,v_freq):
    '''
    GAN train, val, testing (Ground truth list include ALL defects from ALL geo channels)
    '''
    time1 = time.time()
    testing_ins =['First','Second','Fourth']
    result_df = pd.DataFrame(columns = MP_list)
# train_val_test on seperate miles
    for MP in MP_list:
        train_ins = 'Third'
        val_ins = 'Fifth'
        train_filename ='Data/acc/{}_MP{}_{}_wl{}.csv'.format(uniChannel,MP,train_ins,window_length)
        X_train = pd.read_csv(train_filename)
        X_train = X_train.iloc[:,1:(window_length+1)]

        val_filename = 'Data/acc/{}_MP{}_{}_wl{}.csv'.format(uniChannel,MP,val_ins,window_length)
        X_val = pd.read_csv(val_filename)
        X_val = X_val.iloc[:,1:(window_length+1)]

        # train_val
        train_val(uniChannel,t,MP,X_train,X_val, glr,dlr, total_epochs,window_length, X_train.shape[0], noise_dim=128, ts_dim = window_length, v_freq=v_freq,foldernum=folder)           
        for test_ins in testing_ins:
#             print('start of testing on Ins {}'.format(test_ins))
            test_filename = 'Data/acc/{}_MP{}_{}_wl{}.csv'.format(uniChannel,MP,test_ins,window_length)
            X_test = pd.read_csv(test_filename)  # include index
            # save peak anomaly predictions for each MP and each ins
            result_df.at[test_ins,MP] = evaluate_D(uniChannel, kde_bandwidth,MP,X_test, window_length,folder)
            
# Evaluation
    for ins in testing_ins:
        # get ground truth defects list for this inspection
        ins_ground_truth = get_ground_truth_list(ins, MP_list)

        # concatenate all the testing miles' results to be the final prediction list
        ins_predicted_anos = []
        for MP in MP_list:
            ins_predicted_anos = [*ins_predicted_anos,*result_df.at[ins,MP]]

        # calculate evaluation scores
        if ins == testing_ins[0]:
            metrics = calculate_scores(ins_ground_truth,ins_predicted_anos)
            metrics = metrics.reshape(1,-1)
        else:
            new_metrics = calculate_scores(ins_ground_truth,ins_predicted_anos)
            new_metrics = new_metrics.reshape(1,-1)
            metrics = np.concatenate((metrics,new_metrics),axis = 0)  # append by row

# calculate average across all testing inspections
    metrics = np.concatenate((metrics,np.average(metrics,axis=0).reshape(1,-1)),axis = 0)
    print(metrics)
    time2 = time.time()
#     print('Time consumption for training on 3, validating on 5, testing on 1, 2, 4 is {}'.format(time2-time1))
    return result_df, metrics
