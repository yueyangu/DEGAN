import numpy as np
import pandas as pd

def Get_InsDate(ins):
   '''
   get the inspection date
   '''
   if ins == 'First':
       InsDate = '5/17/2017'
   elif ins == 'Second':
       InsDate = '7/26/2017'
   elif ins == 'Third':
       InsDate = '12/20/2017'
   elif ins == 'Fourth':
       InsDate = '2/28/2018'
   else:
       InsDate = '4/12/2018'
   return InsDate

#  def Get_InsDate(ins):
#     '''
#     get the inspection date
#     '''
#     if ins == 'First':
#         InsDate = '2017-05-17'
#     elif ins == 'Second':
#         InsDate = '2017-7-26'
#     elif ins == 'Third':
#         InsDate = '2017-12-20'
#     elif ins == 'Fourth':
#         InsDate = '2018-02-28'
#     else:
#         InsDate = '2018-04-12'
#     return InsDate


def create_y(ins):
    '''
    create testing set with labels
    no sliding windows! Just original time series
    '''
    InsDate = Get_InsDate(ins)
    label_data = pd.read_excel('Data/Label Data_2017_2018.xlsx')
    InsLabel = label_data.loc[label_data['SurveyDate']==InsDate]
    df = pd.read_csv(path + 'Data/' + ins + ' inspection.csv')
    df['label']=0
    df['level']=0
    for index, row in InsLabel.iterrows():
        # binary label
        df.loc[(df['Mile']==int(row.MP)) & (df['SyncCnt']==row.SyncCnt) & (df['SyncFt']==row.SyncFT), 'label']= 1
        # defect level
        df.loc[(df['Mile']==int(row.MP)) & (df['SyncCnt']==row.SyncCnt) & (df['SyncFt']==row.SyncFT), 'level']= row.Level
    return df

def window_stack(a, interval =1, wl=2):
    n = a.shape[0]
    return np.hstack( a[i:1+n+i-wl:interval] for i in range(0,wl) )


def df_normalize(df):
    '''zero-mean normalization of dataframe: conducted by row'''
    df_t=df.T
    for column in df_t.columns:
        df_t[column] = df_t[column] - df_t[column].mean()
    df=df_t.T
    return df

def create_X(MP_list, ins, interval, window_length, uniChannel):
    '''
    Create X_train, X_val and X_test 
    (1) Sliding window on each MP and then concatenate to avoid involving new patterns
    (2) zero-mean normalization
    '''
    ins_df = pd.read_csv('Data/' + ins + ' inspection.csv')
    for mp in MP_list:
        df_mp = ins_df.loc[ins_df['Mile']==mp, uniChannel]
        mp_index = df_mp.index[0]
        df_mp = df_mp.to_numpy().reshape(-1,1)
        df_mp = window_stack(df_mp, interval =interval, wl=window_length)
        if mp == MP_list[0]:
            df = pd.DataFrame(df_mp)
            df.index += mp_index
        else:
            df_mp = pd.DataFrame(df_mp)
            df_mp.index += mp_index
            df = df.append(df_mp)
    df = df_normalize(df) #zero-mean
    return df

def create_clean_X(MP,ins, interval, window_length, uniChannel):
    '''
    Create X_train, X_val 
    (1) Sliding window on each normal segment and then concatenate to avoid involving new patterns
    (2) zero-mean normalization
    Note: no original index info
    '''
    ins_df = pd.read_excel(path + 'Data/Railroad/' + ins + '.xlsx')
    # time series of this mp
    df_mp = ins_df.loc[ins_df['MP']==MP, uniChannel]
    # ground truth defect list of this mp
    gt_mp = get_ground_truth_list(ins, [MP])
    print('gt_mp: {}'.format(gt_mp))
    idx_gt = 0
    starting_index = df_mp.index[0]
    ending_index = df_mp.index[-1]
    df = pd.DataFrame()
    while idx_gt < len(gt_mp):
        df_seg = df_mp.loc[range(starting_index,gt_mp[idx_gt])]
        if df_seg.shape[0]>=window_length:
            df_seg = df_seg.to_numpy().reshape(-1,1)
            df_seg = window_stack(df_seg, interval =interval, wl=window_length)
            df_seg = pd.DataFrame(df_seg)
            df = pd.concat([df, df_seg])
        starting_index = gt_mp[idx_gt] # move on to the next normal segment
        idx_gt += 1 # move on to the next normal segment
    print(starting_index)   
    print(ending_index)
    df_last_seg = df_mp.loc[range(starting_index,ending_index)]
    if  df_last_seg.shape[0]>=window_length:
        df_last_seg = df_last_seg.to_numpy().reshape(-1,1)
        df_last_seg = window_stack(df_last_seg, interval =interval, wl=window_length)
        df_last_seg = pd.DataFrame(df_last_seg)
        df = pd.concat([df, df_last_seg])
        
    df = df_normalize(df) #zero-mean
    return df

