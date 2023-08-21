import json
import random
import numpy as np
import tqdm
import pandas as pd
import time

from sklearn.metrics import roc_auc_score,accuracy_score,mean_squared_error,precision_score,recall_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

import clf_models
import warnings
warnings.filterwarnings('ignore')
import config
exp_list = config.exp_list
date=time.strftime("%Y-%m-%d",time.localtime(time.time()))

seeds = [1,2,3,4,5]
model_random_state=99

# prepare data
# data_df = pd.read_csv('setting1_data.csv')

# max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
# for feature in ['unique_iid','unique_uid']:
#     data_df[feature+'_norm'] = data_df[[feature]].apply(max_min_scaler)

# rating = data_df['immersion']

# data_df = data_df.drop([data_df.keys()[0],'unique_iid','unique_uid','immersion'], axis=1)
# attrlist = data_df.columns.tolist()
# print(attrlist)
# data_np = data_df.values
# print(data_np)
# np.save('setting1_data.npy',data_np)
# np.save('setting1_rating.npy',np.array(rating))


setting=1 #1,2,3

# load data
rating = np.load('setting1_rating.npy')
if setting==1:
    data= np.load('setting1_noEEG_data.npy') # 'EEG_data.npy', 
elif setting==2:
     data= np.load('setting2_EEG_data.npy')
elif setting==3:
    data1=np.load('setting1_noEEG_data.npy')
    data2= np.load('setting2_EEG_data.npy')
    data = np.hstack((data1, data2))
else:
    raise ValueError('setting must be 1, 2 or 3')
print(data.shape)

print(data.shape)
# split data (get index)
N_split = 5
N_times = 5
train_all_index = []
test_all_index = []
for time in range(N_times):
    train_all_index.append([])
    test_all_index.append([])
    kf = KFold(n_splits=N_split, shuffle=True, random_state=seeds[time])
    for train_index, test_index in  kf.split(data):
        train_all_index[time].append(train_index)
        test_all_index[time].append(test_index)

def pred_svm():
    print('\nstart SVM ......')
    random.seed(0)

    def train_baseline(c, kernel, gamma=1/data.shape[1]):
        tmp_mse = []
        
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = rating[train_index], rating[test_index]
                
                clf = SVR(C=c, kernel=kernel, gamma=gamma,max_iter=1000)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                mse = mean_squared_error(y_test, y_pred)           
                tmp_mse.append(mse)
                
        print('C='+str(c)+',kernel='+kernel+',gamma='+str(gamma)+
                ':\tmse=',np.mean(tmp_mse),'(var=',np.var(tmp_mse),')')
        return tmp_mse
    

    tmp_best_mse = float('inf')
    performances=[]
    C_list = [0.1,1,10,100,1000]
    kernel_list = ['linear','poly','rbf', 'sigmoid']
    gamma_list = [1/100,1/50,1/10,1]
    for kernel in kernel_list:
        for c in C_list:
            if kernel == 'linear':
                tmp_mse = train_baseline(c,kernel)
                if np.mean(tmp_mse)<tmp_best_mse:
                    tmp_best_mse = np.mean(tmp_mse)
                performances.append({'c':c,'kernel':kernel,'gamma':'0','mse':tmp_mse,'mean_mse':np.mean(tmp_mse)})
            else:
                for gamma in gamma_list:
                    tmp_mse=train_baseline(c,kernel,gamma)
                    if np.mean(tmp_mse)<tmp_best_mse:
                        tmp_best_mse = np.mean(tmp_mse)
                    performances.append({'c':c,'kernel':kernel,'gamma':gamma,'mse':tmp_mse,'mean_mse':np.mean(tmp_mse)})
    
    print(tmp_best_mse)
    f = open('regression_result_CIKM/setting1_svm_' + date +  '.txt', 'w', encoding='utf-8') 
    for result in performances:
        print('mse='+str(result['mean_mse'])+' kernel='+str(result['kernel'])+' C='+str(result['c'])+' gamma='+str(result['gamma']))
        f.writelines('mse='+str(result['mean_mse'])+' kernel='+str(result['kernel'])+' C='+str(result['c'])+' gamma='+str(result['gamma'])+
                     ' mse_list='+ str(result['mse']) + '\n')
    f.close()
    json.dump(performances, open('regression_result_CIKM/setting1_svm_' + date + '.json', 'w'))
    
# pred_svm()

def pred_MLP():
    print('\nstart MLP ......')

    def train_baseline(hidden, activate, solve, l2):
        tmp_mse = []
        
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = rating[train_index], rating[test_index]
                
                clf = MLPRegressor(hidden_layer_sizes = hidden, activation=activate, solver = solve, alpha=l2, max_iter = 1000,random_state=model_random_state)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                mse = mean_squared_error(y_test, y_pred)           
                tmp_mse.append(mse)
                
        print('hidden='+str(hidden)+',activate='+activate+',solve'+solve+',l2=',l2,
                ':\tmse=',np.mean(tmp_mse),'(var=',np.var(tmp_mse),')')
        return tmp_mse
    

    tmp_best_mse = float('inf')
    performances=[]
    hidden_list=[(50,),(30,),(10,),(10,10),(30,10),(10,30),(30,30),(25,10,5)]
    activate_list = ['identity','tanh', 'relu']
    solve_list = ['lbfgs', 'adam']
    l2_list=[1e-6,1e-4,1e-2,0]
    for hidden in hidden_list:
        for activate in activate_list:
            for solve in solve_list:
                for l2 in l2_list:
                    tmp_mse = train_baseline(hidden, activate, solve, l2)
                    if np.mean(tmp_mse)<tmp_best_mse:
                        tmp_best_mse = np.mean(tmp_mse)
                    performances.append({'hidden':str(hidden),'activate':activate,'solve':solve,'l2':l2,'mse':tmp_mse,'mean_mse':np.mean(tmp_mse)})
    
    print(tmp_best_mse)
    f = open('regression_result_CIKM/setting1_mlp_' + date +  '.txt', 'w', encoding='utf-8') 
    for result in performances:
        print('mse='+str(result['mean_mse'])+' hidden='+str(result['hidden'])+' activate='+str(result['activate'])+' solve'+str(result['solve']) +' l2='+str(result['l2']))
        f.writelines('auc='+str(result['mean_mse'])+' hidden='+str(result['hidden'])+' activate='+str(result['activate'])+' solve'+str(result['solve']) +' l2='+str(result['l2'])+
                     ' mse_list='+ str(result['mse']) + '\n')
    f.close()
    json.dump(performances, open('regression_result_CIKM/setting1_mlp_' + date + '.json', 'w'))
    
# pred_MLP()


def pred_xgboost():
    print('\nstart xgboost ......')

    def train_baseline(colsample,lr,n_es,gamma):
        tmp_mse = []
        
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = rating[train_index], rating[test_index]
                
                clf = xgb.XGBRegressor(colsample_bytree = colsample, learning_rate = lr, n_estimators = n_es, gamma=gamma,verbosity=0,seed = model_random_state, max_iter=1000)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)           
                tmp_mse.append(mse)
                
        print('lr=',lr,',n_es=',n_es,',gamma=',gamma,
                ':\tmse=',np.mean(tmp_mse),'(var=',np.var(tmp_mse),')')
        return tmp_mse
    

    tmp_best_mse = float('inf')
    performances=[]
    nes_list = [10,30,50,100]
    # colsample_list=[0.7,1]
    # max_depth_list=[3,4,6,9]
    lr_list=[0.01,0.05,0.1]
    gamma_list=[0,0.2,0.4]
    for lr in lr_list:
        for n_es in nes_list:
            for gamma in gamma_list:
                tmp_mse = train_baseline(0.7,lr,n_es,gamma)
                if np.mean(tmp_mse)<tmp_best_mse:
                    tmp_best_mse = np.mean(tmp_mse)
                performances.append({'lr':lr,'n_es':n_es,'gamma':gamma,'mse':tmp_mse,'mean_mse':np.mean(tmp_mse)})
    
    print(tmp_best_mse)
    f = open('regression_result_CIKM/setting1_xgboost_' + date +  '.txt', 'w', encoding='utf-8') 
    for result in performances:
        print('mse='+str(result['mean_mse'])+' lr='+str(result['lr'])+' n_es='+str(result['n_es'])+' gamma='+str(result['gamma']))
        f.writelines('mse='+str(result['mean_mse'])+' lr='+str(result['lr'])+' n_es='+str(result['n_es'])+' gamma='+str(result['gamma'])+
                     ' mse_list='+ str(result['mse']) + '\n')
    f.close()
    json.dump(performances, open('regression_result_CIKM/setting1_xgboost_' + date + '.json', 'w'))
    
pred_xgboost()
