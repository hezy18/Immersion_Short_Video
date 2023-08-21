import json
import random
import numpy as np
import tqdm
import pandas as pd
import time

from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

import clf_models
import warnings
warnings.filterwarnings('ignore')
import config
exp_list = config.exp_list
date=time.strftime("%Y-%m-%d",time.localtime(time.time()))

seeds = [1,2,3,4,5]
model_random_state=99

setting=1 #1,2,3

# load data
label = np.load('setting1_label.npy')
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

    def train_baseline(c, kernel, gamma=1/data.shape[1]):
        tmp_acc = []
        
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                
                clf = SVC(C=c, kernel=kernel, gamma=gamma, decision_function_shape='ovo',random_state=model_random_state,max_iter=1000)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                acc = accuracy_score(y_test, y_pred)           
                tmp_acc.append(acc)
                
        print('C='+str(c)+',kernel='+kernel+',gamma='+str(gamma)+
                ':\tacc=',np.mean(tmp_acc),'(var=',np.var(tmp_acc),')')
        return tmp_acc
    

    tmp_best_acc = 0
    performances=[]
    C_list = [0.1,1,10,100,1000]
    kernel_list = ['linear','poly','rbf', 'sigmoid']
    gamma_list = [1/100,1/50,1/10,1,5]
    for kernel in kernel_list:
        for c in C_list:
            if kernel == 'linear':
                tmp_acc = train_baseline(c,kernel)
                if np.mean(tmp_acc)>tmp_best_acc:
                    tmp_best_acc = np.mean(tmp_acc)
                performances.append({'c':c,'kernel':kernel,'gamma':'0','acc':tmp_acc,'mean_acc':np.mean(tmp_acc)})
            else:
                for gamma in gamma_list:
                    tmp_acc=train_baseline(c,kernel,gamma)
                    if np.mean(tmp_acc)>tmp_best_acc:
                        tmp_best_acc = np.mean(tmp_acc)
                    performances.append({'c':c,'kernel':kernel,'gamma':gamma,'acc':tmp_acc,'mean_acc':np.mean(tmp_acc)})
    
    print(tmp_best_acc)
    f = open('predict_result_CIKM/setting1_svm_' + date +  '.txt', 'w', encoding='utf-8') 
    for result in performances:
        print('acc='+str(result['mean_acc'])+' kernel='+str(result['kernel'])+' C='+str(result['c'])+' gamma='+str(result['gamma']))
        f.writelines('acc='+str(result['mean_acc'])+' kernel='+str(result['kernel'])+' C='+str(result['c'])+' gamma='+str(result['gamma'])+
                     ' acc_list='+ str(result['acc']) + '\n')
    f.close()
    json.dump(performances, open('predict_result_CIKM/setting1_svm_' + date + '.json', 'w'))
    
# pred_svm()

def pred_MLP():
    print('\nstart MLP ......')

    def train_baseline(hidden, activate, solve, l2):
        tmp_acc = []
        
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                
                clf = MLPClassifier(hidden_layer_sizes = hidden, activation=activate, solver = solve, alpha=l2, max_iter = 1000,random_state=model_random_state)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                acc = accuracy_score(y_test, y_pred)           
                tmp_acc.append(acc)
                
        print('hidden='+str(hidden)+',activate='+activate+',solve'+solve+',l2=',l2,
                ':\tacc=',np.mean(tmp_acc),'(var=',np.var(tmp_acc),')')
        return tmp_acc
    

    tmp_best_acc = 0
    performances=[]
    hidden_list=[(50,),(30,),(10,),(10,10),(30,10),(10,30),(30,30),(25,10,5)]
    activate_list = ['identity','tanh', 'relu']
    solve_list = ['lbfgs', 'adam']
    l2_list=[1e-6,1e-4,1e-2,0]
    for hidden in hidden_list:
        for activate in activate_list:
            for solve in solve_list:
                for l2 in l2_list:
                    tmp_acc = train_baseline(hidden, activate, solve, l2)
                    if np.mean(tmp_acc)>tmp_best_acc:
                        tmp_best_acc = np.mean(tmp_acc)
                    performances.append({'hidden':str(hidden),'activate':activate,'solve':solve,'l2':l2,'acc':tmp_acc,'mean_acc':np.mean(tmp_acc)})
    
    print(tmp_best_acc)
    f = open('predict_result_CIKM/setting1_mlp_' + date +  '.txt', 'w', encoding='utf-8') 
    for result in performances:
        print('acc='+str(result['mean_acc'])+' hidden='+str(result['hidden'])+' activate='+str(result['activate'])+' solve'+str(result['solve']) +' l2='+str(result['l2']))
        f.writelines('acc='+str(result['mean_acc'])+' hidden='+str(result['hidden'])+' activate='+str(result['activate'])+' solve'+str(result['solve']) +' l2='+str(result['l2'])+
                     ' acc_list='+ str(result['acc']) + '\n')
    f.close()
    json.dump(performances, open('predict_result_CIKM/setting1_mlp_' + date + '.json', 'w'))
    
# pred_MLP()


def pred_xgboost():
    print('\nstart xgboost ......')

    def train_baseline(lr,n_es,gamma):
        tmp_acc = []
        
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                
                clf = xgb.XGBClassifier(colsample_bytree = 0.7, learning_rate = lr, n_estimators = n_es, gamma=gamma,verbosity=0,seed = model_random_state,max_iter=1000)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)           
                tmp_acc.append(acc)
                
        print('lr=',lr,',n_es=',n_es,',gamma=',gamma,
                ':\tacc=',np.mean(tmp_acc),'(var=',np.var(tmp_acc),')')
        return tmp_acc
    

    tmp_best_acc = 0
    performances=[]
    nes_list = [10,30,50,100]
    # colsample_list=[0.7,1]
    # max_depth_list=[3,4,6,9]
    lr_list=[0.01,0.05,0.1]
    gamma_list=[0,0.2,0.4]
    for lr in lr_list:
        for n_es in nes_list:
            for gamma in gamma_list:
                tmp_acc = train_baseline(lr,n_es,gamma)
                if np.mean(tmp_acc)>tmp_best_acc:
                    tmp_best_acc = np.mean(tmp_acc)
                performances.append({'lr':lr,'n_es':n_es,'gamma':gamma,'acc':tmp_acc,'mean_acc':np.mean(tmp_acc)})
    
    print(tmp_best_acc)
    f = open('predict_result_CIKM/setting1_xgboost_' + date +  '.txt', 'w', encoding='utf-8') 
    for result in performances:
        print('acc='+str(result['mean_acc'])+' lr='+str(result['lr'])+' n_es='+str(result['n_es'])+' gamma='+str(result['gamma']))
        f.writelines('acc='+str(result['mean_acc'])+' lr='+str(result['lr'])+' n_es='+str(result['n_es'])+' gamma='+str(result['gamma'])+
                     ' acc_list='+ str(result['acc']) + '\n')
    f.close()
    json.dump(performances, open('predict_result_CIKM/setting1_xgboost_' + date + '.json', 'w'))
    
pred_xgboost()
