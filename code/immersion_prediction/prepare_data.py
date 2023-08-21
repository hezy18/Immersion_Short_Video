import json
import scipy.stats

import copy
import numpy as np
import config
exp_list = config.exp_list
import pandas as pd

print(len(exp_list))
picked_channels = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2", ]
total_channels = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2", ]

select_index = [idx for idx in range(len(total_channels)) if total_channels[idx] in picked_channels]
# make data

import json

from tqdm import tqdm

def get_user_feauture():
    exp_feature={}
    age_list=[]
    expname_find={}
    subject_df = pd.read_csv('participants.csv',dtype={'exp':str})
    num_gender_1=0
    for i in range(len(subject_df)):
        exp = subject_df['exp'][i]
        print(exp)
        # print(exp)
        find=-1
        for expname in exp_list:
            if expname.split('_')[0]==str(exp):
                if expname in expname_find:
                    continue
                find=expname
                print(expname,exp,i)
            
        if find==-1:
            continue
        expname_find[find]=i
        gender = int(subject_df['6、您的性别'][i])
        age = int(subject_df['7、您的年龄'][i])
        usage = int(subject_df['8、您使用抖音APP的年限？'][i])
        if gender==1:
            num_gender_1+=1
        age_list.append(age)
        exp_feature[find] = [gender,age,usage]
    print(exp_feature)
    for expname in exp_list:
        if expname not in expname_find:
            print('no',expname)
    
    print('age',np.mean(age_list),np.std(age_list,ddof=1),max(age_list),min(age_list))
    print('gender_1', num_gender_1)
    return exp_feature


def no_EEG_feature(): # setting1
    exp_feature = get_user_feauture()
    VF_df = pd.read_csv('VIDEO_FEATURES.csv', dtype={'id':str})

    video2tag_en = json.load(open('video2tag_en.json'))

    audio_feature_list=['audspec_lengthL1norm_sma_range',
              'pcm_RMSenergy_sma_amean', 'pcm_RMSenergy_sma_range',
              'F0final_sma_amean', 'F0final_sma_range', 'F0final_sma_stddev',
              'logHNR_sma_amean', 'logHNR_sma_range',
              'pcm_fftMag_fband250-650_sma_amean', 'pcm_fftMag_fband250-650_sma_range', 'pcm_fftMag_fband250-650_sma_stddev', 
              'pcm_fftMag_psySharpness_sma_range', 'pcm_fftMag_psySharpness_sma_amean', 'pcm_fftMag_psySharpness_sma_stddev', 
              'shimmerLocal_sma_amean','shimmerLocal_sma_range','shimmerLocal_sma_stddev',
              'pcm_fftMag_spectralFlux_sma_amean','pcm_fftMag_spectralFlux_sma_range','pcm_fftMag_spectralFlux_sma_stddev' 
              ]
    video_feature_list=['count', 'brightness', 'dif_brightness', 'E_1D', 'dif_E_1D', 'E_2D', 'dif_E_2D', 
                    'contrast', 'dif_contrast', 'laplace_var', 'dif_laplace_var', 'color_cast', 'dif_color_cast',
                    'hue', 'dif_hue', 'saturation', 'dif_saturation', 'value', 'dif_value']
    columns = ['iid','uid','gender', 'age', 'usage','tagid','music_tempo','audio_category']
    
    for audio_feature in audio_feature_list:
        columns.append(audio_feature)
    for video_feature in video_feature_list:
        columns.append(video_feature)
    print('feature_num:', len(columns))
    df_ui=pd.DataFrame(columns=columns)
    tmp_iid=0
    tmp_uid=0
    tmp_tagid=0
    iid_unique={}
    uid_unique={}
    tagid_unique={}
    y=[]
    video_id_list=[]
    for exp_name in tqdm(exp_list):
        v2info = json.load(open('data/raw/'+exp_name+'_MAEs.json'))
        eeg_data = json.load(open('data/raw/'+exp_name+'_idx2de_nor_avg.json'))
        last_session_id=-1
        this_video_id=1
        if exp_name not in uid_unique:
            uid_unique[exp_name]=tmp_uid
            tmp_uid+=1
        for v in v2info.keys():
            if 'immersion' not in v2info[v].keys():
                continue
            if np.isnan(eeg_data[str(v2info[v]['idx'])][0][0]):
                continue
            if str(v2info[v]['idx']) not in eeg_data:
                continue
            immersion = v2info[v]['immersion']
            if v not in iid_unique:
                iid_unique[v]=tmp_iid
                tmp_iid+=1
            if video2tag_en[v] not in tagid_unique:
                tagid_unique[video2tag_en[v]]=tmp_tagid
                tmp_tagid+=1
            session_id = v2info[v]["session_id"] # 0、1、2、...
            if session_id==last_session_id:
                video_id=this_video_id
                this_video_id+=1
            else:
                this_video_id=1
                video_id=this_video_id
                this_video_id+=1
                last_session_id = session_id
            
            if v2info[v]['preference'] == 'random':
                preference = 1
            elif v2info[v]['preference'] == 'recommendation':
                preference = 3
            elif v2info[v]['preference'] == 'typical':
                preference = 0
            else:
                preference = 2
            video_id_list.append(video_id)
            line_dict = {'iid': uid_unique[exp_name],'uid':iid_unique[v],'session_id':session_id,'preference':preference,
                         'gender':exp_feature[exp_name][0], 'age':exp_feature[exp_name][1], 'usage':exp_feature[exp_name][2],
                         'tagid':tagid_unique[video2tag_en[v]]}
            
            mask = VF_df['id']==str(v)
            pos = np.flatnonzero(mask)
            if len(pos)==0 or VF_df['color_cast'][pos[0]] > 100:
                df_ui = df_ui.append(pd.DataFrame([line_dict]),ignore_index=True)
                y.append(immersion)
                continue
            category = VF_df['audio_category'][pos[0]]
            if category == 'Music':
                line_dict['audio_category']=0
            elif category == 'Speech':
                line_dict['audio_category']=1
            else:
                line_dict['audio_category']=2
            for column in columns:
                if column not in line_dict:
                    if not isinstance(VF_df[column][pos[0]], str):
                        line_dict[column] = float(VF_df[column][pos[0]])
                    else:
                        print('is str', column)
                        line_dict[column] = str(VF_df[column][pos[0]])
            df_ui = df_ui.append(pd.DataFrame([line_dict]),ignore_index=True)

            y.append(immersion)
    label = [1 if x<3 else 0 for x in y]
    
    # 处理空值
    df_filled = df_ui.fillna(df_ui.mean()) # 均值

    # 归一化
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    nor_X_np = scaler.fit_transform(df_filled)
    
    print(nor_X_np)
    print(nor_X_np.shape)
    
    y_np = np.array(y)
    print(y_np.shape)
    np.save('setting1_noEEG_data.npy',nor_X_np)
    np.save('setting1_rating.npy',y_np)
    np.save('setting1_label.npy',np.array(label))
    print(video_id_list)
    np.save('video_order_list.npy',np.array(video_id_list))
      
no_EEG_feature()

def run_topo(exp_names):
    data = {
    'ims':[],
    'not_ims':[]
    }   
    for exp_name in exp_names:
        v2info = json.load(open('data/raw/'+exp_name+'_MAEs.json'))
        idx2de = json.load(open('data/raw/'+exp_name+'_idx2de_nor_avg.json'))
        for v in v2info.keys():
            if 'immersion' not in v2info[v].keys():
                continue
            if np.isnan(idx2de[str(v2info[v]['idx'])][0][0]):
                continue
            if v2info[v]['immersion'] >= 4:
                data['ims'].append(idx2de[str(v2info[v]['idx'])])
            elif v2info[v]['immersion'] <= 3:
                data['not_ims'].append(idx2de[str(v2info[v]['idx'])])

    data['not_ims'] = np.array(data['not_ims'])
    data['ims'] = np.array(data['ims'])
    significance = np.ones((62,5))
    diff = np.zeros((62,5))
    i=0
    channel_band=[]
    for channel in range(62):
        for band in range(5):
            dislike_list = data['not_ims'][:,channel,band]
            like_list = data['ims'][:,channel,band]
            y = [0 for i in range(len(dislike_list))] + [1 for i in range(len(like_list))]
            x = dislike_list.tolist() + like_list.tolist()
            r, pval = scipy.stats.pearsonr(x, y)
            significance[channel,band] = pval
            diff[channel,band] = r
            # print(channel,band,r,pval)
            if pval<0.05 and abs(r)>0.05:
                # print(channel,band,r,pval)
                channel_band.append((channel,band))
                i+=1
    print(i)
    return channel_band


def only_EEG_feature(): # setting2
    channel_band = run_topo(exp_list)    
    X=[]
    y=[]
    tmp_uid=0
    uid_unique={}
    for exp_name in tqdm(exp_list):
        v2info = json.load(open('data/raw/'+exp_name+'_MAEs.json'))
        eeg_data = json.load(open('data/raw/'+exp_name+'_idx2de_nor_avg.json'))
        
        if exp_name not in uid_unique:
            uid_unique[exp_name]=tmp_uid
            tmp_uid+=1
            
        for v in v2info.keys():
            if 'immersion' not in v2info[v].keys():
                continue
            if str(v2info[v]['idx']) not in eeg_data:
                continue
            if np.isnan(eeg_data[str(v2info[v]['idx'])][0][0]):
                continue
            eeg = eeg_data[str(v2info[v]['idx'])]
            immersion = v2info[v]['immersion']
            line_eeg=[]
            for pair in channel_band:
                line_eeg.append(eeg[pair[0]][pair[1]])
            
            # print(len(line_eeg))
            
            X.append(line_eeg)
            y.append(immersion)
    
    
    y_set=set(y)
    y_dict={}
    for item in y_set:
        y_dict.update({item:y.count(item)})
    print(y_dict)
    
    for item in y_set:
        print(item, float(y_dict[item]/len(y)))
    
    label = [1 if x<3 else 0 for x in y]
    
    X_np = np.array(X)
    print(X_np.shape)
    
    y_np = np.array(y)
    
    
    
    print(y_np.shape)
    np.save('setting2_EEG_data.npy',X_np)
    np.save('setting2_rating.npy',y_np)
    np.save('setting2_label.npy',np.array(label))
            
only_EEG_feature()

