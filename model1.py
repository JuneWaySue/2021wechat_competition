import pandas as pd
import numpy as np
import time
import os
from scipy import sparse
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import warnings
import requests
import pickle
import gc
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

data_process_path='../input/my-competition-wechat-data/data_process/'
npzs_path='../input/my-competition-wechat-data/npzs/'
feature_importances_path='../input/my-competition-wechat-data/feature_importances/'
wechat_algo_data1_path='../input/my-competition-wechat-data/wechat_algo_data1/'

feed_info=pd.read_csv(wechat_algo_data1_path+'feed_info.csv')
user_action=pd.read_csv(wechat_algo_data1_path+'user_action.csv')
test_a=pd.read_csv(wechat_algo_data1_path+'test_a.csv')

submit=test_a[['userid','feedid']]

func={
    'date_':'count','device':'nunique','play':'sum','stay':'sum',
    'read_comment':'max','like':'max','click_avatar':'max','forward':'max',
    'comment':'max','follow':'max','favorite':'max'
}
user_action=user_action.groupby(['userid','feedid']).agg(func).reset_index()

xs=['userid','feedid','device']
ys=['read_comment','like','click_avatar','forward']
vec_cols=['description','ocr','asr','description_char','ocr_char','asr_char']
vec_cols2=['manual_keyword_list','machine_keyword_list','manual_tag_list','machine_tag_list']
num_cols=['play','stay','videoplayseconds']
one_hot_cols=['authorid','bgm_song_id','bgm_singer_id']
del_cols=['date_','comment','follow','favorite']

start_time=time.time()

for label in ys:
    s=time.time()
    
    feature = pd.DataFrame()
    
    feature1 = pd.DataFrame()
    feature1 = sparse.hstack((feature1,sparse.load_npz(f'{data_process_path}one_hot_cols_{label}.npz')))
    feature=sparse.hstack((feature,feature1))
    del feature1
    gc.collect()
    print('1.已清理部分内存！！')
    
    feature2 = pd.DataFrame()
    feature2 = sparse.hstack((feature2,sparse.load_npz(f'{data_process_path}vec_cols2_{label}.npz')))
    feature=sparse.hstack((feature,feature2))
    del feature2
    gc.collect()
    print('2.已清理部分内存！！')
    
    feed_embeddings_pca=pd.read_hdf('../input/my-competition-wechat-data/pca/feed_embeddings_pca.h5',key='pca')
    feature=sparse.hstack((feature,feed_embeddings_pca)).tocsc()
    del feed_embeddings_pca
    gc.collect()
    print('3.已清理部分内存！！')
    
    print(label,'feature 的 shape',feature.shape)
    
    data=pd.concat([user_action[['userid','feedid',label]],test_a[['userid','feedid']]],ignore_index=True).fillna(-1)

    train_index=data[data[label]!=-1].index.to_list()
    test_index=data[data[label]==-1].index.to_list()
    y=data.loc[train_index,label]

    ind_evals=[]
    ind_evals.extend(y[y==0].sample(frac=0.1,random_state=2021).index.to_list())
    ind_evals.extend(y[y==1].sample(frac=0.1,random_state=2021).index.to_list())
    ind_train=y.drop(index=ind_evals).index.to_list()

    train_x=feature[ind_train,:]
    train_y=y[ind_train]

    evals_x=feature[ind_evals,:]
    evals_y=y[ind_evals]

    test_x=feature[test_index,:]
    print(label,'data is ready')
    
    del feature,data
    gc.collect()
    print('4.已清理部分内存！！')
    
    clf = LGBMClassifier(boosting_type='gbdt', num_leaves=40, max_depth=-1, learning_rate=0.1, 
                n_estimators=10000, subsample_for_bin=200000, objective=None, 
                class_weight=None, min_split_gain=0.0, min_child_weight=0.001, 
                min_child_samples=20, subsample=0.7, subsample_freq=1, 
                colsample_bytree=0.7, 
                reg_alpha=6, reg_lambda=3,
                random_state=2021, n_jobs=-1, silent=True)
    
    model_file=f"{label}.pickle.dat"
    if os.path.exists(model_file):
        clf=pickle.load(open(model_file, "rb"))
    else:
        print(label,'fitting...')
        clf.fit(train_x,train_y,eval_set=[(train_x, train_y),(evals_x, evals_y)], 
                eval_names =['train','valid'],
                eval_metric='auc',early_stopping_rounds=50)

        print(label,'dumping...')
        pickle.dump(clf, open(model_file, "wb"))
    
    print(label,'predicting test...')
    test_pred=clf.predict_proba(test_x,num_iteration=clf.best_iteration_)[:,1]
    submit[label]=test_pred
    submit.to_csv(f'my_submit_{label}.csv',index=False)
    
    del clf,train_x,test_x,train_y,evals_x,evals_y
    gc.collect()
    print('5.已清理部分内存！！')
    print(label,'耗时',int(time.time()-s),'s')
    
submit_file=f'submit_a_model1.csv'
submit.to_csv(submit_file,index=False)