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


feed_info=pd.read_csv('wechat_algo_data1/feed_info.csv')
user_action=pd.read_csv('wechat_algo_data1/user_action.csv')
test_a=pd.read_csv('wechat_algo_data1/test_a.csv')

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


for npz_file in ['one_hot_cols','vec_cols2','vec_cols_description']:
    start_time=time.time()
    evals_score={}

    for label in ys:
        s=time.time()

        feature = pd.DataFrame()
        feature = sparse.hstack((feature,sparse.load_npz(f'data_process/{npz_file}.npz').tocsr())).tocsc()
        
        print(npz_file,label,'feature 的 shape',feature.shape)
        
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
        print(npz_file,label,'data is ready')
        
        del feature,data
        gc.collect()
        print('已清理部分内存！！')
        
        clf = LGBMClassifier(boosting_type='gbdt',
                             num_leaves=31, max_depth=-1, 
                             learning_rate=0.1, n_estimators=10000, 
                             subsample_for_bin=200000, objective=None,
                             class_weight=None, min_split_gain=0.0, 
                             min_child_weight=0.001,
                             min_child_samples=20, subsample=1.0, subsample_freq=1,
                             colsample_bytree=1.0,
                             reg_alpha=0.0, reg_lambda=0.0, random_state=2021,
                             n_jobs=-1, silent=True)
        
        model_file=f"model/{npz_file}_{label}.pickle.dat"
        if os.path.exists(model_file):
            clf=pickle.load(open(model_file, "rb"))
        else:
            print(npz_file,label,'fitting...')
            clf.fit(train_x,train_y,eval_set=[(train_x, train_y),(evals_x, evals_y)], 
                    eval_names =['train','valid'],
                    eval_metric='auc',early_stopping_rounds=50)

            print(npz_file,label,'dumping...')
            pickle.dump(clf, open(model_file, "wb"))
            
        se = pd.Series(clf.feature_importances_)
        se = se[se>0]
        col =list(se.sort_values(ascending=False).index)
        filename=f'data_process/feature_importances_{npz_file}_{label}.csv'
        pd.Series(col).to_csv(filename,index=False)
        print(npz_file,label,'特征重要性不为零的编码特征有',len(se),'个')
        n = clf.best_iteration_
        print(npz_file,label,'n',n)
        baseloss = clf.best_score_['valid']['auc']
        print(npz_file,label,'baseloss',baseloss)

        print(npz_file,label,'predicting evals...')
        evals_pred=clf.predict_proba(evals_x,num_iteration=clf.best_iteration_)[:,1]
        score=roc_auc_score(evals_y.values,evals_pred)
        print(npz_file,label,'很高兴本次模型验证集得分为',score)
        evals_score[label]=score
        
        print(npz_file,label,'predicting test...')
        test_pred=clf.predict_proba(test_x,num_iteration=clf.best_iteration_)[:,1]
        submit[label]=test_pred
        
        del clf,train_x,evals_x,test_x,train_y,evals_y,y
        gc.collect()
        print(npz_file,label,'耗时',int(time.time()-s),'s')
        
    now_time=int(time.time())
    submit_file=f'submit/my_submit_{npz_file}_{now_time}.csv'
    submit.to_csv(submit_file,index=False)

    evals_file=f'submit/evals_score_{npz_file}_{now_time}.csv'
    evals_df=pd.DataFrame([evals_score])
    evals_df.to_csv(evals_file,index=False)
    print(npz_file,'线下得分',np.sum(evals_df.values[0] * np.array([4,3,2,1]))/10)
    print(npz_file,'总耗时',int(time.time()-start_time),'s')

for y in ys:
    for npz_file in ['one_hot_cols','vec_cols2','vec_cols_description']:
        col=pd.read_csv(f'data_process/feature_importances_{npz_file}_{y}.csv')['0'].values.tolist()
        feature = pd.DataFrame()
        feature = sparse.hstack((feature,sparse.load_npz(f'data_process/{npz_file}.npz').tocsr()[:,col]))
        print(y,npz_file,feature.shape)
        sparse.save_npz(f'data_process/{npz_file}_{y}.npz',feature)