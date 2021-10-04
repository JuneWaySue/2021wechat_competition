import pandas as pd
import numpy as np
import time
import os
from scipy import sparse
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import gc
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

feed_info=pd.read_csv('wechat_algo_data1/feed_info.csv')
user_action=pd.read_csv('wechat_algo_data1/user_action.csv')
feed_embeddings=pd.read_csv('wechat_algo_data1/feed_embeddings.csv')
test_a=pd.read_csv('wechat_algo_data1/test_a.csv')

func={
    'date_':'count','device':'nunique','play':'sum','stay':'sum',
    'read_comment':'max','like':'max','click_avatar':'max','forward':'max',
    'comment':'max','follow':'max','favorite':'max'
}
user_action=user_action.groupby(['userid','feedid']).agg(func).reset_index()

feed_info['manual_keyword_list']=feed_info['manual_keyword_list'].fillna('').apply(lambda x:' '.join(x.split(';')))
feed_info['machine_keyword_list']=feed_info['machine_keyword_list'].fillna('').apply(lambda x:' '.join(x.split(';')))
feed_info['manual_tag_list']=feed_info['manual_tag_list'].fillna('').apply(lambda x:' '.join(x.split(';')))
feed_info['machine_tag_list']=feed_info['machine_tag_list'].fillna('').apply(lambda x:' '.join([i.split(' ')[0] for i in x.split(';')]))

xs=['userid','feedid','device']
ys=['read_comment','like','click_avatar','forward']
vec_cols=['description','ocr','asr','description_char','ocr_char','asr_char']
vec_cols2=['manual_keyword_list','machine_keyword_list','manual_tag_list','machine_tag_list']
num_cols=['play','stay','videoplayseconds']
one_hot_cols=['authorid','bgm_song_id','bgm_singer_id']
del_cols=['date_','comment','follow','favorite']

def creat_npz(data):
    df_feature=pd.DataFrame()
    for col in one_hot_cols+['userid','feedid']:
        s = time.time()
        LE=LabelEncoder()
        if col == 'userid':
            try:
                LE.fit(user_action[col].apply(int))
            except:
                LE.fit(user_action[col])
            user_action[col]=LE.transform(user_action[col])
            data[col]=LE.transform(data[col])
            OHE=OneHotEncoder()
            OHE.fit(user_action[col].values.reshape(-1, 1))
            arr=OHE.transform(data[col].values.reshape(-1, 1))
            df_feature = sparse.hstack((df_feature,arr))
            print(col,int(time.time()-s),'s')
        else:
            try:
                LE.fit(feed_info[col].apply(int))
            except:
                LE.fit(feed_info[col])
            feed_info[col]=LE.transform(feed_info[col])
            data[col]=LE.transform(data[col])
            OHE=OneHotEncoder()
            OHE.fit(feed_info[col].values.reshape(-1, 1))
            arr=OHE.transform(data[col].values.reshape(-1, 1))
            df_feature = sparse.hstack((df_feature,arr))
            print(col,int(time.time()-s),'s')
    sparse.save_npz("data_process/one_hot_cols.npz",df_feature)

    df_feature=pd.DataFrame()
    for col in vec_cols:
        s = time.time()
        print(col,'start...')
        CV=CountVectorizer()
        CV.fit(feed_info[col].fillna('-1'))
        arr = CV.transform(data[col].fillna('-1'))
        print(col,'hstack...')
        df_feature = sparse.hstack((df_feature,arr))
        arr=[]
        print(f'{col} save...')
        sparse.save_npz(f"data_process/vec_cols_{col}.npz",df_feature)
        print(col,int(time.time()-s),'s')
        df_feature=pd.DataFrame()
        
    df_feature=pd.DataFrame()
    for col in vec_cols2:
        s = time.time()
        print(col,'start...')
        CV=CountVectorizer()
        CV.fit(feed_info[col])
        arr = CV.transform(data[col])
        print(col,'hstack...')
        df_feature = sparse.hstack((df_feature,arr))
        arr=[]
        print(col,int(time.time()-s),'s')
    sparse.save_npz("data_process/vec_cols2.npz",df_feature)

data=pd.concat([user_action[['userid','feedid']],test_a[['userid','feedid']]],ignore_index=True)
data=pd.merge(data,feed_info[['feedid']+one_hot_cols+vec_cols+vec_cols2],on='feedid',how='left')
creat_npz(data)

def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,100*(start_mem-end_mem)/start_mem,(time.time()-starttime)/60))
    return df

def process(x):
    num_list=x.split(' ')[:-1]
    res={}
    for i,num in enumerate(num_list):
        res[i]=float(num)
    return pd.Series(res)
    
feed_embeddings_512=feed_embeddings.feed_embedding.apply(process)
pca = PCA(n_components=32,random_state=2021)
feed_embeddings_32 = pd.DataFrame(pca.fit_transform(feed_embeddings_512))
del feed_embeddings['feed_embedding']
feed_embeddings=pd.concat([feed_embeddings,feed_embeddings_32],axis=1)
feed_embeddings_pca=pd.concat([user_action[['userid','feedid']],test_a[['userid','feedid']]],ignore_index=True)
feed_embeddings_pca=pd.merge(feed_embeddings_pca,feed_embeddings,on='feedid',how='left')
del feed_embeddings_pca['userid']
del feed_embeddings_pca['feedid']
del feed_embeddings_512,feed_embeddings_32,feed_embeddings
gc.collect()
feed_embeddings_pca=reduce_mem(feed_embeddings_pca)
feed_embeddings_pca.to_hdf('data_process/feed_embeddings_pca.h5',key='pca',mode='w')