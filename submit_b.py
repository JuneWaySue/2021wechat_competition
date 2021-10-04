import pandas as pd
import numpy as np
import time
import os
from scipy import sparse
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, KFold
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

models_path='../input/my-competition-wechat-data/models/'
data_process_b_path='../input/my-competition-wechat-data/data_process_b/'
data_process_path='../input/my-competition-wechat-data/data_process/'
npzs_path='../input/my-competition-wechat-data/npzs/'
feature_importances_path='../input/my-competition-wechat-data/feature_importances/'
wechat_algo_data1_path='../input/my-competition-wechat-data/wechat_algo_data1/'

user_action=pd.read_csv(wechat_algo_data1_path+'user_action.csv')
test_b=pd.read_csv(data_process_b_path+'test_b.csv')

submit_b_model1=test_b[['userid','feedid']]
submit_b_model2=test_b[['userid','feedid']]
submit_b=test_b[['userid','feedid']]
test_b=reduce_mem(test_b)

func={
	'date_':'count','device':'nunique','play':'sum','stay':'sum',
	'read_comment':'max','like':'max','click_avatar':'max','forward':'max',
	'comment':'max','follow':'max','favorite':'max'
}
user_action=user_action.groupby(['userid','feedid']).agg(func).reset_index()
user_action=reduce_mem(user_action)

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
	feature1 = sparse.hstack((feature1,sparse.load_npz(f'{data_process_b_path}one_hot_cols_{label}.npz')))
	feature=sparse.hstack((feature,feature1))
	del feature1
	gc.collect()
	print('1.已清理部分内存！！')
	
	feature2 = pd.DataFrame()
	feature2 = sparse.hstack((feature2,sparse.load_npz(f'{data_process_b_path}vec_cols2_{label}.npz')))
	feature=sparse.hstack((feature,feature2))
	del feature2
	gc.collect()
	print('2.已清理部分内存！！')
	
	feed_embeddings_pca=pd.read_hdf(f'{data_process_b_path}feed_embeddings_pca_b.h5',key='pca')
	feature=sparse.hstack((feature,feed_embeddings_pca)).tocsc()
	del feed_embeddings_pca
	gc.collect()
	print('3.已清理部分内存！！')
	
	print(label,'feature 的 shape',feature.shape)
	
	data=pd.concat([user_action[['userid','feedid',label]],test_b[['userid','feedid']]],ignore_index=True).fillna(-1)
	test_index=data[data[label]==-1].index.to_list()
	test_x=feature[test_index,:]
	print(label,'data is ready')
	
	del feature,data
	gc.collect()
	print('4.已清理部分内存！！')
	
	model_file=f"{models_path}{label}.pickle.dat"
	clf=pickle.load(open(model_file, "rb"))
	
	print(label,'predicting test...')
	test_pred=clf.predict_proba(test_x,num_iteration=clf.best_iteration_)[:,1]
	submit_b_model1[label]=test_pred
	
	del clf,test_x
	gc.collect()
	print('5.已清理部分内存！！')
	print(label,'耗时',int(time.time()-s),'s')
	
submit_file=f'submit_b_model1.csv'
submit_b_model1.to_csv(submit_file,index=False)


feed_info=pd.read_csv(wechat_algo_data1_path+'feed_info.csv')
user_action=pd.read_csv(wechat_algo_data1_path+'user_action.csv')
test_b=pd.read_csv(data_process_b_path+'test_b.csv')

func={
	'date_':'count','device':'nunique','play':'sum','stay':'sum',
	'read_comment':'max','like':'max','click_avatar':'max','forward':'max',
	'comment':'max','follow':'max','favorite':'max'
}
user_action=user_action.groupby(['userid','feedid']).agg(func).reset_index()

def label_encode(data):
	df_feature=pd.DataFrame()
	for col in one_hot_cols+['userid','feedid']:
		LE=LabelEncoder()
		if col == 'userid':
			try:
				LE.fit(user_action[col].apply(int))
			except:
				LE.fit(user_action[col])
			data[col]=LE.transform(data[col])
		else:
			try:
				LE.fit(feed_info[col].apply(int))
			except:
				LE.fit(feed_info[col])
			data[col]=LE.transform(data[col])

sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)

for label in ys:
	sj=time.time()
	feature=pd.concat([user_action[['userid','feedid']],test_b[['userid','feedid']]],ignore_index=True)
	feature=pd.merge(feature,feed_info[['feedid']+one_hot_cols],on='feedid',how='left')
	label_encode(feature)

	feed_embeddings_pca=pd.read_hdf(f'{data_process_b_path}feed_embeddings_pca_b.h5',key='pca')
	feature=pd.concat([feature,feed_embeddings_pca],axis=1)
	print('feature 的 shape',feature.shape)
	del feed_embeddings_pca
	gc.collect()
	
	data=pd.concat([user_action[['userid','feedid',label]],test_b[['userid','feedid']]],ignore_index=True).fillna(-1)
	test_index=data[data[label]==-1].index.to_list()
	test_feature=feature.loc[test_index,:]

	del feature,data
	gc.collect()
	print(label,'data is ready')

	for k_fold in range(5):
		print('k_fold',k_fold,'begin')
		s=time.time()
		
		model_file=f"{models_path}{label}.k_fold{k_fold}.pickle.dat"
		clf=pickle.load(open(model_file, "rb"))
		
		print('k_fold',k_fold,label,'predicting test...')
		test_pred=clf.predict_proba(test_feature,num_iteration=clf.best_iteration_)[:,1]
		submit_b_model2[label+f'_k_fold{k_fold}']=test_pred
		
		del clf
		gc.collect()
		print('已清理部分内存！！')
		print('k_fold',k_fold,label,'耗时',int(time.time()-s),'s')
	del test_feature
	gc.collect()
	print(label,'总耗时',int(time.time()-sj),'s')

for label in ys:
	cols=[i for i in submit_b_model2.columns.to_list() if label+'_k_fold' in i]
	submit_b_model2[label]=submit_b_model2[cols].mean(axis=1)
submit_b_model2=submit_b_model2[[i for i in submit_b_model2.columns.to_list() if 'k_fold' not in i]]

submit_file=f'submit_b_model2.csv'
submit_b_model2.to_csv(submit_file,index=False)

for col in submit_b_model1.columns.to_list()[2:]:
	submit_b[col]=(submit_b_model1[col]+submit_b_model2[col])/2
submit_b.to_csv('submit_b.csv',index=False)

print('总耗时',int(time.time()-start_time),'s')