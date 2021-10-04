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

def get_feature_data(label):
	feature=pd.concat([user_action[['userid','feedid']],test_a[['userid','feedid']]],ignore_index=True)
	feature=pd.merge(feature,feed_info[['feedid']+one_hot_cols],on='feedid',how='left')
	label_encode(feature)

	feed_embeddings_pca=pd.read_hdf('../input/my-competition-wechat-data/pca/feed_embeddings_pca.h5',key='pca')
	feature=pd.concat([feature,feed_embeddings_pca],axis=1)
	del feed_embeddings_pca
	gc.collect()
	
	data=pd.concat([user_action[['userid','feedid',label]],test_a[['userid','feedid']]],ignore_index=True).fillna(-1)
	return feature,data

start_time=time.time()
sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
evals_score=pd.DataFrame(columns=ys)

for label in ys:
	sj=time.time()
	feature,data=get_feature_data(label)
	print('feature 的 shape',feature.shape)

	train_index=data[data[label]!=-1].index.to_list()
	test_index=data[data[label]==-1].index.to_list()
	y=data.loc[train_index,label]

	train_feature=feature.loc[train_index,:]
	test_feature=feature.loc[test_index,:]

	del feature,data
	gc.collect()
	print(label,'data is ready')

	for k_fold,(k_train_index,k_evals_index) in enumerate(sk.split(train_feature,y)):
		print('k_fold',k_fold,'begin')
		s=time.time()

		train_x=train_feature.loc[k_train_index,:]
		train_y=y[k_train_index]

		evals_x=train_feature.loc[k_evals_index,:]
		evals_y=y[k_evals_index]
		
		clf = LGBMClassifier(boosting_type='gbdt', num_leaves=40, max_depth=-1, learning_rate=0.1, 
					n_estimators=10000, subsample_for_bin=200000, objective=None, 
					class_weight=None, min_split_gain=0.0, min_child_weight=0.001, 
					min_child_samples=20, subsample=0.7, subsample_freq=1, 
					colsample_bytree=0.7,categorical_feature=[0,1,2,3,4],
					reg_alpha=6, reg_lambda=3,
					random_state=2021, n_jobs=-1, silent=True)
		
		model_file=f"{label}.k_fold{k_fold}.pickle.dat"
		if os.path.exists(model_file):
			clf=pickle.load(open(model_file, "rb"))
		else:
			print('k_fold',k_fold,label,'fitting...')
			clf.fit(train_x,train_y,eval_set=[(train_x, train_y),(evals_x, evals_y)], 
					eval_names =['train','valid'],
					eval_metric='auc',early_stopping_rounds=50)

			print('k_fold',k_fold,label,'dumping...')
			pickle.dump(clf, open(model_file, "wb"))

		score=clf.best_score_['valid']['auc']
		evals_score.loc[k_fold,label]=score
		
		print('k_fold',k_fold,label,'predicting test...')
		test_pred=clf.predict_proba(test_feature,num_iteration=clf.best_iteration_)[:,1]
		submit[label+f'_k_fold{k_fold}']=test_pred
		
		del clf,train_x,evals_x,train_y,evals_y
		gc.collect()
		print('已清理部分内存！！')
		print('k_fold',k_fold,label,'耗时',int(time.time()-s),'s')
	del train_feature,test_feature
	gc.collect()
	print(label,'总耗时',int(time.time()-sj),'s')
print('总耗时',int(time.time()-start_time),'s')
evals_score.to_csv('evals_score_model2.csv',index=False)
score=[]
for col in evals_score.columns.to_list():
    score.append(evals_score[col].mean())
print('线下得分',sum((np.array(score)*np.array([4,3,2,1]))/10))
for label in ys:
    cols=[i for i in submit.columns.to_list() if label+'_k_fold' in i]
    submit[label]=submit[cols].mean(axis=1)
submit[[i for i in submit.columns.to_list() if 'k_fold' not in i]].to_csv('submit_a_model2.csv',index=False)