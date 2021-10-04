import pandas as pd

submit_a_model1=pd.read_csv('submit_a_model1.csv')
submit_a_model2=pd.read_csv('submit_a_model2.csv')
submit_a=submit_model1[['userid','feedid']].copy()
for col in submit_a_model1.columns.to_list()[2:]:
    submit_a[col]=(submit_a_model1[col]+submit_a_model2[col])/2
submit_a.to_csv('submit_a.csv',index=False)