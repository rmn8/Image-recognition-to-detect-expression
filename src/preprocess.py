
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt





file_arr=[]
for root, dirs, files in os.walk("../IMFDB_final"):
    for file in files:
        if file.endswith(".txt"):
             file_arr.append(os.path.join(root, file))
temp_data = []

for fi in file_arr:
	if os.stat(fi).st_size == 0:
		continue
	data = pd.read_table(str(fi), header = None,error_bad_lines=False)
	if(data.shape[1]!=17 or data[0].dtype=='int64'):
		continue
	
	data.columns=['Frame name','pic1','pic2','X','Y','Z','A','film','year','Actor/Actress','Sex','Expression','No','Res','Age','Profile','Sm']	
	dir_1= '/'.join(fi.split('/')[0:3])
	
	data['path']=data[['film','pic2']].apply(lambda x: dir_1+'/'+'/images/'.join(x),axis=1)
	temp_data.append(data)
	

data_repository=pd.concat(temp_data)

data_repository = data_repository.sample(frac=1).reset_index(drop=True)
data_repository=data_repository[data_repository['path'].apply(lambda x:os.path.exists(x))]

train_set=data_repository[['path','Expression']].iloc[0:int(data_repository.shape[0]*0.7)]
test_set=data_repository[['path','Expression']].iloc[0:int(data_repository.shape[0]*0.3)]

train_set.to_csv('../data/train.txt', index=None)
test_set.to_csv('../data/test.txt', index=None)

