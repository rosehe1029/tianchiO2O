#xgboost模型里，对于缺失值有自己的处理方式。
#所以用xgboost不需要把所有缺失值全部进行处理，但对于一些业务角度很容易填补的缺失值还是建议填充
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import os 
import numpy as np 
cpu_jobs = os.cpu_count() - 1

path = os.path.dirname(os.path.realpath(__file__))

dataset1=pd.read_csv(path+'/data_process2/dataset1.csv')
dataset1.label.replace(-1,0,inplace=True)
dataset2=pd.read_csv(path+'/data_process2/dataset2.csv')
dataset2.label.replace(-1,0,inplace=True)
dataset3=pd.read_csv(path+'/data_process2/dataset3.csv')

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)

#将训练集一和训练集二合并，作为调参后的总训练数据集
dataset12=pd.concat([dataset1,dataset2],axis=0)

#这里删除了两个特征day_gap_before和day_gap_after，是由于这两个特征容易导致过拟合
#我们也可以不删除，跑模型调参试试，再根据效果进行特征的筛选
dataset1_y=dataset1.label
dataset1_x=dataset1.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)
dataset2_y=dataset2.label
dataset2_x=dataset2.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)
dataset12_y=dataset12.label
dataset12_x=dataset12.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)
dataset3_preds=dataset3[['user_id','coupon_id','date_received']]
dataset3_x=dataset3.drop(['user_id','coupon_id','date_received','day_gap_before','day_gap_after'],axis=1)

print(dataset1_x.shape,dataset2_x.shape,dataset3_x.shape,dataset12_x.shape)

#转换为xgb需要的数据类型
dataset1=xgb.DMatrix(dataset1_x,label=dataset1_y)
dataset2=xgb.DMatrix(dataset2_x,label=dataset2_y)
dataset12=xgb.DMatrix(dataset12_x,label=dataset12_y)
dataset3=xgb.DMatrix(dataset3_x)

'''
params={'booster':'gbtree',
       'objective':'rank:pairwise',
       'eval_metric':'auc',
       'gamma':0.1,
       'min_child_weight':1.1,
       'max_depth':5,
       'lambda':10,
       'subsample':0.7,
       'colsample_bytree':0.7,
       'colsample_bylevel':0.7,
       'eta':0.01,
       'tree_method':'exact',
       'seed':0,
       'nthread':12}
'''  

params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'gamma': 0.1,
              'min_child_weight': 1.1,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.01,
            #   'tree_method': 'gpu_hist',
            #   'n_gpus': '-1',
              'seed': 0,
              'nthread': cpu_jobs,
            #   'predictor': 'gpu_predictor'
              }

# 使用xgb.cv优化num_boost_round参数
cvresult = xgb.cv(params, dataset12, num_boost_round=10000, nfold=2, metrics='auc', seed=0, #callbacks=[
       #xgb.callback.print_evaluation(show_stdv=False),nfold=2
       #xgb.callback.early_stop(50)
       early_stopping_rounds=50,)
num_round_best = cvresult.shape[0] - 1
print('Best round num: ', num_round_best)


#训练模型
watchlist=[(dataset12,'train')]
model=xgb.train(params,dataset12,num_boost_round=num_round_best,evals=watchlist)

#predict test set
dataset3_preds['label']=model.predict(dataset3)
dataset3_preds.label=MinMaxScaler().fit_transform(np.array(dataset3_preds.label).reshape(-1,1))
dataset3_preds.sort_values(by=['coupon_id','label'],inplace=True)

dataset3_preds.describe()

dataset3_preds.to_csv(path+'/result/xgb_preds4.csv',index=None,header=None)

