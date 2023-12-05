import datetime
import os
import time
from concurrent.futures import ProcessPoolExecutor
from math import ceil

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
#from sklearn.externals import joblib
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import lightgbm as lgb 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
import os 


path = os.path.dirname(os.path.realpath(__file__))


def get_processed_data():
    dataset1 = pd.read_csv(path+'/data_preprocessed/ProcessDataSet1.csv')
    dataset2 = pd.read_csv(path+'/data_preprocessed/ProcessDataSet2.csv')
    dataset3 = pd.read_csv(path+'/data_preprocessed/ProcessDataSet3.csv')

    dataset1.drop_duplicates(inplace=True)
    dataset2.drop_duplicates(inplace=True)
    dataset3.drop_duplicates(inplace=True)

    dataset12 = pd.concat([dataset1, dataset2], axis=0)

    dataset12.fillna(0, inplace=True)
    dataset3.fillna(0, inplace=True)

    return dataset12, dataset3


def train_lightgbm(dataset12, dataset3):
    predict_dataset = dataset3[['User_id', 'Coupon_id', 'Date_received']].copy()
    predict_dataset.Date_received = pd.to_datetime(predict_dataset.Date_received, format='%Y-%m-%d')
    predict_dataset.Date_received = predict_dataset.Date_received.dt.strftime('%Y%m%d')

    # 将数据转化为dmatric格式
    dataset12_x = dataset12.drop(
        columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
                 'Date', 'Coupon_id', 'label'], axis=1)
    dataset3_x = dataset3.drop(
        columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
                 'Coupon_id'], axis=1)

    train_dmatrix = lgb.Dataset(dataset12_x, label=dataset12.label)
    predict_dmatrix =dataset3_x# lgb.Dataset(dataset3_x)

    # lightgbm模型训练
    params = {    
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'nthread':4,
          'learning_rate':0.1,
          'num_leaves':30, 
          'max_depth': 5,   
          'subsample': 0.8, 
          'colsample_bytree': 0.8, 
    }
    ###
    from lightgbm import log_evaluation, early_stopping
    callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=30)]

    # 使用xgb.cv优化num_boost_round参数
    cvresult = lgb.cv(params, train_dmatrix, num_boost_round=10000, nfold=2, stratified=False, shuffle=True, metrics='auc',seed=0,callbacks=callbacks)
    print(cvresult)
    num_round_best = len(cvresult['auc-mean'])
    print('Best round num:', len(cvresult['auc-mean']))
    #print('Best round num: ', num_round_best)

    # 使用优化后的num_boost_round参数训练模型
    #watchlist = [(train_dmatrix, 'train')]
    model = lgb.train(params, train_dmatrix)#, num_boost_round=1000)#, evals=watchlist)

    #model.save_model(path+'/train_dir/lightgbmmodel')
    params['predictor'] = 'cpu_predictor'
    #model = lgb.Booster(model_file=path+'/train_dir/lightgbmmodel')
    #model.load_model(path+'/train_dir/lightgbmmodel')

    # predict test set
    dataset3_predict = predict_dataset.copy()
    dataset3_predict['label'] = model.predict(predict_dmatrix)

    # 标签归一化
    dataset3_predict.label = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(
        dataset3_predict.label.values.reshape(-1, 1))
    dataset3_predict.sort_values(by=['Coupon_id', 'label'], inplace=True)
    dataset3_predict.to_csv(path+'/result/result_finally_lightgbm_preds2.csv', index=None, header=None)
    print(dataset3_predict.describe())



    temp = dataset12[['Coupon_id', 'label']].copy()
    temp['pred'] = model.predict(dataset12_x)#lgb.Dataset(dataset12_x))
    temp.pred = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(temp['pred'].values.reshape(-1, 1))
    print(myauc(temp))


# 性能评价函数
def myauc(test):
    testgroup = test.groupby(['Coupon_id'])
    aucs = []
    for i in testgroup:
        tmpdf = i[1]
        if len(tmpdf['label'].unique()) != 2:
            continue
        fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred'], pos_label=1)
        aucs.append(auc(fpr, tpr))
    return np.average(aucs)


if __name__ == '__main__':
    start = datetime.datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))
    # log = '%s\n' % start.strftime('%Y-%m-%d %H:%M:%S')
    cpu_jobs = os.cpu_count() - 1
    date_null = pd.to_datetime('1970-01-01', format='%Y-%m-%d')

    dataset12, dataset3 = get_processed_data()
    # analysis()
    # detect_duplicate_columns()
    # feature_importance_score()

    # grid_search_gbdt()
    # train_gbdt()
    # predict('gbdt')

    # grid_search_xgb()
    train_lightgbm(dataset12, dataset3)

    # print('predict: start predicting......')
    # # predict('xgb')
    # print('predict: predicting finished.')

    # log += 'time: %s\n' % str((datetime.datetime.now() - start)).split('.')[0]
    # log += '----------------------------------------------------\n'
    # open('%s.log' % os.path.basename(__file__), 'a').write(log)
    # print(log)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %s s' % (datetime.datetime.now() - start).seconds)