#导入相关库
import numpy as np
import pandas as pd
from datetime import date
import os 


path = os.path.dirname(os.path.realpath(__file__))

#导入数据

#这里用了keep_default_na=False,加载后数据中的缺省值默认是null,大部分是数字字段的数据类型是object即可以看做是字符串，
#当不写这句话时默认缺省值NAN，即大部分是数字字段是float，这也直接导致了怎么判断缺省值的问题：
#当是null时很好说，比如判断date字段时是否是空省值就可用：off_train=='null'
#当时NAN时，需要用函数isnull或者notnull函数来判断

off_train=pd.read_csv(path+'/data/ccf_offline_stage1_train.csv',header=0,keep_default_na=False)
off_train.columns=['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']  #标题小写统一

off_test=pd.read_csv(path+'/data/ccf_offline_stage1_test_revised.csv',header=0,keep_default_na=False)
off_test.columns=['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']

on_train=pd.read_csv(path+'/data/ccf_online_stage1_train.csv',header=0,keep_default_na=False)
on_train.columns=['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']

#划分数据集

#测试集数据
dataset3=off_test

#用于测试集提取特征的数据区间
feature3=off_train[((off_train.date>='20160315')&(off_train.date<='20160630'))|
                   ((off_train.date=='null')&(off_train.date_received>='20160315')&(off_train.date_received<='20160630'))]

#训练集二
dataset2=off_train[(off_train.date_received>='20160515')&(off_train.date_received<='20160615')]

#用于训练集二提取特征的数据区间
feature2=off_train[((off_train.date>='20160201')&(off_train.date<='20160514'))|
                   ((off_train.date=='null')&(off_train.date_received>='20160201')&(off_train.date_received<='20160514'))]

#训练集一
dataset1=off_train[(off_train.date_received>='20160414')&(off_train.date_received<='20160514')]

#用于训练集一提取特征的数据区间
feature1=off_train[((off_train.date>='20160101')&(off_train.date<='20160413'))|
                   ((off_train.date=='null')&(off_train.date_received>='20160101')&(off_train.date_received<='20160413'))]

#去除重复

#原始数据收集中，一般会存在重复项，对后面的特征提取会造成干扰。
#在同一模型参数下，去除重复可以使最后的结果有所提升。
for i in [dataset3,feature3,dataset2,feature2,dataset1,feature1]:
    i.drop_duplicates(inplace=True)
    i.reset_index(drop=True,inplace=True)

#5 other feature
#这部分特征是利用测试集数据提取出的特征，在实际业务中是获取不到的

def get_other_feature(dataset3,filename='other_feature3'):
    
    #用户领取的所有优惠券数目
    t=dataset3[['user_id']]
    t['this_month_user_receive_all_coupon_count']=1
    t=t.groupby('user_id').agg('sum').reset_index()

    #用户领取的特定优惠券数目
    t1=dataset3[['user_id','coupon_id']]
    t1['this_month_user_receive_same_coupon_count']=1
    t1=t1.groupby(['user_id','coupon_id']).agg('sum').reset_index()

    #如果用户领取特定优惠券2次以上，那么提取出第一次和最后一次领取的时间
    t2=dataset3[['user_id','coupon_id','date_received']]
    t2.date_received=t2.date_received.astype('str')
    t2=t2.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
    t2['receive_number']=t2.date_received.apply(lambda s:len(s.split(':')))
    t2=t2[t2.receive_number>1]
    t2['max_date_received']=t2.date_received.apply(lambda s:max([int(d) for d in s.split(':')]))
    t2['min_date_received']=t2.date_received.apply(lambda s:min([int(d) for d in s.split(':')]))
    t2=t2[['user_id','coupon_id','max_date_received','min_date_received']]

    #用户领取特定优惠券的时间，是不是最后一次&第一次
    t3=dataset3[['user_id','coupon_id','date_received']]
    t3=pd.merge(t3,t2,on=['user_id','coupon_id'],how='left')
    t3['this_month_user_receive_same_coupon_lastone']=t3.max_date_received-t3.date_received.astype('int')
    t3['this_month_user_receive_same_coupon_firstone']=t3.date_received.astype('int')-t3.min_date_received
    def is_firstlastone(x):
        if x==0:
            return 1
        elif x>0:
            return 0
        else:
            return -1  #those only receive once

    t3.this_month_user_receive_same_coupon_lastone=t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
    t3.this_month_user_receive_same_coupon_firstone=t3.this_month_user_receive_same_coupon_firstone.apply(is_firstlastone)
    t3=t3[['user_id','coupon_id','date_received','this_month_user_receive_same_coupon_lastone','this_month_user_receive_same_coupon_firstone']]

    #用户在领取优惠券的当天，共领取了多少张优惠券
    t4=dataset3[['user_id','date_received']]
    t4['this_day_user_receive_all_coupon_count']=1
    t4=t4.groupby(['user_id','date_received']).agg('sum').reset_index()

    #用户在领取特定优惠券的当天，共领取了多少张特定的优惠券
    t5=dataset3[['user_id','coupon_id','date_received']]
    t5['this_day_user_receive_same_coupon_count']=1
    t5=t5.groupby(['user_id','coupon_id','date_received']).agg('sum').reset_index()

    #对用户领取特定优惠券的日期进行组合
    t6=dataset3[['user_id','coupon_id','date_received']]
    t6.date_received=t6.date_received.astype('str')
    t6=t6.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
    t6.rename(columns={'date_received':'dates'},inplace=True)

    def get_day_gap_before(s):
        date_received,dates=s.split('-')
        dates=dates.split(':')
        gaps=[]
        for d in dates:
            this_gap=(date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))-
                     date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
            if this_gap>0:
                gaps.append(this_gap)
        if len(gaps)==0:
            return -1
        else:
            return min(gaps)

    def get_day_gap_after(s):
        date_received,dates=s.split('-')
        dates=dates.split(':')
        gaps=[]
        for d in dates:
            this_gap=(date(int(d[0:4]),int(d[4:6]),int(d[6:8]))-
                     date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))).days
            if this_gap>0:
                gaps.append(this_gap)
        if len(gaps)==0:
            return -1
        else:
            return min(gaps)

    #用户领取特定优惠券的当天，与上一次/下一次领取此优惠券的相隔天数
    t7=dataset3[['user_id','coupon_id','date_received']]
    t7=pd.merge(t7,t6,on=['user_id','coupon_id'],how='left')
    t7['date_received_date']=t7.date_received.astype('str')+'-'+t7.dates
    t7['day_gap_before']=t7.date_received_date.apply(get_day_gap_before)
    t7['day_gap_after']=t7.date_received_date.apply(get_day_gap_after)
    t7=t7[['user_id','coupon_id','date_received','day_gap_before','day_gap_after']]

    #上述提取的特征进行合并
    other_feature3=pd.merge(t1,t,on='user_id')
    other_feature3=pd.merge(other_feature3,t3,on=['user_id','coupon_id'])
    other_feature3=pd.merge(other_feature3,t4,on=['user_id','date_received'])
    other_feature3=pd.merge(other_feature3,t5,on=['user_id','coupon_id','date_received'])
    other_feature3=pd.merge(other_feature3,t7,on=['user_id','coupon_id','date_received'])

    #去重；重置索引
    other_feature3.drop_duplicates(inplace=True)
    other_feature3.reset_index(drop=True,inplace=True)
    other_feature3.to_csv(path+'/data_process2/'+filename+'.csv',index=None)
    return other_feature3


#对数据集进行other_feature的提取
other_feature3=get_other_feature(dataset3,filename='other_feature3')
other_feature2=get_other_feature(dataset2,filename='other_feature2')
other_feature1=get_other_feature(dataset1,filename='other_feature1')

#4 coupon related feature
def get_coupon_related_feature(dataset3,filename='coupon3_feature'):
    
    #计算折扣率函数
    def calc_discount_rate(s):
        s=str(s)
        s=s.split(':')
        if len(s)==1:
            return float(s[0])
        else:
            return 1.0-float(s[1])/float(s[0])

    #提取满减优惠券中，满对应的金额
    def get_discount_man(s):
        s=str(s)
        s=s.split(':')
        if len(s)==1:
            return 'null'
        else:
            return int(s[0])

    #提取满减优惠券中，减对应的金额
    def get_discount_jian(s):
        s=str(s)
        s=s.split(':')
        if len(s)==1:
            return 'null'
        else:
            return int(s[1])

    #是不是满减卷
    def is_man_jian(s):
        s=str(s)
        s=s.split(':')
        if len(s)==1:
            return 0
        else:
            return 1.0
    
    #周几领取的优惠券
    dataset3['day_of_week']=dataset3.date_received.astype('str').apply(lambda x:date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1)
    
    #每月的第几天领取的优惠券
    dataset3['day_of_month']=dataset3.date_received.astype('str').apply(lambda x:int(x[6:8]))
    
    #领取优惠券的时间与当月初距离多少天
    dataset3['days_distance']=dataset3.date_received.astype('str').apply(lambda x:(date(int(x[0:4]),int(x[4:6]),int(x[6:8]))-date(2016,6,30)).days)

    #满减优惠券中，满对应的金额
    dataset3['discount_man']=dataset3.discount_rate.apply(get_discount_man)
    
    #满减优惠券中，减对应的金额
    dataset3['discount_jian']=dataset3.discount_rate.apply(get_discount_jian)
    
    #优惠券是不是满减卷
    dataset3['is_man_jian']=dataset3.discount_rate.apply(is_man_jian)
    
    #优惠券的折扣率（满减卷进行折扣率转换）
    dataset3['discount_rate']=dataset3.discount_rate.apply(calc_discount_rate)

    #特定优惠券的总数量
    d=dataset3[['coupon_id']]
    d['coupon_count']=1
    d=d.groupby('coupon_id').agg('sum').reset_index()
    dataset3=pd.merge(dataset3,d,on='coupon_id',how='left')

    dataset3.to_csv(path+'/data_process2/'+filename+'.csv',index=None)
    return dataset3


#对数据集进行coupon_related_feature的提取
coupon3_feature=get_coupon_related_feature(dataset3,filename='coupon3_feature')
coupon2_feature=get_coupon_related_feature(dataset2,filename='coupon2_feature')
coupon1_feature=get_coupon_related_feature(dataset1,filename='coupon1_feature')

#3 merchant related feature
#这部分特征是在特征数据集中提取
def get_merchant_related_feature(feature3,filename='merchant3_feature'):
    
    merchant3=feature3[['merchant_id','coupon_id','distance','date_received','date']]

    #提取不重复的商户集合
    t=merchant3[['merchant_id']]
    t.drop_duplicates(inplace=True)

    #商户的总销售次数
    t1=merchant3[merchant3.date!='null'][['merchant_id']]
    t1['total_sales']=1
    t1=t1.groupby('merchant_id').agg('sum').reset_index()

    #商户被核销优惠券的销售次数
    t2=merchant3[(merchant3.date!='null')&(merchant3.coupon_id!='null')][['merchant_id']]
    t2['sales_use_coupon']=1
    t2=t2.groupby('merchant_id').agg('sum').reset_index()

    #商户发行优惠券的总数
    t3=merchant3[merchant3.coupon_id!='null'][['merchant_id']]
    t3['total_coupon']=1
    t3=t3.groupby('merchant_id').agg('sum').reset_index()

    #商户被核销优惠券的用户-商户距离，转化为int数值类型
    t4=merchant3[(merchant3.date!='null')&(merchant3.coupon_id!='null')][['merchant_id','distance']]
    t4.replace('null',-1,inplace=True)
    t4.distance=t4.distance.astype('int')
    t4.replace(-1,np.nan,inplace=True)

    #商户被核销优惠券的最小用户-商户距离
    t5=t4.groupby('merchant_id').agg('min').reset_index()
    t5.rename(columns={'distance':'merchant_min_distance'},inplace=True)

    #商户被核销优惠券的最大用户-商户距离
    t6=t4.groupby('merchant_id').agg('max').reset_index()
    t6.rename(columns={'distance':'merchant_max_distance'},inplace=True)

    #商户被核销优惠券的平均用户-商户距离
    t7=t4.groupby('merchant_id').agg('mean').reset_index()
    t7.rename(columns={'distance':'merchant_mean_distance'},inplace=True)

    #商户被核销优惠券的用户-商户距离的中位数
    t8=t4.groupby('merchant_id').agg('median').reset_index()
    t8.rename(columns={'distance':'merchant_median_distance'},inplace=True)

    #合并上述特征
    merchant3_feature=pd.merge(t,t1,on='merchant_id',how='left')
    merchant3_feature=pd.merge(merchant3_feature,t2,on='merchant_id',how='left')
    merchant3_feature=pd.merge(merchant3_feature,t3,on='merchant_id',how='left')
    merchant3_feature=pd.merge(merchant3_feature,t5,on='merchant_id',how='left')
    merchant3_feature=pd.merge(merchant3_feature,t6,on='merchant_id',how='left')
    merchant3_feature=pd.merge(merchant3_feature,t7,on='merchant_id',how='left')
    merchant3_feature=pd.merge(merchant3_feature,t8,on='merchant_id',how='left')

    #商户被核销优惠券的销售次数，如果为空，填充为0
    merchant3_feature.sales_use_coupon=merchant3_feature.sales_use_coupon.replace(np.nan,0)
    
    #商户发行优惠券的转化率
    merchant3_feature['merchant_coupon_transfer_rate']=merchant3_feature.sales_use_coupon.astype('float')/merchant3_feature.total_coupon
    
    #商户被核销优惠券的销售次数占比
    merchant3_feature['coupon_rate']=merchant3_feature.sales_use_coupon.astype('float')/merchant3_feature.total_sales
    merchant3_feature.total_coupon=merchant3_feature.total_coupon.replace(np.nan,0)

    merchant3_feature.to_csv(path+'/data_process2/'+filename+'.csv',index=None)
    return merchant3_feature

#对特征数据集进行merchant_related_feature的提取
merchant3_feature=get_merchant_related_feature(feature3,filename='merchant3_feature')
merchant2_feature=get_merchant_related_feature(feature2,filename='merchant2_feature')
merchant1_feature=get_merchant_related_feature(feature1,filename='merchant1_feature')

#2 user related feature
def get_user_related_feature(feature3,filename='user3_feature'):
    
    #用户核销优惠券与领取优惠券日期间隔
    def get_user_date_datereceived_gap(s):
        s=s.split(':')
        return (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8]))-
               date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days

    user3=feature3[['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']]

    #提取不重复的所有用户集合
    t=user3[['user_id']]
    t.drop_duplicates(inplace=True)

    #用户在特定商户的消费次数
    t1=user3[user3.date!='null'][['user_id','merchant_id']]
    t1.drop_duplicates(inplace=True)
    t1.merchant_id=1
    t1=t1.groupby('user_id').agg('sum').reset_index()
    t1.rename(columns={'merchant_id':'count_merchant'},inplace=True)

    #提取用户核销优惠券的用户-商户距离
    t2=user3[(user3.date!='null')&(user3.coupon_id!='null')][['user_id','distance']]
    t2.replace('null',-1,inplace=True)
    t2.distance=t2.distance.astype('int')
    t2.replace(-1,np.nan,inplace=True)
    
    #用户核销优惠券中的最小用户-商户距离
    t3=t2.groupby('user_id').agg('min').reset_index()
    t3.rename(columns={'distance':'user_min_distance'},inplace=True)

    #用户核销优惠券中的最大用户-商户距离
    t4=t2.groupby('user_id').agg('max').reset_index()
    t4.rename(columns={'distance':'user_max_distance'},inplace=True)

    #用户核销优惠券中的平均用户-商户距离
    t5=t2.groupby('user_id').agg('mean').reset_index()
    t5.rename(columns={'distance':'user_mean_distance'},inplace=True)

    #用户核销优惠券的用户-商户距离的中位数
    t6=t2.groupby('user_id').agg('median').reset_index()
    t6.rename(columns={'distance':'user_median_distance'},inplace=True)

    #用户核销优惠券的总次数
    t7=user3[(user3.date!='null')&(user3.coupon_id!='null')][['user_id']]
    t7['buy_use_coupon']=1
    t7=t7.groupby('user_id').agg('sum').reset_index()

    #用户购买的总次数
    t8=user3[user3.date!='null'][['user_id']]
    t8['buy_total']=1
    t8=t8.groupby('user_id').agg('sum').reset_index()

    #用户领取优惠券的总次数
    t9=user3[user3.coupon_id!='null'][['user_id']]
    t9['coupon_received']=1
    t9=t9.groupby('user_id').agg('sum').reset_index()

    #用户核销优惠券与领取优惠券的日期间隔
    t10=user3[(user3.date_received!='null')&(user3.date!='null')][['user_id','date_received','date']]
    t10['user_date_datereceived_gap']=t10.date+':'+t10.date_received
    t10.user_date_datereceived_gap=t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
    t10=t10[['user_id','user_date_datereceived_gap']]

    #用户核销优惠券与领取优惠券的日期间隔的平均值
    t11=t10.groupby('user_id').agg('mean').reset_index()
    t11.rename(columns={'user_date_datereceived_gap':'avg_user_date_datereceived_gap'},inplace=True)
    
    #用户核销优惠券与领取优惠券的日期间隔的最小值
    t12=t10.groupby('user_id').agg('min').reset_index()
    t12.rename(columns={'user_date_datereceived_gap':'min_user_date_datereceived_gap'},inplace=True)
    
    #用户核销优惠券与领取优惠券的日期间隔的最大值
    t13=t10.groupby('user_id').agg('max').reset_index()
    t13.rename(columns={'user_date_datereceived_gap':'max_user_date_datereceived_gap'},inplace=True)

    #合并上述特征
    user3_feature=pd.merge(t,t1,on='user_id',how='left')
    user3_feature=pd.merge(user3_feature,t3,on='user_id',how='left')
    user3_feature=pd.merge(user3_feature,t4,on='user_id',how='left')
    user3_feature=pd.merge(user3_feature,t5,on='user_id',how='left')
    user3_feature=pd.merge(user3_feature,t6,on='user_id',how='left')
    user3_feature=pd.merge(user3_feature,t7,on='user_id',how='left')
    user3_feature=pd.merge(user3_feature,t8,on='user_id',how='left')
    user3_feature=pd.merge(user3_feature,t9,on='user_id',how='left')
    user3_feature=pd.merge(user3_feature,t10,on='user_id',how='left')
    user3_feature=pd.merge(user3_feature,t11,on='user_id',how='left')
    user3_feature=pd.merge(user3_feature,t12,on='user_id',how='left')
    user3_feature=pd.merge(user3_feature,t13,on='user_id',how='left')

    #特征缺失值填充
    user3_feature.count_merchant=user3_feature.count_merchant.replace(np.nan,0)
    user3_feature.buy_use_coupon=user3_feature.buy_use_coupon.replace(np.nan,0)

    #用户核销优惠券消费次数占用户总消费次数的比例
    user3_feature['buy_use_coupon_rate']=user3_feature.buy_use_coupon.astype('float')/user3_feature.buy_total.astype('float')
    
    #用户核销优惠券消费次数占用户领取优惠券次数的比例
    user3_feature['user_coupon_transfer_rate']=user3_feature.buy_use_coupon.astype('float')/user3_feature.coupon_received.astype('float')

    #特征缺失值填充
    user3_feature.buy_total=user3_feature.buy_total.replace(np.nan,0)
    user3_feature.coupon_received=user3_feature.coupon_received.replace(np.nan,0)

    user3_feature.to_csv(path+'/data_process2/'+filename+'.csv',index=None)
    return user3_feature

#对特征数据集进行user_related_feature的提取
user3_feature=get_user_related_feature(feature3,filename='user3_feature')
user2_feature=get_user_related_feature(feature2,filename='user2_feature')
user1_feature=get_user_related_feature(feature1,filename='user1_feature')

#1 user_merchant related feature
def get_user_merchant_related_feature(feature3,filename='user_merchant3'):
    
    #提取用户-商户交叉集合
    all_user_merchant=feature3[['user_id','merchant_id']]
    all_user_merchant.drop_duplicates(inplace=True)

    #用户在特定商户下的消费次数
    t=feature3[['user_id','merchant_id','date']]
    t=t[t.date!='null'][['user_id','merchant_id']]
    t['user_merchant_buy_total']=1
    t=t.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    t.drop_duplicates(inplace=True)

    #用户在特定商户处领取优惠券次数
    t1=feature3[['user_id','merchant_id','coupon_id']]
    t1=t1[t1.coupon_id!='null'][['user_id','merchant_id']]
    t1['user_merchant_received']=1
    t1=t1.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    t1.drop_duplicates(inplace=True)

    #用户在特定商户处核销优惠券的次数
    t2=feature3[['user_id','merchant_id','date','date_received']]
    t2=t2[(t2.date!='null')&(t2.date_received!='null')][['user_id','merchant_id']]
    t2['user_merchant_buy_use_coupon']=1
    t2=t2.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    t2.drop_duplicates(inplace=True)

    #用户在特定商户处发生行为的总次数
    t3=feature3[['user_id','merchant_id']]
    t3['user_merchant_any']=1
    t3=t3.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    t3.drop_duplicates(inplace=True)

    #用户在特定商户处未领取优惠券产生的消费次数
    t4=feature3[['user_id','merchant_id','date','coupon_id']]
    t4=t4[(t4.date!='null')&(t4.coupon_id=='null')][['user_id','merchant_id']]
    t4['user_merchant_buy_common']=1
    t4=t4.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    t4.drop_duplicates(inplace=True)

    #合并上述特征
    user_merchant3=pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
    user_merchant3=pd.merge(user_merchant3,t1,on=['user_id','merchant_id'],how='left')
    user_merchant3=pd.merge(user_merchant3,t2,on=['user_id','merchant_id'],how='left')
    user_merchant3=pd.merge(user_merchant3,t3,on=['user_id','merchant_id'],how='left')
    user_merchant3=pd.merge(user_merchant3,t4,on=['user_id','merchant_id'],how='left')

    #相关特征缺失值填充
    user_merchant3.user_merchant_buy_use_coupon=user_merchant3.user_merchant_buy_use_coupon.replace(np.nan,0)
    user_merchant3.user_merchant_buy_common=user_merchant3.user_merchant_buy_common.replace(np.nan,0)

    #用户在特定商户处核销优惠券占领取优惠券数量的比例
    user_merchant3['user_merchant_coupon_transfer_rate']=user_merchant3.user_merchant_buy_use_coupon.astype('float')/user_merchant3.user_merchant_received.astype('float')

    #用户在特定商户处核销优惠券占购买次数的比例
    user_merchant3['user_merchant_coupon_buy_rate']=user_merchant3.user_merchant_buy_use_coupon.astype('float')/user_merchant3.user_merchant_buy_total.astype('float')

    #用户在特定商户处购买次数占发生行为次数的比例
    user_merchant3['user_merchant_rate']=user_merchant3.user_merchant_buy_total.astype('float')/user_merchant3.user_merchant_any.astype('float')

    #用户在特定商户下未用优惠券购买占购买次数的占比
    user_merchant3['user_merchant_common_buy_rate']=user_merchant3.user_merchant_buy_common.astype('float')/user_merchant3.user_merchant_buy_total.astype('float')

    user_merchant3.to_csv(path+'/data_process2/'+filename+'.csv',index=None)
    return user_merchant3

#对特征数据集进行user_merchant_related_feature的提取
user_merchant3=get_user_merchant_related_feature(feature3,filename='user_merchant3')
user_merchant2=get_user_merchant_related_feature(feature2,filename='user_merchant2')
user_merchant1=get_user_merchant_related_feature(feature1,filename='user_merchant1')

#generate training and testing set

#提取题目要求的标签：15天内核销
def get_label(s):
    s=s.split(':')
    if s[0]=='nan':
        return 0
    elif (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8]))-
          date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days<=15:
        return 1
    else:
        return -1

##for dataset3

#提取相关特征
coupon3=pd.read_csv(path+'/data_process2/coupon3_feature.csv')
merchant3=pd.read_csv(path+'/data_process2/merchant3_feature.csv')
user3=pd.read_csv(path+'/data_process2/user3_feature.csv')
user_merchant3=pd.read_csv(path+'/data_process2/user_merchant3.csv')
other_feature3=pd.read_csv(path+'/data_process2/other_feature3.csv')

#合并相关特征
dataset3=pd.merge(coupon3,merchant3,on='merchant_id',how='left')
dataset3=pd.merge(dataset3,user3,on='user_id',how='left')
dataset3=pd.merge(dataset3,user_merchant3,on=['user_id','merchant_id'],how='left')
dataset3=pd.merge(dataset3,other_feature3,on=['user_id','coupon_id','date_received'],how='left')
dataset3.drop_duplicates(inplace=True)

#相关特征缺失值填充
dataset3.user_merchant_buy_total=dataset3.user_merchant_buy_total.replace(np.nan,0)
dataset3.user_merchant_any=dataset3.user_merchant_any.replace(np.nan,0)
dataset3.user_merchant_received=dataset3.user_merchant_received.replace(np.nan,0)

#用户领取优惠券日期是否在周末
dataset3['is_weekend']=dataset3.day_of_week.apply(lambda x:1 if x in (6,7) else 0)

#对优惠券领取日期进行ont-hot编码
weekday_dummies=pd.get_dummies(dataset3.day_of_week)
weekday_dummies.columns=['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset3=pd.concat([dataset3,weekday_dummies],axis=1)

#删除相关特征，这里coupon_count应该是在后面根据模型进行特征筛选来踢出一些不太相关或者容易导致过拟合的特征
dataset3.drop(['merchant_id','day_of_week','coupon_count'],axis=1,inplace=True)
dataset3=dataset3.replace('null',np.nan)

dataset3.to_csv(path+'/data_process2/dataset3.csv',index=None)

##for dataset2
#提取相关特征
coupon2=pd.read_csv(path+'/data_process2/coupon2_feature.csv')
merchant2=pd.read_csv(path+'/data_process2/merchant2_feature.csv')
user2=pd.read_csv(path+'/data_process2/user2_feature.csv')
user_merchant2=pd.read_csv(path+'/data_process2/user_merchant2.csv')
other_feature2=pd.read_csv(path+'/data_process2/other_feature2.csv')

#合并相关特征
dataset2=pd.merge(coupon2,merchant2,on='merchant_id',how='left')
dataset2=pd.merge(dataset2,user2,on='user_id',how='left')
dataset2=pd.merge(dataset2,user_merchant2,on=['user_id','merchant_id'],how='left')
dataset2=pd.merge(dataset2,other_feature2,on=['user_id','coupon_id','date_received'],how='left')
dataset2.drop_duplicates(inplace=True)

#处理基本与上述dataset3一致，这里特殊有一部添加需要的label
dataset2.user_merchant_buy_total=dataset2.user_merchant_buy_total.replace(np.nan,0)
dataset2.user_merchant_any=dataset2.user_merchant_any.replace(np.nan,0)
dataset2.user_merchant_received=dataset2.user_merchant_received.replace(np.nan,0)

dataset2['is_weekend']=dataset2.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies=pd.get_dummies(dataset2.day_of_week)
weekday_dummies.columns=['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset2=pd.concat([dataset2,weekday_dummies],axis=1)

dataset2['label']=dataset2.date.astype('str')+':'+dataset2.date_received.astype('str')
dataset2.label=dataset2.label.apply(get_label)

dataset2.drop(['merchant_id','day_of_week','date','date_received','coupon_id','coupon_count'],axis=1,inplace=True)

dataset2=dataset2.replace('null',np.nan)
dataset2=dataset2.replace('nan',np.nan)

dataset2.to_csv(path+'/data_process2/dataset2.csv',index=None)

##for dataset1
#与上述dataset2的处理流程一致
coupon1=pd.read_csv(path+'/data_process2/coupon1_feature.csv')
merchant1=pd.read_csv(path+'/data_process2/merchant1_feature.csv')
user1=pd.read_csv(path+'/data_process2/user1_feature.csv')
user_merchant1=pd.read_csv(path+'/data_process2/user_merchant1.csv')
other_feature1=pd.read_csv(path+'/data_process2/other_feature1.csv')

dataset1=pd.merge(coupon1,merchant1,on='merchant_id',how='left')
dataset1=pd.merge(dataset1,user1,on='user_id',how='left')
dataset1=pd.merge(dataset1,user_merchant1,on=['user_id','merchant_id'],how='left')
dataset1=pd.merge(dataset1,other_feature1,on=['user_id','coupon_id','date_received'],how='left')
dataset1.drop_duplicates(inplace=True)

dataset1.user_merchant_buy_total=dataset1.user_merchant_buy_total.replace(np.nan,0)
dataset1.user_merchant_any=dataset1.user_merchant_any.replace(np.nan,0)
dataset1.user_merchant_received=dataset1.user_merchant_received.replace(np.nan,0)

dataset1['is_weekend']=dataset1.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies=pd.get_dummies(dataset1.day_of_week)
weekday_dummies.columns=['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset1=pd.concat([dataset1,weekday_dummies],axis=1)

dataset1['label']=dataset1.date.astype('str')+':'+dataset1.date_received.astype('str')
dataset1.label=dataset1.label.apply(get_label)

dataset1.drop(['merchant_id','day_of_week','date','date_received','coupon_id','coupon_count'],axis=1,inplace=True)

dataset1=dataset1.replace('null',np.nan)
dataset1=dataset1.replace('nan',np.nan)

dataset1.to_csv(path+'/data_process2/dataset1.csv',index=None)
