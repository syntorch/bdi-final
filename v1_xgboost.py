import numpy as np
import pandas as pd
import xgboost as xgb
from pandas import DataFrame
from utils import score, get_train_dataset

def time_feature(df):
    df['time']=pd.to_datetime(df['time'])
    df['month']=df.time.dt.month
    df['hour']=df.time.dt.hour
    df['weekday']=df.time.dt.weekday
    return df

def user_portrait(df:DataFrame):
    uid_and_commentCount=df.groupby('uid')['comment_count'].sum().reset_index(name='uid_comment_count')
    uid_and_commentMean=df.groupby('uid')['comment_count'].mean().reset_index(name='uid_comment_mean')
    uid_and_commentMax=df.groupby('uid')['comment_count'].max().reset_index(name='uid_comment_max')

    uid_and_likeCount=df.groupby('uid')['like_count'].sum().reset_index(name='uid_like_count')
    uid_and_likeMean=df.groupby('uid')['like_count'].mean().reset_index(name='uid_like_mean')
    uid_and_likeMax=df.groupby('uid')['like_count'].max().reset_index(name='uid_like_max')

    uid_and_forwardCount=df.groupby('uid')['forward_count'].sum().reset_index(name='uid_forward_count')
    uid_and_forwardMean=df.groupby('uid')['forward_count'].mean().reset_index(name='uid_forward_mean')
    uid_and_forwardMax=df.groupby('uid')['forward_count'].max().reset_index(name='uid_forward_max')
    
    df = (
        uid_and_commentCount
        .merge(uid_and_commentMean, on='uid')
        .merge(uid_and_commentMax, on='uid')
        .merge(uid_and_likeCount, on='uid')
        .merge(uid_and_likeMean, on='uid')
        .merge(uid_and_likeMax, on='uid')
        .merge(uid_and_forwardCount, on='uid')
        .merge(uid_and_forwardMean, on='uid')
        .merge(uid_and_forwardMax, on='uid')
    )
    return df

if '__main__' == __name__:
    train_data_path = 'data/weibo_split_train_data.txt'
    eval_data_path = 'data/weibo_eval_data.txt'
    raw_train_data = get_train_dataset(train_data_path=train_data_path)
    raw_eval_data = get_train_dataset(train_data_path=eval_data_path)

    train_data = time_feature(raw_train_data)
    eval_data = time_feature(raw_eval_data)
    # 用户feature table
    user_feature = user_portrait(train_data)
    train_data = train_data.merge(user_feature, on='uid', how='left').fillna(-1)
    eval_data = eval_data.merge(user_feature, on='uid', how='left').fillna(-1)
    # 删除
    train_data = train_data.drop(['uid','mid','time'],axis=1)
    eval_data = eval_data.drop(['uid','mid','time'],axis=1)
    # 暂时不考虑content
    train_data = train_data.drop(['content'],axis=1)
    eval_data = eval_data.drop(['content'],axis=1)

    # 训练
    y_train=train_data.loc[:,['forward_count','comment_count','like_count']]
    X_train=train_data.drop(['forward_count','comment_count','like_count'],axis=1)

    y_test=eval_data.loc[:,['forward_count','comment_count','like_count']]
    X_test=eval_data.drop(['forward_count','comment_count','like_count'],axis=1)

    model_xgb = xgb.XGBRegressor()
    model_xgb.fit(X_train,y_train)
    xgb_pred=model_xgb.predict(X_test)

    xgb_pred = np.round(xgb_pred)
    y_test = y_test.to_numpy()
    print('正确率:',score(y_test,xgb_pred))
