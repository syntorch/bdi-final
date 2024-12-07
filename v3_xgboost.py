import numpy as np
import pandas as pd
import xgboost as xgb
from pandas import DataFrame
from utils import score, get_train_dataset
import re

holiday_2015 = pd.DataFrame([
    '2015-1-1', '2015-2-18' '2015-2-19', '2015-2-20', '2015-2-21', '2015-2-22','2015-2-23','2015-2-24',
    '2015-4-3', '2015-4-4', '2015-4-5', '2015-5-1', '2015-6-20', '2015-9-27',
    '2015-10-1', '2015-10-2', '2015-10-3', '2015-10-4', '2015-10-5', '2015-10-6', '2015-10-7'
])

def time_feature(df):
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df.time.dt.month
    df['hour'] = df.time.dt.hour
    df['weekday'] = df.time.dt.weekday
    df['is_holiday'] = df.time.dt.date.isin(holiday_2015)
    return df

def user_portrait(df:DataFrame):
    uid_comment_count=df.groupby('uid')['comment_count'].sum().reset_index(name='uid_comment_count')
    uid_comment_mean=df.groupby('uid')['comment_count'].mean().reset_index(name='uid_comment_mean')
    uid_comment_max=df.groupby('uid')['comment_count'].max().reset_index(name='uid_comment_max')

    uid_like_count=df.groupby('uid')['like_count'].sum().reset_index(name='uid_like_count')
    uid_like_mean=df.groupby('uid')['like_count'].mean().reset_index(name='uid_like_mean')
    uid_like_max=df.groupby('uid')['like_count'].max().reset_index(name='uid_like_max')

    uid_forward_count=df.groupby('uid')['forward_count'].sum().reset_index(name='uid_forward_count')
    uid_forward_mean=df.groupby('uid')['forward_count'].mean().reset_index(name='uid_forward_mean')
    uid_forward_max=df.groupby('uid')['forward_count'].max().reset_index(name='uid_forward_max')

    uid_post_count = df.groupby('uid')['mid'].count().reset_index(name='uid_post_count')
    uid_total_content_len = df.groupby('uid')['content_length'].sum().reset_index(name='uid_total_content_len')
    uid_avg_content_len = df.groupby('uid')['content_length'].mean().reset_index(name='uid_avg_content_len')

    # month_post_counts = df.groupby(['uid', 'month']).size().reset_index(name='month_post_num')
    post_counts = df.groupby(['uid', 'month']).size().reset_index(name='post_count')
    pivot_table = post_counts.pivot(index='uid', columns='month', values='post_count').fillna(0).astype(int)
    pivot_table.columns = [f'Month_{col}_post_num' for col in pivot_table.columns]

    df = (
        uid_comment_count
        .merge(uid_comment_mean, on='uid')
        .merge(uid_comment_max, on='uid')
        .merge(uid_like_count, on='uid')
        .merge(uid_like_mean, on='uid')
        .merge(uid_like_max, on='uid')
        .merge(uid_forward_count, on='uid')
        .merge(uid_forward_mean, on='uid')
        .merge(uid_forward_max, on='uid')
        .merge(uid_post_count, on='uid')
        .merge(uid_total_content_len, on='uid')
        .merge(uid_avg_content_len, on='uid')
        .merge(pivot_table, on='uid')
    )

    # def calculate_slope(row):
    #     y = row[pivot_table.columns].values.astype(int)
    #     slope, _ = np.polyfit(range(1, len(pivot_table.columns)+1), y, 1)
    #     return 1 if slope >= 0 else 0
    # df['user_post_slope'] = df.apply(calculate_slope, axis=1)

    return df

def post_feature(df:DataFrame):
    def count_url(text):
        url_pattern = re.compile(
            r'((https?:\/\/)?([a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,}(:\d+)?(\/[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]*)?)'
        )
        return len(url_pattern.findall(text))
    
    def has_tag(text):
        tag_pattern = re.compile(r'#([\u4e00-\u9fa5a-zA-Z0-9_]+)#')
        return bool(tag_pattern.search(text))
    
    def has_bracket(text):
        tag_pattern = re.compile(r'【([\u4e00-\u9fa5a-zA-Z0-9_]+)】')
        return bool(tag_pattern.search(text))
    
    def has_at(text):
        at_pattern = re.compile(r'@([\u4e00-\u9fa5a-zA-Z0-9_]+)')
        return bool(at_pattern.search(text))

    df['content'] = df['content'].apply(lambda x: x.encode('utf-8').decode('utf-8') if isinstance(x, str) else '')
    df['content_length'] = df['content'].apply(len)
    df['url_num'] = df['content'].apply(count_url)
    df['has_tag'] = df['content'].apply(has_tag)
    df['has_at'] = df['content'].apply(has_at)
    df['has_bracket'] = df['content'].apply(has_bracket)

    return df

def preprocess(df:DataFrame):
    '''
    use (uid, mid, time, conten) only
    '''
    df = time_feature(df)
    df = post_feature(df)
    return df

if '__main__' == __name__:
    train_data_path = 'data/weibo_split_train_data.txt'
    eval_data_path = 'data/weibo_eval_data.txt'
    raw_train_data = get_train_dataset(train_data_path=train_data_path)
    raw_eval_data = get_train_dataset(train_data_path=eval_data_path)

    train_data = preprocess(raw_train_data)
    eval_data = preprocess(raw_eval_data)

    # 用户feature table
    user_feature = user_portrait(train_data)
    train_data = train_data.merge(user_feature, on='uid', how='left').fillna(-1)
    eval_data = eval_data.merge(user_feature, on='uid', how='left').fillna(-1)

    # debug: 中间结果
    eval_data.head(50).to_csv('tmp.csv', index=False)

    # 删除高维数据
    train_data = train_data.drop(['uid','mid','time','content'],axis=1)
    eval_data = eval_data.drop(['uid','mid','time','content'],axis=1)

    print(train_data.shape, eval_data.shape)

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
