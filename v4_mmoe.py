# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
from pandas import DataFrame
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import MMOE

def time_process(df:DataFrame):
    df['time'] = pd.to_datetime(df['time'])
    # Extract time components
    df['month'] = df['time'].dt.month
    df['hour'] = df['time'].dt.hour
    df['weekday'] = df['time'].dt.weekday
    df['is_weekend'] = df['time'].dt.weekday >= 5

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df

if __name__ == "__main__":
    # from v3_xgboost
    data_train = pd.read_csv('tmp_train.csv')
    data_eval = pd.read_csv('tmp_eval.csv')

    data_train = time_process(data_train)
    data_eval = time_process(data_eval)

    sparse_features = ["uid", "mid"]
    dense_features = ["is_holiday", "content_length", "url_num", "uid_comment_count", "uid_forward_count",
                        'uid_comment_mean', 'uid_comment_max', 'uid_like_count',
                        'uid_like_mean', 'uid_like_max', 'uid_forward_count',
                        'uid_forward_mean', 'uid_forward_max', 'uid_post_count',
                        'uid_total_content_len', 'uid_avg_content_len', 'Month_2_post_num',
                        'Month_3_post_num', 'Month_4_post_num', 'Month_5_post_num',
                        'Month_6_post_num', 'Month_7_post_num', 'user_post_slope',
                         'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin',
                        'month_cos'] + ["has_tag", "has_at", "has_bracket"]
    # bool_feature = ["has_tag", "has_at", "has_bracket"]

    target = ['forward_count', 'comment_count', 'like_count']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data_train[feat] = lbe.fit_transform(data_train[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data_train[dense_features] = mms.fit_transform(data_train[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data_train[feat].max() + 1, embedding_dim=4)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    # split_boundary = int(data.shape[0] * 0.8)
    # train, test = data[:split_boundary], data[split_boundary:]
    train, test = data_train, data_eval
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = MMOE(dnn_feature_columns, task_types=['regression', 'regression', 'regression'],
                 l2_reg_embedding=1e-5, task_names=target, device=device)
    model.compile("adagrad", loss=["mse", "mse", "mse"],
                  metrics=['mse'], )

    history = model.fit(train_model_input, train[target].values, batch_size=32, epochs=10, verbose=2)
    pred_ans = model.predict(test_model_input, 256)
    print("=======")
    pred_ans = np.round(pred_ans).astype(int)
    pred_ans = np.where(pred_ans < 0, 0, pred_ans)
    from utils import score
    print('正确率:',score(test[target].to_numpy(), pred_ans))
