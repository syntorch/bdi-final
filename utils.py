import numpy as np
import pandas as pd

def score(np_lfc_r, np_lfc_p):
    numerator = list()
    denominator = list()
    for i in range(0, np_lfc_r.shape[0]):
        fp = np_lfc_p[i, 0]
        fr = np_lfc_r[i, 0]
        lp = np_lfc_p[i, 1]
        lr = np_lfc_r[i, 1]
        cp = np_lfc_p[i, 2]
        cr = np_lfc_r[i, 2]
        fc_deviation = np.abs(fp - fr) / (fr + 5)
        lc_deviation = np.abs(lp - lr) / (lr + 3)
        cc_deviation = np.abs(cp - cr) / (cr + 3)
        precisioni = 1 - 0.5 * fc_deviation - 0.25 * cc_deviation - 0.25 * lc_deviation
        count_temp = fr + lr + cr
        count_i = 100 if count_temp > 100 else count_temp
        sgn_fuc = 1 if (precisioni - 0.8) > 0 else 0
        numerator.append((count_i + 1) * sgn_fuc)
        denominator.append(count_i + 1)
    precision = np.sum(numerator) / np.sum(denominator)
    return precision

def get_dataset(train_data_path, test_data_path):
    train_col_names = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content']
    data_train = pd.read_csv(train_data_path, sep='\t', header=None, names=train_col_names)
    test_col_names = ['uid', 'mid', 'time', 'content']
    data_test = pd.read_csv(test_data_path, sep='\t', header=None, names=test_col_names)
    return data_train, data_test

def get_train_dataset(train_data_path):
    train_col_names = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content']
    data_train = pd.read_csv(train_data_path, sep='\t', names=train_col_names)
    return data_train

def data_gen():
    data_path = 'data/weibo_train_data.txt'
    eval_data_path = 'data/weibo_eval_data.txt'
    train_data_path = 'data/weibo_split_train_data.txt'
    raw_data = get_train_dataset(train_data_path=data_path)
    raw_eval_data = get_train_dataset(train_data_path=eval_data_path)
    raw_train_data = raw_data.merge(raw_eval_data, how='outer', indicator=True).query('_merge != "both"').drop('_merge', axis=1)
    assert len(raw_train_data) + len(raw_eval_data) == len(raw_data)
    raw_train_data.to_csv(train_data_path, index=False, header=False, sep='\t')