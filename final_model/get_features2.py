from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import gc
import collections
from tqdm import tqdm
import  time
import pickle
import  re
from  util import *


"""
tfidf 的部分特征
"""


##############################################sklearn tfidf################################################





def split(x):
    return x.split(' ')

model_path='../final_recall/model/'
tfidf_path='{}sklearn_tfidf_model.pkl'.format(model_path)
with open(tfidf_path,'rb') as handle:
    tft=pickle.load(handle)
print(tft)



i = 1
path = 'train_set/'
train_all=pd.read_csv(path + 'test_data_merge_bm25_tfidf_20.csv')
print(i)

print(train_all.isnull().sum())
train_all['abstract_pre'] = train_all['abstract_pre'].apply(
        lambda x: np.nan if str(x) == 'nan' or len(x) < 9 else x)

train_all['abstract_pre'] = train_all['abstract_pre'].apply(
        lambda x: 'none' if str(x) == 'nan' or str(x).split(' ') == ['n', 'o', 'n', 'e'] else x)

train_all['paper_content_pre'] = train_all['title_pro'].values + ' ' + train_all['abstract_pre'].values + ' ' + train_all[
        'keywords'].apply(lambda x: ' '.join(x.split(';') if str(x) != 'nan' else 'none')).values

# 长度
train_all['key_text_len'] = train_all['key_text_pre'].apply(lambda x: len(x.split(' ')))

# 长度append
train_all[ 'description_text_pre_len'] = train_all['description_text_pre'].apply(lambda x: len(x.split(' ')))

train_all.loc[train_all[ 'key_text_len'] < 7, 'key_text_pre'] = train_all[train_all[ 'key_text_len'] < 7][
    'description_text'].apply(
    lambda x: ' '.join(pre_process(re.sub(r'[\[|,]+\*\*\#\#\*\*[\]|,]+', '', x)))).values





def get_tf_sim(train_query_tf,train_title_tf):
    # 余弦
    v_num = np.array(train_query_tf.multiply(train_title_tf).sum(axis=1))[:, 0]
    v_den = np.array(np.sqrt(train_query_tf.multiply(train_query_tf).sum(axis=1)))[:, 0] * np.array(
            np.sqrt(train_title_tf.multiply(train_title_tf).sum(axis=1)))[:, 0]
    v_num[np.where(v_den == 0)] = 1
    v_den[np.where(v_den == 0)] = 1
    v_score1 = 1 - v_num / v_den

    # 欧式
    v_score = train_query_tf - train_title_tf
    v_score2 = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

    # 曼哈顿
    v_score = np.abs(train_query_tf - train_title_tf)
    v_score3 = v_score.sum(axis=1)

    return  v_score1,v_score2,v_score3




features = train_all[['description_id']]
train_query_tf = tft.transform(train_all['key_text_pre'].values)
train_query_tf2 = tft.transform(train_all['description_text_pre'].values)
train_title_tf = tft.transform(train_all['title_pro'].values)
train_title_tf2 = tft.transform(train_all['abstract_pre'].values)

features['tfidf_cos'],features['tfidf_os'] ,features['tfidf_mhd']=get_tf_sim(train_query_tf,train_title_tf)
features['tfidf_cos2'],features['tfidf_os2'] ,features['tfidf_mhd2']=get_tf_sim(train_query_tf,train_title_tf2)

features['tfidf_cos_2'],features['tfidf_os_2'] ,features['tfidf_mhd_2']=get_tf_sim(train_query_tf2,train_title_tf)
features['tfidf_cos2_2'],features['tfidf_os2_2'] ,features['tfidf_mhd2_2']=get_tf_sim(train_query_tf2,train_title_tf2)


del  train_title_tf,train_query_tf,train_query_tf2,train_title_tf2
gc.collect()


#tfidf match share
print('get tfidf match share:')

def get_weight(count, eps=100, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


def load_weight(data):
    words = [x for y in data for x in y.split()]
    counts = collections.Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    del counts
    del words
    del data
    gc.collect()
    return weights


def tfidf_match_share(queries, titles, weights):
    ret = []
    for i in tqdm(range(len(queries))):
        q, t = queries[i].split(), titles[i].split()
        q1words = {}
        q2words = {}
        for word in q:
            q1words[word] = 1
        for word in t:
            q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            R = 0
        else:
            shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + \
                             [weights.get(w, 0) for w in q2words.keys() if w in q1words]
            total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

            R = np.sum(shared_weights) / np.sum(total_weights)
        ret.append(R)
    return ret

model_path='../final_recall/model/'
with open('{}train_content.pkl'.format(model_path),'rb') as fr:
    content= pickle.load(fr)

words = [x for y in content for x in y]
counts = collections.Counter(words)
del content
gc.collect()

weights = {word: get_weight(count) for word, count in counts.items()}
del counts
gc.collect()

features['tfidf_match_share'] = tfidf_match_share(train_all['key_text_pre'].values, train_all['title_pro'].values, weights)
features['tfidf_match_share_pa'] = tfidf_match_share(train_all['key_text_pre'].values, train_all['abstract_pre'].values, weights)

features['tfidf_match_share_2'] = tfidf_match_share(train_all['description_text_pre'].values, train_all['title_pro'].values, weights)
features['tfidf_match_share_pa_2'] = tfidf_match_share(train_all['description_text_pre'].values, train_all['abstract_pre'].values, weights)


features = features.drop(['description_id'], axis=1)
print(features.head())
features.columns=['num_'+col for col in list(features.columns)]
features.to_csv('feat/test_data_merge_bm25_tfidf_20_featall_tfidf2.csv')

