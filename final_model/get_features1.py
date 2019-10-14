import re
import gc
import  numpy as np
import  pandas as pd
import Levenshtein
from  tqdm import  tqdm
from fuzzywuzzy import fuzz
from gensim.summarization import bm25
from gensim.summarization.bm25 import BM25
from tqdm import tqdm_notebook
import os
import time
import math
from multiprocessing import Process,cpu_count,Manager,Pool
import collections
from sklearn.externals import joblib
from gensim import corpora,similarities,models
from util import *
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import pickle
import warnings
import time

from final_model.util import pool_extract, pre_process

warnings.filterwarnings('ignore')

"""
大部分相似度特征
"""


model_path='../final_recall/model/'
dictionary = corpora.Dictionary.load('{}train_dictionary.dict'.format(model_path))
tfidf = models.TfidfModel.load("{}train_tfidf.model".format(model_path))
index = similarities.SparseMatrixSimilarity.load('{}train_index.index'.format(model_path))
item_id_list = joblib.load('{}paper_id.pkl'.format(model_path))

with open('{}train_content.pkl'.format(model_path),'rb') as fr:
    corpus = pickle.load(fr)



####bm5模型
bm25Model = bm25.BM25(corpus)
average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())

del corpus
gc.collect()


tqdm.pandas()
# 后面加载训练好的w2v模型时也需要有这个类的定义, 否则load会报找不到这个类的错误
class EpochSaver(CallbackAny2Vec):
    '''用于保存模型, 打印损失函数等等'''
    def __init__(self, savedir, save_name="word2vector.model"):
        os.makedirs(savedir, exist_ok=True)
        self.save_path = os.path.join(savedir, save_name)
        self.epoch = 0
        self.pre_loss = 0
        self.best_loss = 999999999.9
        self.since = time.time()

    def on_epoch_end(self, model):
        self.epoch += 1
        cum_loss = model.get_latest_training_loss() # 返回的是从第一个epoch累计的
        epoch_loss = cum_loss - self.pre_loss
        time_taken = time.time() - self.since
        print("Epoch %d, loss: %.2f, time: %dmin %ds" %
                    (self.epoch, epoch_loss, time_taken//60, time_taken%60))
        if self.best_loss > epoch_loss:
            self.best_loss = epoch_loss
            print("Better model. Best loss: %.2f" % self.best_loss)
            model.save(self.save_path)
            print("Model %s save done!" % self.save_path)

        self.pre_loss = cum_loss
        self.since = time.time()
# word2vec_path='model/word2vec1.model'
# vec_model = Word2Vec.load(word2vec_path)

print('模型加载完成')


##################################features works##########################################################
#n-gram距离
def get_df_grams(train_sample,values,cols):
    def create_ngram_set(input_list, ngram_value=2):
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    def get_n_gram(df, values=2):
        train_query = df.values
        train_query = [[word for word in str(sen).replace("'", '').split(' ')] for sen in train_query]
        train_query_n = []
        for input_list in train_query:
            train_query_n_gram = set()
            for value in range(2, values + 1):
                train_query_n_gram = train_query_n_gram | create_ngram_set(input_list, value)
            train_query_n.append(train_query_n_gram)
        return train_query_n

    train_query = get_n_gram(train_sample[cols[0]], values)
    train_title = get_n_gram(train_sample[cols[1]], values)
    sim = list(map(lambda x, y: len(x) + len(y) - 2 * len(x & y),
                       train_query, train_title))
    sim_number_rate=list(map(lambda x, y:   len(x & y)/ len(x)  if len(x)!=0 else 0,
                       train_query, train_title))
    return sim ,sim_number_rate

def get_features(data_or,vec_model):
    print('get features:')

    data = data_or.copy()

    data['abstract_pre'] = data['abstract_pre'].apply(
        lambda x: np.nan if str(x) == 'nan' or len(x) < 9 else x)

    data['abstract_pre'] = data['abstract_pre'].apply(
        lambda x: 'none' if str(x) == 'nan' or str(x).split(' ') == ['n', 'o', 'n', 'e'] else x)

    prefix = 'num_'
    
    # 长度
    data[prefix + 'key_text_len'] = data['key_text_pre'].apply(lambda x: len(x.split(' ')))

    # 长度append
    data[prefix + 'description_text_pre_len'] = data['description_text_pre'].apply(lambda x: len(x.split(' ')))

    data.loc[data[prefix + 'key_text_len'] < 7, 'key_text_pre'] = data[data[prefix + 'key_text_len'] < 7][
        'description_text'].apply(
        lambda x: ' '.join(pre_process(re.sub(r'[\[|,]+\*\*\#\#\*\*[\]|,]+', '', x)))).values

    # abstract是否为空
    data[prefix + 'cate_pa_isnull'] = data['abstract_pre'].apply(lambda x: 1 if str(x) == 'none' else 0)

    # key_words是否为空
    data[prefix + 'cate_pkeywords_isnull'] = data['keywords'].apply(lambda x: 1 if str(x) == 'nan' else 0)


    #描述在key_word中出现的次数
    def get_num_key(x,y):
        if str(y)=='nan':
            return -1
        y=y.strip(';').split(';')
        num=0
        for i in y:
            if i in x:
                num+=1
        return num

    data[prefix+'key_in_key_word_number']=list(map(lambda x,y: get_num_key(x,y),data['key_text_pre'],data['keywords']))
    #描述在key_word中出现的次数/key_words的个数
    data[prefix+'key_in_key_word_number_rate']=list(map(lambda x,y: 0 if x==-1 else x/len(y.strip(';').split(';')),data[prefix+'key_in_key_word_number'],
                                                data['keywords']))

    #append
    data[prefix+'key_in_key_word_number2']=list(map(lambda x,y: get_num_key(x,y),data['description_text_pre'],data['keywords']))
    #描述在key_word中出现的次数/key_words的个数
    data[prefix+'key_in_key_word_number2_rate']=list(map(lambda x,y: 0 if x==-1 else x/len(y.strip(';').split(';')),data[prefix+'key_in_key_word_number2'],
                                                data['keywords']))

    # 描述在title出现单词的统计
    def get_num_common_words_and_ratio(merge, col):
        # merge data
        merge = merge[col]
        merge.columns = ['q1', 'q2']
        merge['q2'] = merge['q2'].apply(lambda x: 'none' if str(x) == 'nan' else x)

        q1_word_set = merge.q1.apply(lambda x: x.split(' ')).apply(set).values
        q2_word_set = merge.q2.apply(lambda x: x.split(' ')).apply(set).values

        q1_word_len = merge.q1.apply(lambda x: len(x.split(' '))).values
        q2_word_len = merge.q2.apply(lambda x: len(x.split(' '))).values

        q1_word_len_set = merge.q1.apply(lambda x: len(set(x.split(' ')))).values
        q2_word_len_set = merge.q2.apply(lambda x: len(set(x.split(' ')))).values

        result = [len(q1_word_set[i] & q2_word_set[i]) for i in range(len(q1_word_set))]
        result_ratio_q = [result[i] / q1_word_len[i] for i in range(len(q1_word_set))]
        result_ratio_t = [result[i] / q2_word_len[i] for i in range(len(q1_word_set))]

        result_ratio_q_set = [result[i] / q1_word_len_set[i] for i in range(len(q1_word_set))]
        result_ratio_t_set = [result[i] / q2_word_len_set[i] for i in range(len(q1_word_set))]

        return result, result_ratio_q, result_ratio_t, q1_word_len, q2_word_len, q1_word_len_set, q2_word_len_set, result_ratio_q_set, result_ratio_t_set

    data[prefix + 'common_words_k_pt'], \
    data[prefix + 'common_words_k_pt_k'], \
    data[prefix + 'common_words_k_pt_pt'], \
    data[prefix + 'k_len'], \
    data[prefix + 'pt_len'], \
    data[prefix + 'k_len_set'], \
    data[prefix + 'pt_len_set'], \
    data[prefix + 'common_words_k_pt_k_set'], \
    data[prefix + 'common_words_k_pt_pt_set'] = get_num_common_words_and_ratio(data, ['key_text_pre', 'title_pro'])

    data[prefix + 'common_words_k_at'], \
    data[prefix + 'common_words_k_at_k'], \
    data[prefix + 'common_words_k_at_at'], \
    data[prefix + 'k_len'], \
    data[prefix + 'at_len'], \
    data[prefix + 'k_len_set'], \
    data[prefix + 'at_len_set'], \
    data[prefix + 'common_words_k_at_k_set'], \
    data[prefix + 'common_words_k_at_at_set'] = get_num_common_words_and_ratio(data, ['key_text_pre', 'abstract_pre'])

    #append
    data[prefix + 'common_words_k_pt_2'], \
    data[prefix + 'common_words_k_pt_k_2'], \
    data[prefix + 'common_words_k_pt_pt_2'], \
    data[prefix + 'k_len_2'], \
    data[prefix + 'pt_len'], \
    data[prefix + 'k_len_set_2'], \
    data[prefix + 'pt_len_set'], \
    data[prefix + 'common_words_k_pt_k_set_2'], \
    data[prefix + 'common_words_k_pt_pt_set_2'] = get_num_common_words_and_ratio(data, ['description_text_pre', 'title_pro'])

    data[prefix + 'common_words_k_at_2'], \
    data[prefix + 'common_words_k_at_k_2'], \
    data[prefix + 'common_words_k_at_at_2'], \
    data[prefix + 'k_len_2'], \
    data[prefix + 'at_len'], \
    data[prefix + 'k_len_set_2'], \
    data[prefix + 'at_len_set'], \
    data[prefix + 'common_words_k_at_k_set_2'], \
    data[prefix + 'common_words_k_at_at_set_2'] = get_num_common_words_and_ratio(data, ['description_text_pre', 'abstract_pre'])



    # Jaccard 相似度
    def jaccard(x, y):
        if str(y) == 'nan':
            y = 'none'
        x = set(x)
        y = set(y)
        return float(len(x & y) / len(x | y))

    data[prefix + 'jaccard_sim_k_pt'] = list(map(lambda x, y: jaccard(x, y), data['key_text_pre'], data['title_pro']))
    data[prefix + 'jaccard_sim_k_pa'] = list(
        map(lambda x, y: jaccard(x, y), data['key_text_pre'], data['abstract_pre']))

    #append
    data[prefix + 'jaccard_sim_k_pt2'] = list(map(lambda x, y: jaccard(x, y), data['description_text_pre'], data['title_pro']))
    data[prefix + 'jaccard_sim_k_pa2'] = list(
        map(lambda x, y: jaccard(x, y), data['key_text_pre'], data['description_text_pre']))

    # 编辑距离
    print('get edict distance:')
    data[prefix + 'edict_distance_k_pt'] = list(
        map(lambda x, y: Levenshtein.distance(x, y) / len(x), tqdm(data['key_text_pre']), data['title_pro']))
    data[prefix + 'edict_jaro'] = list(
        map(lambda x, y: Levenshtein.jaro(x, y), tqdm(data['key_text_pre']), data['title_pro']))
    data[prefix + 'edict_ratio'] = list(
        map(lambda x, y: Levenshtein.ratio(x, y), tqdm(data['key_text_pre']), data['title_pro']))
    data[prefix + 'edict_jaro_winkler'] = list(
        map(lambda x, y: Levenshtein.jaro_winkler(x, y), tqdm(data['key_text_pre']), data['title_pro']))

    data[prefix + 'edict_distance_k_pa'] = list(
        map(lambda x, y: Levenshtein.distance(x, y) / len(x), tqdm(data['key_text_pre']),
            data['abstract_pre']))
    data[prefix + 'edict_jaro_pa'] = list(
        map(lambda x, y: Levenshtein.jaro(x, y), tqdm(data['key_text_pre']), data['abstract_pre']))
    data[prefix + 'edict_ratio_pa'] = list(
        map(lambda x, y: Levenshtein.ratio(x, y), tqdm(data['key_text_pre']), data['abstract_pre']))
    data[prefix + 'edict_jaro_winkler_pa'] = list(
        map(lambda x, y: Levenshtein.jaro_winkler(x, y), tqdm(data['key_text_pre']), data['abstract_pre']))

    #append
    print('get edict distance:')
    data[prefix + 'edict_distance_k_pt_2'] = list(
        map(lambda x, y: Levenshtein.distance(x, y) / len(x), tqdm(data['description_text_pre']), data['title_pro']))
    data[prefix + 'edict_jaro_2'] = list(
        map(lambda x, y: Levenshtein.jaro(x, y), tqdm(data['description_text_pre']), data['title_pro']))
    data[prefix + 'edict_ratio_2'] = list(
        map(lambda x, y: Levenshtein.ratio(x, y), tqdm(data['description_text_pre']), data['title_pro']))
    data[prefix + 'edict_jaro_winkler_2'] = list(
        map(lambda x, y: Levenshtein.jaro_winkler(x, y), tqdm(data['description_text_pre']), data['title_pro']))

    data[prefix + 'edict_distance_k_pa_2'] = list(
        map(lambda x, y: Levenshtein.distance(x, y) / len(x), tqdm(data['description_text_pre']),
            data['abstract_pre']))
    data[prefix + 'edict_jaro_pa_2'] = list(
        map(lambda x, y: Levenshtein.jaro(x, y), tqdm(data['description_text_pre']), data['abstract_pre']))
    data[prefix + 'edict_ratio_pa_2'] = list(
        map(lambda x, y: Levenshtein.ratio(x, y), tqdm(data['description_text_pre']), data['abstract_pre']))
    data[prefix + 'edict_jaro_winkler_pa_2'] = list(
        map(lambda x, y: Levenshtein.jaro_winkler(x, y), tqdm(data['description_text_pre']), data['abstract_pre']))

    #余弦相似度
    def get_sim(doc, corpus):
        corpus = corpus.split(' ')
        corpus_vec = [dictionary.doc2bow(corpus)]
        corpus_tfidf = tfidf[corpus_vec]
        featurenum = len(dictionary.token2id.keys())
        index_i = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=featurenum)
        doc = doc.split(' ')
        vec = dictionary.doc2bow(doc)
        vec_tfidf = tfidf[vec]
        sim = index_i.get_similarities(vec_tfidf)
        return sim[0]

    data[prefix + 'sim'] = list(map(lambda x, y: get_sim(x, y), tqdm(data['key_text_pre']), data['title_pro']))
    data[prefix + 'sim_pa'] = list(map(lambda x, y: get_sim(x, y), tqdm(data['key_text_pre']), data['abstract_pre']))

    #append
    data[prefix + 'sim_2'] = list(map(lambda x, y: get_sim(x, y), tqdm(data['description_text_pre']), data['title_pro']))
    data[prefix + 'sim_pa_2'] = list(map(lambda x, y: get_sim(x, y), tqdm(data['description_text_pre']), data['abstract_pre']))

    # tfidf
    def get_simlilary(query, title):
        def get_weight_counter_and_tf_idf(x, y):
            x = x.split()
            y = y.split()
            corups = x + y
            obj = dict(collections.Counter(corups))
            x_weight = []
            y_weight = []
            idfs = []
            for key in obj.keys():
                idf = 1
                w = obj[key]
                if key in x:
                    idf += 1
                    x_weight.append(w)
                else:
                    x_weight.append(0)
                if key in y:
                    idf += 1
                    y_weight.append(w)
                else:
                    y_weight.append(0)
                idfs.append(math.log(3.0 / idf) + 1)
            return [np.array(x_weight), np.array(y_weight), np.array(x_weight) * np.array(idfs),
                    np.array(y_weight) * np.array(idfs), np.array(list(obj.keys()))]

        weight = list(map(lambda x, y: get_weight_counter_and_tf_idf(x, y),
                          tqdm(query), title))
        x_weight_couner = []
        y_weight_couner = []
        x_weight_tfidf = []
        y_weight_tfidf = []
        words = []
        for i in weight:
            x_weight_couner.append(i[0])
            y_weight_couner.append(i[1])
            x_weight_tfidf.append(i[2])
            y_weight_tfidf.append(i[3])
            words.append(i[4])

        # 曼哈顿距离
        def mhd_simlilary(x, y):
            return np.linalg.norm(x - y, ord=1)

        mhd_simlilary_counter = list(map(lambda x, y: mhd_simlilary(x, y),
                                         x_weight_couner, y_weight_couner))
        mhd_simlilary_tfidf = list(map(lambda x, y: mhd_simlilary(x, y),
                                       x_weight_tfidf, y_weight_tfidf))

        # 余弦相似度
        def cos_simlilary(x, y):
            return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

        cos_simlilary_counter = list(map(lambda x, y: cos_simlilary(x, y),
                                         x_weight_couner, y_weight_couner))
        cos_simlilary_tfidf = list(map(lambda x, y: cos_simlilary(x, y),
                                       x_weight_tfidf, y_weight_tfidf))

        # 欧式距离
        def Euclidean_simlilary(x, y):
            return np.sqrt(np.sum(x - y) ** 2)

        Euclidean_simlilary_counter = list(map(lambda x, y: Euclidean_simlilary(x, y),
                                               x_weight_couner, y_weight_couner))
        Euclidean__simlilary_tfidf = list(map(lambda x, y: Euclidean_simlilary(x, y),
                                              x_weight_tfidf, y_weight_tfidf))

        return mhd_simlilary_counter, mhd_simlilary_tfidf, cos_simlilary_counter, \
               cos_simlilary_tfidf, Euclidean_simlilary_counter, Euclidean__simlilary_tfidf

    data[prefix + 'mhd_similiary'], data[prefix + 'tf_mhd_similiary'], \
    data[prefix + 'cos_similiary'], data[prefix + 'tf_cos_similiary'], \
    data[prefix + 'os_similiary'], data[prefix + 'tf_os_similiary'] = get_simlilary(data['key_text_pre'],data['title_pro'])


    data[prefix + 'mhd_similiary_pa'], data[prefix + 'tf_mhd_similiary_pa'], \
    data[prefix + 'cos_similiary_pa'], data[prefix + 'tf_cos_similiary_pa'], \
    data[prefix + 'os_similiary_pa'], data[prefix + 'tf_os_similiary_pa'] = get_simlilary(data['key_text_pre'],data['abstract_pre'])

    '词向量平均的相似度'

    def get_vec(x):
        vec = []
        for word in x.split():
            if word in vec_model:
                vec.append(vec_model[word])
        if len(vec) == 0:
            return np.nan
        else:
            return np.mean(np.array(vec), axis=0)

    data['key_text_pre_vec'] = data['key_text_pre'].progress_apply(lambda x: get_vec(x))
    data['title_pro_vec'] = data['title_pro'].progress_apply(lambda x: get_vec(x))
    data['abstract_pre_vec'] = data['abstract_pre'].progress_apply(lambda x: get_vec(x))
    data['description_text_pre_vec'] = data['description_text_pre'].progress_apply(lambda x: get_vec(x))

    # cos
    data[prefix + 'cos_mean_word2vec'] = list(map(lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)),
                                                  tqdm(data['key_text_pre_vec']), data['title_pro_vec']))
    data[prefix + 'cos_mean_word2vec'] = data[prefix + 'cos_mean_word2vec'].progress_apply(
        lambda x: np.nan if np.isnan(x).any() else x)

    # 欧式距离
    data[prefix + 'os_mean_word2vec'] = list(map(lambda x, y: np.sqrt(np.sum((x - y) ** 2)),
                                                 tqdm(data['key_text_pre_vec']), data['title_pro_vec']))

    # mhd
    data[prefix + 'mhd_mean_word2vec'] = list(map(lambda x, y: np.nan if np.isnan(x).any() or np.isnan(y).any() else
    np.linalg.norm(x - y, ord=1), tqdm(data['key_text_pre_vec']), data['title_pro_vec']))


    # cos
    data[prefix + 'cos_mean_word2vec_pa'] = list(map(lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)),
                                                  tqdm(data['key_text_pre_vec']), data['abstract_pre_vec']))
    data[prefix + 'cos_mean_word2vec_pa'] = data[prefix + 'cos_mean_word2vec_pa'].progress_apply(
        lambda x: np.nan if np.isnan(x).any() else x)

    # 欧式距离
    data[prefix + 'os_mean_word2vec_pa'] = list(map(lambda x, y: np.sqrt(np.sum((x - y) ** 2)),
                                                 tqdm(data['key_text_pre_vec']), data['abstract_pre_vec']))

    # mhd
    data[prefix + 'mhd_mean_word2vec_pa'] = list(map(lambda x, y: np.nan if np.isnan(x).any() or np.isnan(y).any() else
    np.linalg.norm(x - y, ord=1), tqdm(data['key_text_pre_vec']), data['abstract_pre_vec']))


    #append
    data[prefix + 'cos_mean_word2vec_2'] = list(map(lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)),
                                                  tqdm(data['description_text_pre_vec']), data['title_pro_vec']))
    data[prefix + 'cos_mean_word2vec_2'] = data[prefix + 'cos_mean_word2vec_2'].progress_apply(
        lambda x: np.nan if np.isnan(x).any() else x)

    # 欧式距离
    data[prefix + 'os_mean_word2vec_2'] = list(map(lambda x, y: np.sqrt(np.sum((x - y) ** 2)),
                                                 tqdm(data['description_text_pre_vec']), data['title_pro_vec']))

    # mhd
    data[prefix + 'mhd_mean_word2vec_2'] = list(map(lambda x, y: np.nan if np.isnan(x).any() or np.isnan(y).any() else
    np.linalg.norm(x - y, ord=1), tqdm(data['description_text_pre_vec']), data['title_pro_vec']))

    # cos
    data[prefix + 'cos_mean_word2vec_pa2'] = list(map(lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)),
                                                  tqdm(data['description_text_pre_vec']), data['abstract_pre_vec']))
    data[prefix + 'cos_mean_word2vec_pa2'] = data[prefix + 'cos_mean_word2vec_pa2'].progress_apply(
        lambda x: np.nan if np.isnan(x).any() else x)

    # 欧式距离
    data[prefix + 'os_mean_word2vec_pa2'] = list(map(lambda x, y: np.sqrt(np.sum((x - y) ** 2)),
                                                 tqdm(data['description_text_pre_vec']), data['abstract_pre_vec']))

    # mhd
    data[prefix + 'mhd_mean_word2vec_pa2'] = list(map(lambda x, y: np.nan if np.isnan(x).any() or np.isnan(y).any() else
    np.linalg.norm(x - y, ord=1), tqdm(data['description_text_pre_vec']), data['abstract_pre_vec']))




    #n-gram距离相关
    data[prefix+'n_gram_sim'],data[prefix+'sim_numeber_rate']=get_df_grams(data,2,['key_text_pre','title_pro'])
    data[prefix+'n_gram_sim_pa'],data[prefix+'sim_numeber_rate_pa']=get_df_grams(data,2,['key_text_pre','abstract_pre'])

    #append
    #n-gram距离相关
    data[prefix+'n_gram_sim_2'],data[prefix+'sim_numeber_rate_2']=get_df_grams(data,2,['description_text_pre','title_pro'])
    data[prefix+'n_gram_sim_pa_2'],data[prefix+'sim_numeber_rate_pa_2']=get_df_grams(data,2,['description_text_pre','abstract_pre'])

    def apply_fun(df):
        df.columns = ['d_id', 'key', 'doc']
        query_id_group = df.groupby(['d_id'])
        bm_list = []
        for name, group in tqdm(query_id_group):
            corpus = group['doc'].values.tolist()
            corpus = [sentence.strip().split() for sentence in corpus]
            query = group['key'].values[0].strip().split()
            bm25Model = BM25(corpus)
            average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
            bmscore = bm25Model.get_scores(query, average_idf)
            bm_list.extend(bmscore)

        return bm_list

    data[prefix + 'bm25'] = apply_fun(data[['description_id', 'key_text_pre', 'title_pro']])
    data[prefix + 'bm25_pa'] = apply_fun(data[['description_id', 'key_text_pre', 'abstract_pre']])

    #append
    data[prefix + 'bm25_2'] = apply_fun(data[['description_id', 'description_text_pre', 'title_pro']])
    data[prefix + 'bm25_pa_2'] = apply_fun(data[['description_id', 'description_text_pre', 'abstract_pre']])


    # get bm25
    def get_bm25(p_id, query):
        query = query.split(' ')
        score = bm25Model.get_score(query, item_id_list.index(p_id), average_idf)
        return score

    data[prefix + 'bm_25_all'] = list(map(lambda x, y: get_bm25(x, y), tqdm(data['paper_id']), data['key_text_pre']))
    #append
    data[prefix + 'bm_25_all_2'] = list(map(lambda x, y: get_bm25(x, y), tqdm(data['paper_id']), data['description_text_pre']))

    feat = []
    for col in data.columns:
        if re.match('num_', col) != None:
            feat.append(col)

    data = data[feat]

    return data




if __name__=='__main__':
    path = 'train_set/'

    test_data = pd.read_csv(path + 'test_data_merge_bm25_tfidf_20.csv')

    word2vec_path = 'model/word2vec3.model'
    vec_model = Word2Vec.load(word2vec_path)
    t1 = time.time()

    test_feat=pool_extract(test_data,get_features,vec_model,3000)
    test_feat.to_csv('feat/test_data_merge_bm25_tfidf_20_featall.csv', index=False)
    del test_data,test_feat
    gc.collect()

    train_data = pd.read_csv(path + 'train_data_merge_bm25_tfidf_20.csv')
    train_feat=pool_extract(train_data,get_features,vec_model,3000)
    train_feat.to_csv('feat/train_data_merge_bm25_tfidf_20_featall.csv',index=False)

    print('success')
    t2 = time.time()
    print((t2 - t1) / 60)


