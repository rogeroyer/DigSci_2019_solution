import pandas as pd
from  tqdm import  tqdm
from gensim import corpora,similarities,models
import pandas as pd
import pickle
from util import pre_process
import os


#################读取候选论文
papers = pd.read_csv('../data/candidate_paper.csv')
papers=papers[papers['paper_id'].notnull()]
print(papers.shape)
#只要no-content的论文
papers=papers[papers['journal']=='no-content']
print(papers.shape)

#填充空值
papers['abstract'] = papers['abstract'].fillna('none')
papers['title'] = papers['title'].fillna('none')
papers['keywords'] = papers['keywords'].fillna('none')


#内容和id
train=papers['title'].values+' '+papers['abstract'].values+' '+papers['keywords'].apply(lambda x: x.replace(';',' ')).values
train_item_id=list(papers['paper_id'].values)

#保存论文的id
with open('paper_id.pkl', 'wb') as fw:
    pickle.dump(train_item_id,fw)


#保存预处理后的论文内容
if not os.path.exists('train_content.pkl'):
    with open('train_content.pkl','wb') as fw:
        train = list(map(lambda x: pre_process(x), tqdm(train)))
        pickle.dump(train,fw)
else:
    with open('train_content.pkl','rb') as fr:
        train = pickle.load(fr)


dictionary = corpora.Dictionary(train)
corpus = [dictionary.doc2bow(text) for text in train]

# corpus是一个返回bow向量的迭代器。下面代码将完成对corpus中出现的每一个特征的IDF值的统计工作
tfidf_model = models.TfidfModel(corpus, dictionary=dictionary)
corpus_tfidf = tfidf_model[corpus]

dictionary.save('train_dictionary.dict')  # 保存生成的词典
tfidf_model.save('train_tfidf.model')
corpora.MmCorpus.serialize('train_corpuse.mm', corpus)
featurenum = len(dictionary.token2id.keys())  # 通过token2id得到特征数
# 稀疏矩阵相似度，从而建立索引,我们用待检索的文档向量初始化一个相似度计算的对象
index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=featurenum)    #这是文档的index
index.save('train_index.index')
