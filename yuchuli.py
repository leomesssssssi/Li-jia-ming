# This is a sample Python script.

import functools
import math
import codecs
import string
from zhon.hanzi import punctuation
from gensim.models import word2vec
import sys
from sklearn.decomposition import PCA
import random
import logging
import gensim.models
import numpy as np
import re
import gensim.models as g
from sklearn.preprocessing import StandardScaler
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import jieba
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def load_data_test(input_file_name_in):
    input_file = open(input_file_name_in, 'r', encoding='utf-8')
    id_list_out = []
    doc_list = []
    label_list_out = []
    str_list = []
    lines_1 = input_file.read().split('\n')
    count = 0
    last = ''
    print('loading:')
    for i in lines_1:
        count += 1
        print('\r', count/len(lines_1)*100, '%', end='')
        if '<review id=' in i:
            isp = i.split(' ')
            for sp in isp:
                if 'id' in sp:
                    id_ind = sp.index("=")
                    new_id = sp[id_ind + 2:-1]
                    id_in = int(new_id)
                    id_list_out.append(id_in)
                elif 'label' in sp:
                    lb_ind = sp.index("=")
                    new_lb = sp[lb_ind + 2:-2]
                    label_in = int(new_lb)
                    label_list_out.append(label_in)
        else:
            if '</review>' in i:
                new_str = ''
                for j in last.split("\n"):
                    new_str += j
                str_list.append(new_str)
                seg_list = jieba.cut(new_str)
                filter_list = word_filter(seg_list)
                if not filter_list:
                    label_list_out.pop()
                    id_list_out.pop()
                else:
                    doc_list.append(filter_list)
                last = ''
            else:
                last += i
    id_list_output = np.array(id_list_out[0:len(id_list_out)])
    label_list_output = np.array(label_list_out[0:len(label_list_out)])
    print('\n')
    print('loading successfully!\n')
    return id_list_output, doc_list, label_list_output


# 加载停用表
def get_stopwords_list():
    stop_word_path = './cn_stopwords.txt'
    stop_words_list = []
    stopwords_file = open(stop_word_path, 'r', encoding='utf-8')
    for sw in stopwords_file.readlines():
        stop_words_list.append(sw.replace('\n', ''))
    return stop_words_list


# 去除干扰词
def word_filter(seg_list_in):
    stopword_list = get_stopwords_list()
    filter_list_out = []
    cn_reg = '^[\u4e00-\u9fa5]+$'
    for seg in seg_list_in:
        word = seg
        if word not in stopword_list and re.search(cn_reg, word):
            filter_list_out.append(word)
    return filter_list_out


# 导入,返回筛词后的分词结果
def load_data_sample(input_file_name, label):
    input_file = open(input_file_name, 'r', encoding='utf-8')
    id_list = []
    str_list = []
    label_list_out = []
    doc_list = []
    lines_1 = input_file.read().split('\n')
    count = 0
    last = ''
    for i in lines_1:
        if '<review id=' in i:
            ind = i.index("=")
            new_id = i[ind + 2:-2]
            id_in = int(new_id)
            id_list.append(id_in)
            label_list_out.append(label)
        else:
            if '</review>' in i:
                new_str = ''
                for j in last.split("\n"):
                    new_str += j
                new_str = re.sub("[a-zA-Z0-9]", "", new_str)
                str_list.append(new_str)
                seg_list = jieba.cut(new_str)
                filter_list = word_filter(seg_list)
                if filter_list == []:
                    label_list_out.pop()
                    id_list.pop()
                else:
                    doc_list.append(filter_list)
                last = ''
            else:
                last += i
    print(doc_list[3])
    return id_list, doc_list, label_list_out


# idf值统计方法
def train_idf(doc_list):
    idf_dic = {}
    tt_count = len(doc_list)
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0
    for k, v in idf_dic.items():
        idf_dic[k] = math.log(tt_count/(1.0+v))
    default_idf = math.log(tt_count/1.0)
    return idf_dic, default_idf


# 排序函数
def cmp(e1, e2):
    import numpy as np
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


# TF-IDF类
class Tfidf(object):
    def __init__(self, idf_dic, default_idf, word_list):
        self.word_list = word_list
        self.idf_dic, self.dafault_idf = idf_dic, default_idf
        self.tf_dic = self.get_tf_dic()

    # 统计tf值
    def get_tf_dic(self):
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0.0) + 1.0
        tt_count = len(self.word_list)
        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / tt_count
        return tf_dic

    # 按公式计算 tf-idf
    def get_tfidf(self):
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.dafault_idf)
            tf = self.tf_dic.get(word, 0)
            tfidf = tf * idf
            tfidf_dic[word] = tfidf
        return tfidf_dic


def tfidf_extract(doc_list_in, word_list):
    idf_dic, default_idf = train_idf(doc_list_in)
    tfidf_model = Tfidf(idf_dic, default_idf, word_list)
    return tfidf_model.get_tfidf()


def doc2vec(file_name, model):
    import jieba
    doc = [w for x in codecs.open(file_name, 'r', 'utf-8').readlines() for w in jieba.cut(x.strip())]
    doc_vec_all = model.infer_vector(doc)
    return doc_vec_all


def simlarityCalu (vector1, vector2):
    vector1Mod = np.sqrt(vector1.dot(vector1))
    vector2Mod = np.sqrt(vector2.dot(vector2))
    if vector2Mod != 0 and vector1Mod != 0:
        simlarity = (vector1.dot(vector2)) / (vector1Mod * vector2Mod)
    else:
        simlarity = 0
    return simlarity

def normalization(data):
    _range = np.max(data, axis=0) - np.min(data, axis=0)
    return (data - np.min(data, axis=0)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def file_write(output_file_name, id_list_in, label_list_in, doc_list_in):
    with open(output_file_name, 'w', encoding='utf-8') as output_file:
        for i in range(0, len(id_list_in)):
            output_file.write('<review id="' + ''.join(str(id_list_in[i])) + '"  label="')
            output_file.write(''.join(str(label_list_in[i])) + '">' + '\n')
            output_file.write(''.join(str(doc_list_in[i])) + '\n')
            output_file.write('</review>' + '\n')
    return 0

def load_doc_list(input_file_name):
    input_file = open(input_file_name, 'r', encoding='utf-8')
    lines_1 = input_file.read().split('\n')
    doc_list = []
    for new_str in lines_1:
        seg_list = jieba.cut(new_str)
        filter_list = word_filter(seg_list)
        if not filter_list:
            continue
        else:
            doc_list.append(filter_list)
    return doc_list
# Press the green button in the gutter to run the script.
# <review id="0"  label="1">


if __name__ == '__main__':
    id_list_test, doc_list_test, label_list_test = load_data_test('./test.label.cn.txt')
    model_path = './doc2vec_test2.model'
    model = g.Doc2Vec.load(model_path)
    doc_vec = []
    doc_list = doc_list_test
    print(len(doc_list))
    for i in range(0, len(doc_list)):
        print('\r', i / len(doc_list) * 100, '%', end='')
        doc_vec_all = model.infer_vector(doc_list[i])
        doc_vec.append(doc_vec_all)
    X = np.array(doc_vec[0:len(doc_list)])
    X_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)
    file_write('test_d2v_300.txt', id_list_test, label_list_test, X)

    pca_98 = PCA(n_components=272).fit(X)
    pca_x98 = pca_98.transform(X)
    print(pca_x98.shape)
    y = np.array(label_list_test[0:len(label_list_test)])
    x98 = standardization(pca_x98)
    print(x98.shape)
    file_write('test_0.98_d2v.txt', id_list_test, label_list_test, x98)






# See PyCharm help at https://www.jetbrains.com/help/pycharm/

""""#for letter in 'Runoob':
    if letter=='o':
        continue
    print('当前字母'+letter)""

id_list_tlc, doc_list_tlc, label_list_tlc = load_data_test('./test.label.cn.txt')

model_path = "wordvec.model"
wordvec = gensim.models.Word2Vec.load(model_path)
print(wordvec.wv.get_vector("华为"))
print(wordvec.wv.most_similar("华为",topn=5))
print(wordvec.wv.similarity("西红柿","番茄"))"""

"""sn_str_list, sn_doc_list = load_data('./sample.negative.txt')
    sp_str_list, sp_doc_list = load_data('./sample.positive.txt')
    doc_list = sp_doc_list
    doc_list.extend(sn_doc_list)
    ran = random.randint(1, len(doc_list))
    tfidf_extract(doc_list, doc_list[ran])
    print("\n\n\n\n\n\n\n\n")
    print(doc_list[ran])
    
    
    input_file_name = './corpus_cn.txt'
    input_file = open(input_file_name, 'r', encoding='utf-8')
    lines_1 = input_file.read().split('\n')
    print(len(lines_1))
    print(lines_1[1])
    print(lines_1[3])
    
    model_path = './doc2vec_test2.model'
    model = g.Doc2Vec.load(model_path)
    p1 = './P1.txt'
    p2 = './P2.txt'
    P1_doc2vec = doc2vec(p1, model)
    P2_doc2vec = doc2vec(p2, model)
    print(simlarityCalu(P1_doc2vec, P2_doc2vec))
    
    doc_vec = []
    model_path = './doc2vec_test2.model'
    model = g.Doc2Vec.load(model_path)
    for i in range(0, len(doc_list_tlc)):
        doc_vec_all = model.infer_vector(doc_list_tlc[i])
        doc_vec.append(doc_vec_all)
    X = np.array(doc_vec[0:len(doc_list_tlc)])
    X_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)
    pca = PCA(n_components=119).fit(X)
    pca_x = pca.transform(X)
    y = np.array(label_list_tlc[0:len(label_list_tlc)])
    x = standardization(pca_x)
    print(x.shape)
    print(y.shape)
    print(x[123])
    file_write('output_0.8_test.txt', id_list_tlc, label_list_tlc, pca_x)"""
