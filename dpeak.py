import numpy as np
import json
import jieba
import matplotlib.pyplot as plt

import time
import math

from pprint import pprint
from sklearn.datasets import make_blobs
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer


def distanceNorm(Norm, D_value, feature1=None, feature2=None):
    # initialization

    # Norm for distance
    if Norm == '1':
        counter = np.absolute(D_value)
        counter = np.sum(counter)
    elif Norm == '2':
        counter = np.power(D_value, 2)
        counter = np.sum(counter)
        counter = np.sqrt(counter)
    elif Norm == 'Infinity':
        counter = np.absolute(D_value)
        counter = np.max(counter)
    elif Norm == 'cos':
        counter = np.power(feature1, 2)
        counter = np.sum(counter)
        counter1 = np.sqrt(counter)
        counter = np.power(feature2, 2)
        counter = np.sum(counter)
        counter2 = np.sqrt(counter)
        counter = feature1 * feature2
        counter = np.sum(counter)
        counter /= (counter1*counter2)
        counter = 1 - counter
    else:
        raise Exception('We will program this later......')
    return counter


def chi(x):
    if x < 0:
        return 1
    else:
        return 0


def distanceNormSparse(d1, d2):
    result  = 0
    for k in d1:
        if k not in d2:
            result += pow(d1[k], 2)
        else:
            result += pow(d1[k]-d2[k], 2)
    for k in d2:
        if k not in d1:
            result += pow(d2[k], 2)
    return math.sqrt(result)


def distanceNormSparse2(d1, d2, pre_cutoff):
    result  = 0
    for k in d1:
        if k not in d2:
            result += pow(d1[k], 2)
        else:
            result += pow(d1[k]-d2[k], 2)
        if result > pre_cutoff:
            return result
    for k in d2:
        if k not in d1:
            result += pow(d2[k], 2)adf
            if result > pre_cutoff:
                return result
    return math.sqrt(result)


def fit(features, t=0.00001, distanceMethod='2', pre_cut_off=0.9999999999999997*1.05):
    labels = features
    # initialization
    distance = np.zeros((len(labels), len(labels)))
    distance_sort = list()
    density = np.zeros(len(labels))
    distance_higherDensity = np.zeros(len(labels))
    # compute distance
    print('start calculate distance')
    s = (time.time())

    # for index_i in range(len(labels)):
    #     for index_j in range(index_i + 1, len(labels)):
    #         if distanceMethod == 'sparse':
    #             distance[index_i, index_j] = distanceNormSparse(features[index_i], features[index_j])
    #         else:
    #             D_value = features[index_i] - features[index_j]
    #             distance[index_i, index_j] = distanceNorm(distanceMethod, D_value, features[index_i], features[index_j])
    #         distance_sort.append(distance[index_i, index_j])

    for index_i in range(len(labels)):
        for index_j in range(index_i + 1, len(labels)):
            if distanceMethod == 'sparse':
                distance[index_i, index_j] = distanceNormSparse2(features[index_i], features[index_j], pre_cut_off)
            else:
                D_value = features[index_i] - features[index_j]
                distance[index_i, index_j] = distanceNorm(distanceMethod, D_value, features[index_i], features[index_j])
            distance_sort.append(distance[index_i, index_j])




    print('end calculate distance')
    print(time.time()-s)
    distance += distance.T
    # compute optimal cutoff
    distance_sort.sort()
    distance_sort = np.array(distance_sort)
    cutoff = distance_sort[int(len(distance_sort) * t)]
    # computer density
    for index_i in range(len(labels)):
        distance_cutoff_i = distance[index_i] - cutoff
        for index_j in range(1, len(labels)):
            density[index_i] += chi(distance_cutoff_i[index_j])

    # search for the max density
    Max = np.max(density)
    MaxIndexList = list()
    for index_i in range(len(labels)):
        if density[index_i] == Max:
            MaxIndexList.extend([index_i])

    # computer distance_higherDensity
    Min = 0
    for index_i in range(len(labels)):
        if index_i in MaxIndexList:
            distance_higherDensity[index_i] = np.max(distance[index_i])
            continue
        else:
            Min = np.max(distance[index_i])
        for index_j in range(1, len(labels)):
            if density[index_i] < density[index_j] and distance[index_i, index_j] < Min:
                Min = distance[index_i, index_j]
            else:
                continue
        distance_higherDensity[index_i] = Min

    return density, distance_higherDensity, cutoff, distance, distance_sort


def load_bert_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        result = json.load(f)
        t = {}
        tmp = []
        title = []
        abstract = []
        for i in range(len(result)):
            t[result[i]["id"]] = np.array(result[i]["cls"])
            tmp.append(np.array(result[i]["cls"]))
            title.append(np.array(result[i]["title"]))
            abstract.append(np.array(result[i]["abstract"]))
        result = t
        n = 0
        return result, tmp, title, abstract

def sol():
    # _, tmp, title, abstract = load_bert_data("hk_news_200924_bert-512.json")

    _, tmp, title, abstract = load_bert_data("hk_news_200924_bert-title-512.json")
    density, distance_higherDensity, cut_off, distance, _ = fit(tmp, t=0.001, distanceMethod='cos')
    center_score = []
    for i in range(len(density)):
        t = density[i] * distance_higherDensity[i]
        center_score.append(t)
    result = cluster(distance, center_score, cut_off)
    for k in result:
        t = result[k]
        print('***'*10)
        for i in t:
            print(title[i], i)
    clussCounts = []
    tmp = []
    for k in result:
        t = result[k]
        if len(t) < 4:
            continue
        tmp.append(t)
        clussCounts.append(len(t))
        print('***' * 10)
        for i in t:
            print(title[i])
    print(len(tmp))


def load_stopwords(stopwordlist=["stopwords/cn_stopwords.txt", "stopwords/hit_stopwords.txt"]):
    stopwords = set()
    for f_name in stopwordlist:
        with open(f_name, 'r', encoding='utf8') as f:
            for line in f :
                t = line.strip()
                if t not in stopwords:
                    stopwords.add(t)
    stopwords.add(' ')
    stopwords.add('\n')
    stopwords.add(u'\u3000')
    stopwords.add(u'\ufeff')
    stopwords.add(u'\xad')
    return stopwords


def cluster(distance, center_score, cut_off):
    clusters = defaultdict(list)
    cluster_index = 0
    visited = [0] * len(center_score)
    while 1:
        Max = max(center_score)
        center_index = []
        for i, v in enumerate(center_score):
            if v == Max:
                center_index = i
                center_score[i] = -1
                visited[i] = 1
                break
        for j, dis in enumerate(distance[center_index]):
            if visited[j]:
                continue
            if dis <= cut_off:
                clusters[cluster_index].append(j)
                visited[j] = 1
        if len(clusters[cluster_index]) <=1:
            clusters[cluster_index] = []
        else:
            cluster_index += 1
        if sum(visited) == len(visited):
            break
    return clusters


def load_json_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        result = json.load(f)
        title = []
        abstract = []

        stopwords = load_stopwords()
        for i in range(len(result)):
            title.append((result[i]["title"]))
            abstract.append((result[i]["abstract"][:200]))
        sentences_title = []
        sentences_doc = []
        for i, v in enumerate(title):
            t = (jieba.cut(v))
            tmp = []
            for word in t:
                if word in stopwords or word.isdigit() or len(word) <= 1:
                    continue
                tmp.append(word)
            sentences_title.append(tmp)

            t = (jieba.cut(abstract[i]))
            tmp2 = []
            for word in t:
                if word in stopwords or word.isdigit() or len(word) <= 1:
                    continue
                tmp2.append(word)
            sentences_doc.append(tmp2)
        return sentences_title, sentences_doc, stopwords, title


def load_txt_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        title = []
        abstract = []
        labels = []
        stopwords = load_stopwords()
        for line in f:
            line = line.split('\t')
            title.append(line[0])
            abstract.append(line[1][:200])
            labels.append(" ".join(line[2:4]))
        sentences_title = []
        sentences_doc = []
        for i, v in enumerate(title):
            t = (jieba.cut(v))
            tmp = []
            for word in t:
                if word in stopwords or word.isdigit() or len(word) <= 1:
                    continue
                tmp.append(word)
            sentences_title.append(tmp)

            t = (jieba.cut(abstract[i]))
            tmp2 = []
            for word in t:
                if word in stopwords or word.isdigit() or len(word) <= 1:
                    continue
                tmp2.append(word)
            sentences_doc.append(tmp2)
        return sentences_title, sentences_doc, stopwords, title, labels



def load_tfidf_data(file):
        sentences_title, sentences_doc, stopwords, title, titleid = load_json_data2(file)
        document = []
        doc_title = [" ".join(sent0) for sent0 in sentences_title]
        doc_abstract = [" ".join(sent0) for sent0 in sentences_doc]
        for i in range(len(doc_title)):
            document.append(doc_title[i] + ' ' + doc_abstract[i])
        print(f" {len(document)}  document")
        stopwords = list(stopwords)
        print('start fit model  and transformer')
        s = time.time()

        tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords, max_features=10000).fit(document)


        document = []
        for i in range(len(doc_title)):
            t = []
            for _ in range(3):
                t.append(doc_title[i])
            t.append(doc_abstract[i])
            document.append(' '.join(t))


        sparse_result = tfidf_model.transform(document)  # 得到tf-idf矩阵，稀疏矩阵表示法
        sparse_result = csr2dict(sparse_result)
        print(f'end fit and transformer  {time.time()-s}')


        density, distance_higherDensity, cut_off, distance, distance_sort = fit(sparse_result, t=0.0006, distanceMethod='sparse')
        print(cut_off)
        center_score = []
        for i in range(len(density)):
            t = density[i] * distance_higherDensity[i]
            center_score.append(t)
        print('start dpeak')
        s1 = time.time()

        result = cluster(distance, center_score, cut_off)
        print(f'end dpeak {time.time()- s1}')
        clussCounts = []
        tmp = []
        stopwords = set(stopwords)
        print(f'end program {time.time()-s}')

        for k in result:
            t = result[k]
            cluster_titles = []
            if len(t) < 4:
                continue
            for i in t:
                cluster_titles.append(title[i])
            labels = title_df(cluster_titles, stopwords)
            tmp.append(t)
            clussCounts.append(len(t))
            titleids =  []
            print('***' * 10)
            for i in t:
                print(titleid[i])
                titleids.append(i)
            print(labels)
        print(len(tmp))
# #

def title_df(title_words, stopwords, threshold=0.6):
    title_words = list(map(set, title_words))
    word_df = {}
    start = 0
    l = len(title_words)
    while start < l:
        cur_title = (title_words[start])
        for k in cur_title:
            if k not in word_df and k not in stopwords:
                word_df[k] = 1
            else:
                continue
            for j in range(start+1, l):
                if k in title_words[j]:
                    word_df[k] += 1
        start += 1
    labels = []
    for k in word_df:
        if word_df[k]/l>= threshold:
            labels.append(k)
    return labels


def csr2dict(sparse_result):
    sparse_matrix = {}
    for i in range(sparse_result.shape[0]):
        t = sparse_result[i].toarray().squeeze(0).tolist()
        tmp = {}
        for j, v in enumerate(t):
            if v != 0:
                tmp[j] = v
        sparse_matrix[i] = tmp
    return sparse_matrix

# load_tfidf_data('20200928-news.txt')


def load_json_data2(file):
    with open(file, 'r', encoding='utf-8') as f:
        result = []
        for line in f.readlines():
            dic = json.loads(line)
            result.append(dic)
        title = []
        abstract = []
        titleids = []
        stopwords = load_stopwords()
        for i in range(len(result)):
            title.append((result[i]["title_words"]))
            abstract.append((result[i]["cont_words"]))
            titleids.append((result[i]["id"]))
        sentences_title = []
        sentences_doc = []
        for i, v in enumerate(title):
            tmp = []
            for word in v:
                if word in stopwords or word.isdigit() or len(word) <= 1:
                    continue
                tmp.append(word)
            sentences_title.append(tmp)

            t = abstract[i]
            tmp2 = []
            for word in t:
                if word in stopwords or word.isdigit() or len(word) <= 1:
                    continue
                tmp2.append(word)
            sentences_doc.append(tmp2)
        return sentences_title, sentences_doc, stopwords, title, titleids


load_tfidf_data("hk_news_200924_row.json")