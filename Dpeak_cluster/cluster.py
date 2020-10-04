import numpy as np
import json
import jieba
import time
import math
import os

from collections import OrderedDict
from pprint import pprint
from sklearn.datasets import make_blobs
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


class extract_label:
    def __init__(self, input_file=None, input_dict=None, inputdata_segment=False, inputdata_has_label=False, inputdata_is_json=False,
                 stopwordlist=["stopwords/cn_stopwords.txt", "stopwords/hit_stopwords.txt"], dis_cutoff_ratio=0.0006,
                 threshold_dict=None, distance_faster=False, pre_cutoff = 1.05):
        self.stopwordlist=stopwordlist
        self.distance_faster = distance_faster
        self.pre_cutoff = pre_cutoff
        self.stopwords = self.load_stopwords()

        if input_dict:
            self.title_segment, self.doc_segment, self.title, self.labels = self.load_input_dict(input_dict)
        elif not input_file:
            print("inputfile and inputdict can not be None at same time")
            return
        elif not inputdata_has_label and not inputdata_segment and inputdata_is_json:
            self.title_segment, self.doc_segment, self.title, self.labels = \
                self.load_json_data_without_label_without_segmentation(input_file)
        elif inputdata_has_label and not inputdata_segment and not inputdata_is_json:
            self.title_segment, self.doc_segment, self.title, self.labels = \
                self.load_txt_data_with_label_without_segmentation(input_file)
        elif not inputdata_has_label and inputdata_segment and inputdata_is_json:
            self.title_segment, self.doc_segment, self.title, self.labels = \
                self.load_json_data_without_label_with_segmentation(input_file)
            # self.title_segment, self.doc_segment, self.title, self.labels = \
            #     self.load_json_data_without_label_with_segmentation1(input_file)
        else:
            print("input func not Implement ")


        self.dis_cutoff_ratio = dis_cutoff_ratio
        if not threshold_dict:
            self.threshold_dict= OrderedDict()
            self.threshold_dict[5] = 0.6
            self.threshold_dict[8] = 0.55
            self.threshold_dict[15] = 0.5
            self.threshold_dict[30] = 0.4
            self.threshold_dict[200] = 0.3
        else:
            self.threshold_dict = threshold_dict




    def load_input_dict(self, inputdict):
        title_segment = []
        doc_segment = []
        title_id = []
        stopwords = self.stopwords
        for i, v in inputdict.items():
            t = v[0]
            tmp = []
            for word in t:
                if word in stopwords or word.isdigit() or len(word) <= 1:
                    continue
                tmp.append(word)
            title_segment.append(tmp)
            t = v[1]
            tmp2 = []
            for word in t:
                if word in stopwords or word.isdigit() or len(word) <= 1:
                    continue
                tmp2.append(word)
            doc_segment.append(tmp2)
            title_id.append(i)
        return title_segment, doc_segment, title_id, None



    def load_txt_data_with_label_without_segmentation(self, file):
        stopwords = self.stopwords
        with open(file, 'r', encoding='utf-8') as f:
            title = []
            abstract = []
            labels = []
            for line in f:
                line = line.split('\t')
                title.append(line[0])
                abstract.append(line[1][:200])
                labels.append(" ".join(line[2:4]))
            title_segment = []
            doc_segment = []
            for i, v in enumerate(title):
                t = (jieba.cut(v))
                tmp = []
                for word in t:
                    if word in stopwords or word.isdigit() or len(word) <= 1:
                        continue
                    tmp.append(word)
                title_segment.append(tmp)

                t = (jieba.cut(abstract[i]))
                tmp2 = []
                for word in t:
                    if word in stopwords or word.isdigit() or len(word) <= 1:
                        continue
                    tmp2.append(word)
                doc_segment.append(tmp2)
            return title_segment, doc_segment, title, labels


    def load_json_data_without_label_without_segmentation(self, file):
        stopwords = self.stopwords
        with open(file, 'r', encoding='utf-8') as f:
            result = json.load(f)
            title = []
            abstract = []
            for i in range(len(result)):
                title.append((result[i]["title"]))
                abstract.append((result[i]["abstract"][:200]))
            title_segment = []
            doc_segment = []
            for i, v in enumerate(title):
                t = (jieba.cut(v))
                tmp = []
                for word in t:
                    if word in stopwords or word.isdigit() or len(word) <= 1:
                        continue
                    tmp.append(word)
                title_segment.append(tmp)

                t = (jieba.cut(abstract[i]))
                tmp2 = []
                for word in t:
                    if word in stopwords or word.isdigit() or len(word) <= 1:
                        continue
                    tmp2.append(word)
                doc_segment.append(tmp2)
            return title_segment, doc_segment, title, None


    #  使用该方法读取数据时不会对 cont_words部分进行截断，截取长度为200个词，在hk_news_200924_row.json上做测试
    #  能聚出来20个类（每个类中最少包含4条新闻）
    def load_json_data_without_label_with_segmentation(self, file):

        with open(file, 'r', encoding='utf-8') as f:
            result = []
            for line in f.readlines():
                dic = json.loads(line)
                result.append(dic)
            title = []
            abstract = []
            titleids = []
            stopwords = self.stopwords
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
            return sentences_title, sentences_doc, titleids, None


    #  使用该方法读取数据时会对 cont_words部分进行截断，截取长度为200个词，在hk_news_200924_row.json上做测试的时候会导致
    #  结果变差，原先能聚出来20个类（每个类中最少包含4条新闻），使用这个方法就只能聚出来9个类（每个类中最少包含4条新闻）
    def load_json_data_without_label_with_segmentation1(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            result = []
            for line in f.readlines():
                dic = json.loads(line)
                result.append(dic)
            title = []
            abstract = []
            titleids = []
            stopwords = self.stopwords
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
                doc_len = 0
                for word in t:
                    if word in stopwords or word.isdigit() or len(word) <= 1:
                        continue
                    doc_len += len(word)
                    if doc_len>200:
                        break
                    tmp2.append(word)
                sentences_doc.append(tmp2)
            return sentences_title, sentences_doc, titleids, None


    def load_stopwords(self):
        stopwords = set()
        for f_name in self.stopwordlist:
            with open(f_name, 'r', encoding='utf8') as f:
                for line in f:
                    t = line.strip()
                    if t not in stopwords:
                        stopwords.add(t)
        stopwords.add(' ')
        stopwords.add('\n')
        stopwords.add(u'\u3000')
        stopwords.add(u'\ufeff')
        stopwords.add(u'\xad')
        return stopwords


    def chi(self, x):
        if x < 0:
            return 1
        else:
            return 0


    def csr2dict(self, sparse_result):
        sparse_matrix = {}
        for i in range(sparse_result.shape[0]):
            t = sparse_result[i].toarray().squeeze(0).tolist()
            tmp = {}
            for j, v in enumerate(t):
                if v != 0:
                    tmp[j] = v
            sparse_matrix[i] = tmp
        return sparse_matrix


    def distanceNorm(self, Norm, D_value, feature1=None, feature2=None):
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
            counter /= (counter1 * counter2)
            counter = 1 - counter
        else:
            raise Exception('We will program this later......')
        return counter


    def distanceNormSparse(self, d1, d2):
        if self.distance_faster:
            result = 0
            for k in d1:
                if k not in d2:
                    result += pow(d1[k], 2)
                else:
                    result += pow(d1[k] - d2[k], 2)
                if result > self.pre_cutoff:
                    return result
            for k in d2:
                if k not in d1:
                    result += pow(d2[k], 2)
                    if result > self.pre_cutoff:
                        return result
            return math.sqrt(result)
        else:
            result = 0
            for k in d1:
                if k not in d2:
                    result += pow(d1[k], 2)
                else:
                    result += pow(d1[k] - d2[k], 2)
            for k in d2:
                if k not in d1:
                    result += pow(d2[k], 2)
            return math.sqrt(result)





    # dis_cutoff_ratio 代表选取截断距离比例，t越大，密度聚类中的eps距离越大
    def calculate_dis(self, features, inputnums, dis_cutoff_ratio=0.006, distanceMethod='sparse'):
        distance = np.zeros((inputnums, inputnums))
        distance_sort = list()
        for index_i in range(inputnums):
            for index_j in range(index_i + 1, inputnums):
                if distanceMethod == 'sparse':
                    distance[index_i, index_j] = self.distanceNormSparse(features[index_i], features[index_j])
                else:
                    D_value = features[index_i] - features[index_j]
                    distance[index_i, index_j] = self.distanceNorm(distanceMethod, D_value, features[index_i],
                                                              features[index_j])
                distance_sort.append(distance[index_i, index_j])
        distance += distance.T
        # compute optimal cutoff
        distance_sort.sort()
        # pprint((len(distance_sort) * t))
        cutoff = distance_sort[int(len(distance_sort) * dis_cutoff_ratio)]
        distance_sort = np.array(distance_sort)
        return distance, distance_sort, cutoff


    def fit(self, features, dis_cutoff_ratio=0.0006, distanceMethod='sparse'):
        # initialization
        inputnums = len(features)
        distance, distance_sort, cutoff = self.calculate_dis(features=features, inputnums=inputnums,
                                                             dis_cutoff_ratio=dis_cutoff_ratio, distanceMethod=distanceMethod)
        inputnums = len(features)
        density = np.zeros(inputnums)
        distance_higherDensity = np.zeros(inputnums)
        # computer density
        for index_i in range(inputnums):
            distance_cutoff_i = distance[index_i] - cutoff
            for index_j in range(1, inputnums):
                density[index_i] += self.chi(distance_cutoff_i[index_j])

        # search for the max density
        Max = np.max(density)
        MaxIndexList = list()
        for index_i in range(inputnums):
            if density[index_i] == Max:
                MaxIndexList.extend([index_i])

        # computer distance_higherDensity
        for index_i in range(inputnums):
            if index_i in MaxIndexList:
                distance_higherDensity[index_i] = np.max(distance[index_i])
                continue
            else:
                Min = np.max(distance[index_i])
            for index_j in range(1, inputnums):
                if density[index_i] < density[index_j] and distance[index_i, index_j] < Min:
                    Min = distance[index_i, index_j]
            distance_higherDensity[index_i] = Min

        return density, distance_higherDensity, cutoff, distance, distance_sort


    def cluster(self, distance, center_score, cut_off):
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
            if len(clusters[cluster_index]) <= 1:
                clusters[cluster_index] = []
            else:
                cluster_index += 1
            if sum(visited) == len(visited):
                break
        return clusters


    def cal_distance_matrix_and_save(self, filedir, title_duplicate_nums=3, distanceMethod='sparse'):
        dis_cutoff_ratio = self.dis_cutoff_ratio
        stopwords = list(self.stopwords)
        document = []
        doc_title = [" ".join(sent0) for sent0 in self.title_segment]
        doc_abstract = [" ".join(sent0) for sent0 in self.doc_segment]
        for i in range(len(doc_title)):
            document.append(doc_title[i] + ' ' + doc_abstract[i])
        tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords, max_features=10000).fit(document)
        document = []
        for i in range(len(doc_title)):
            tmp = []
            for _ in range(title_duplicate_nums):
                tmp.append(doc_title[i])
            tmp.append(doc_abstract[i])
            document.append(' '.join(tmp))

        sparse_result = tfidf_model.transform(document)  # 得到tf-idf矩阵，稀疏矩阵表示法
        sparse_result = self.csr2dict(sparse_result)

        density, distance_higherDensity, cut_off, distance, distance_sort = self.fit(sparse_result, dis_cutoff_ratio, distanceMethod)
        np.save(os.path.join(filedir, 'density.npy'), density)
        np.save(os.path.join(filedir, 'distance_higherDensity.npy'), distance_higherDensity)
        np.save(os.path.join(filedir, 'distance.npy'), distance)
        np.save(os.path.join(filedir, 'distance_sort.npy'), distance_sort)
        d = {'cut_off':cut_off}
        with open(os.path.join(filedir, 'cut_off.json'), 'w', encoding='utf8') as f:
            json.dump(d, f, ensure_ascii=False)
        d = {}
        for i, k in enumerate(self.title):
            d[i] = k
        with open(os.path.join(filedir, 'index2id.json'), 'w', encoding='utf8') as f:
            json.dump(d, f, ensure_ascii=False)



    def cluster_from_pretrain(self, filedir, select_index=None, online=False):
        density = np.load(os.path.join(filedir, 'density.npy'))
        distance_higherDensity = np.load(os.path.join(filedir, 'distance_higherDensity.npy'))
        distance = np.load(os.path.join(filedir, 'distance.npy'))
        print(density.shape)
        with open(os.path.join(filedir, 'cut_off.json'), 'r', encoding='utf8') as f:
            d = json.load(f)
        cut_off = d["cut_off"]
        center_score = []
        for i in range(len(density)):
            t = density[i] * distance_higherDensity[i]
            center_score.append(t)
        result = self.cluster(distance, center_score, cut_off)
        result = self.filter_result(result)
        if not online:
            return result
        tmp = {}
        for k, v in result.items():
            kk = '_'.join(v['labels'])
            tmp[kk] = (v['id'], len(v['id']))
        tmp = sorted(tmp.items(), key=lambda x: x[1][1], reverse=True)
        result = OrderedDict()
        for t in tmp:
            result[t[0]] = t[1][0]
        return result


    def sol(self, title_duplicate_nums=3, distanceMethod='sparse', online=True):
        dis_cutoff_ratio = self.dis_cutoff_ratio
        stopwords = list(self.stopwords)
        document = []
        doc_title = [" ".join(sent0) for sent0 in self.title_segment]
        doc_abstract = [" ".join(sent0) for sent0 in self.doc_segment]
        for i in range(len(doc_title)):
            document.append(doc_title[i] + ' ' + doc_abstract[i])
        tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords, max_features=10000).fit(document)
        document = []
        for i in range(len(doc_title)):
            tmp = []
            for _ in range(title_duplicate_nums):
                tmp.append(doc_title[i])
            tmp.append(doc_abstract[i])
            document.append(' '.join(tmp))

        sparse_result = tfidf_model.transform(document)  # 得到tf-idf矩阵，稀疏矩阵表示法
        sparse_result = self.csr2dict(sparse_result)

        density, distance_higherDensity, cut_off, distance, distance_sort = self.fit(sparse_result, dis_cutoff_ratio, distanceMethod)
        center_score = []
        for i in range(len(density)):
            t = density[i] * distance_higherDensity[i]
            center_score.append(t)
        result = self.cluster(distance, center_score, cut_off)
        result = self.filter_result(result)
        if not online:
            return result
        tmp = {}
        for k, v in result.items():
            kk = '_'.join(v['labels'])
            tmp[kk] = (v['id'], len(v['id']))
        tmp = sorted(tmp.items(), key=lambda x: x[1][1], reverse=True)
        result = OrderedDict()
        for t in tmp:
            result[t[0]] = t[1][0]
        return result


    def title_df(self, title_words, stopwords, threshold=0.6):
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
                for j in range(start + 1, l):
                    if k in title_words[j]:
                        word_df[k] += 1
            start += 1
        labels = []
        for k in word_df:
            if word_df[k] / l >= threshold:
                labels.append(k)
        return labels


    def filter_result(self, result, threshold=4):
        tmp = {}
        index = 0
        for k in result:
            t = result[k]
            cluster_titles = []
            if len(t) < threshold or len(t)> 50:
                continue
            for i in t:
                cluster_titles.append(self.title_segment[i])
            labels = self.title_df(cluster_titles, self.stopwords)
            titleids = []
            for i in t:
                titleids.append(self.title[i])
            tmp[index] = {'labels': labels,
                             'id': titleids}
            index += 1
        return tmp


    def extract_label_baseon_title_df(self, title_words, stopwords):
        threshold_dict = self.threshold_dict
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
                for j in range(start + 1, l):
                    if k in title_words[j]:
                        word_df[k] += 1
            start += 1
        labels = []
        threshold = 0.25
        for k in threshold_dict:
            if l<=k:
                threshold = threshold_dict[k]
        for k in word_df:
            if word_df[k] / l >= threshold:
                labels.append(k)
        return labels



#
if __name__ == "__main__":

    import  time
    start = time.time()
    s = extract_label(input_file="hk_news_200924_row.json", inputdata_segment=True, inputdata_has_label=False,
                      inputdata_is_json=True, dis_cutoff_ratio=0.0006, distance_faster=True)

    # # s = extract_label(input_file="hk_news_200924.json", inputdata_segment=False, inputdata_has_label=False,
    # # #                   inputdata_is_json=True, dis_cutoff_ratio=0.0003)
    #
    t = s.cluster_from_pretrain(filedir='test')
    # # pprint(t)
    # print(time.time()-start)


    #
    # s = extract_label(input_file="test.txt", inputdata_segment=False, inputdata_has_label=True,
    #                   inputdata_is_json=False, dis_cutoff_ratio=0.01, distance_faster=False)
    # t = s.sol()
    # pprint(t)