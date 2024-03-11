import json
import os
import requests
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from nltk.corpus import stopwords
from tqdm import tqdm
from collections import defaultdict
from string import punctuation
import numpy as np
import torch
import torch.nn as nn
import spacy
from spacy.lang.zh.stop_words import STOP_WORDS
nlp = spacy.load('zh_core_web_sm')


class TheLittlePrinceDataset:
    def __init__(self, tokenize=True):
        # 利用nltk函数进行分句和分词
        text = open('the little prince.txt', 'r', encoding='utf-8').read()
        if tokenize:
            self.sentences = sent_tokenize(text.lower())
            self.tokens = [word_tokenize(sent) for sent in self.sentences]
        else:
            self.text = text

    def build_vocab(self, min_freq=1):
        # 统计词频
        frequency = defaultdict(int)
        for sentence in self.tokens:
            for token in sentence:
                frequency[token] += 1
        self.frequency = frequency

        # 加入<unk>处理未登录词，加入<pad>用于对齐变长输入进而加速
        self.token2id = {'<unk>': 1, '<pad>': 0}
        self.id2token = {1: '<unk>', 0: '<pad>'}
        for token, freq in sorted(frequency.items(), key=lambda x: -x[1]):
            # 丢弃低频词
            if freq > min_freq:
                self.token2id[token] = len(self.token2id)
                self.id2token[len(self.id2token)] = token
            else:
                break

    def get_word_distribution(self):
        distribution = np.zeros(vocab_size)
        for token, freq in self.frequency.items():
            if token in dataset.token2id:
                distribution[dataset.token2id[token]] = freq
            else:
                # 不在词表中的词按<unk>计算
                distribution[1] += freq
        distribution /= distribution.sum()
        return distribution

    # 将分词结果转化为索引表示
    def convert_tokens_to_ids(self, drop_single_word=True):
        self.token_ids = []
        for sentence in self.tokens:
            token_ids = [self.token2id.get(token, 1) for token in sentence]
            # 忽略只有一个token的序列，无法计算loss
            if len(token_ids) == 1 and drop_single_word:
                continue
            self.token_ids.append(token_ids)
        
        return self.token_ids


class TFIDF:
    def __init__(self, vocab_size, norm='l2', smooth_idf=True, sublinear_tf=True):
        self.vocab_size = vocab_size
        self.norm = norm
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
    
    def fit(self, X):
        doc_freq = np.zeros(self.vocab_size, dtype=np.float64)
        for data in X:
            for token_id in set(data):
                doc_freq[token_id] += 1
        doc_freq += int(self.smooth_idf)
        n_samples = len(X) + int(self.smooth_idf)
        self.idf = np.log(n_samples / doc_freq) + 1
    
    def transform(self, X):
        assert hasattr(self, 'idf')
        term_freq = np.zeros((len(X), self.vocab_size), dtype=np.float64)
        for i, data in enumerate(X):
            for token in data:
                term_freq[i, token] += 1
        if self.sublinear_tf:
            term_freq = np.log(term_freq + 1)
        Y = term_freq * self.idf
        if self.norm:
            row_norm = (Y**2).sum(axis=1)
            row_norm[row_norm == 0] = 1
            Y /= np.sqrt(row_norm)[:, None]
        return Y
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class BooksDataset:
    def __init__(self):
        train_file, test_file = 'train.jsonl', 'test.jsonl'

        # 下载数据为JSON格式，转化为Python对象
        def read_file(file_name):
            with open(file_name, 'r', encoding='utf-8') as fin:
                json_list = list(fin)
            data_split = []
            for json_str in json_list:
                data_split.append(json.loads(json_str))
            return data_split

        self.train_data, self.test_data = read_file(train_file), read_file(test_file)
        print('train size =', len(self.train_data), ', test size =', len(self.test_data))
        
        # 建立文本标签和数字标签的映射
        self.label2id, self.id2label = {}, {}
        for data_split in [self.train_data, self.test_data]:
            for data in data_split:
                txt = data['class']
                if txt not in self.label2id:
                    idx = len(self.label2id)
                    self.label2id[txt] = idx
                    self.id2label[idx] = txt
                label_id = self.label2id[txt]
                data['label'] = label_id

    def tokenize(self, attr='book'):
        # 使用以下两行命令安装spacy用于中文分词
        # pip install -U spacy
        # python -m spacy download zh_core_web_sm
        # 去除文本中的符号和停用词
        for data_split in [self.train_data, self.test_data]:
            for data in tqdm(data_split):
                # 转为小写
                text = data[attr].lower()
                # 符号替换为空
                tokens = [t.text for t in nlp(text) if t.text not in STOP_WORDS]
                # 这一步比较耗时，因此把tokenize的结果储存起来
                data['tokens'] = tokens

    # 根据分词结果建立词表，忽略部分低频词，可以设置词最短长度和词表最大大小
    def build_vocab(self, min_freq=3, min_len=2, max_size=None):
        frequency = defaultdict(int)
        for data in self.train_data:
            tokens = data['tokens']
            for token in tokens:
                frequency[token] += 1 

        print(f'unique tokens = {len(frequency)}, total counts = {sum(frequency.values())}, '
              f'max freq = {max(frequency.values())}, min freq = {min(frequency.values())}')    

        self.token2id = {}
        self.id2token = {}
        total_count = 0
        for token, freq in sorted(frequency.items(), key=lambda x: -x[1]):
            if max_size and len(self.token2id) >= max_size:
                break
            if freq > min_freq:
                if (min_len is None) or (min_len and len(token) >= min_len):
                    self.token2id[token] = len(self.token2id)
                    self.id2token[len(self.id2token)] = token
                    total_count += freq
            else:
                break
        print(f'min_freq = {min_freq}, min_len = {min_len}, max_size = {max_size}, '
              f'remaining tokens = {len(self.token2id)}, '
              f'in-vocab rate = {total_count / sum(frequency.values())}')

    # 将分词后的结果转化为数字索引
    def convert_tokens_to_ids(self):
        for data_split in [self.train_data, self.test_data]:
            for data in data_split:
                data['token_ids'] = []
                for token in data['tokens']:
                    if token in self.token2id:
                        data['token_ids'].append(self.token2id[token])


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True, diagonal=False):
        super(Biaffine, self).__init__()
        # n_in：输入特征大小
        # n_out：输出的打分数量（边预测为1，标签预测即为标签数量）
        # bias_x：为输入x加入线性层
        # bias_y：为输入y加入线性层
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.diagonal = diagonal
        # 对角线化参数，让原本的参数矩阵变成了对角线矩阵，从而大幅度减少运算复杂度，一般在计算标签的得分时会使用
        if self.diagonal:
            self.weight = nn.Parameter(torch.Tensor(n_out,
                                                    n_in + bias_x))
        else:
            self.weight = nn.Parameter(torch.Tensor(n_out,
                                                    n_in + bias_x,
                                                    n_in + bias_y))
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x, y):
        # 当bias_x或bias_y为True时，为输入x或y的向量拼接额外的1
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        # torch中的einsum可以很简单的实现矩阵运算
        # 思路是为输入的张量的每个维度分别定义一个符号（例如输入x、y的第一维是批大小，定义为b）
        # 并且定义输出的张量大小，这个函数会自动地根据前后的变化计算张量乘法、求和的过程
        # 例如下面的bxi,byi,oi->boxy，表示的是输入的三个张量大小分别为b * x * i，b * y * i和o * i
        # 输出则是b * o * x * y
        # 根据这个式子，我们可以看出三个张量都有i这个维度，在输出时被消除了
        # 因此三个张量的i维通过张量乘法（三者按位相乘、然后求和）进行消除
        # 这个算法的好处是相比于手动实现，einsum可以更容易地避免运算过程中出现很大的张量大幅占用显存
        # 同时也避免了手动实现的流程
        # 具体使用方法请参考https://pytorch.org/docs/stable/generated/torch.einsum.html
        if self.diagonal:
            s = torch.einsum('bxi,byi,oi->boxy', x, y, self.weight)
        else:
            s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # 当n_out=1时，将第一维移除
        s = s.squeeze(1)

        return s
