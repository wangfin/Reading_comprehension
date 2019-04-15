#!/usr/bin/env python
# @Time    : 2019/4/7 20:38
# @Author  : wb
# @File    : vocab.py

# 将单词转换成id，将id转换成单词
import numpy as np
from comprehension.config import Config

'''
实现生成词表，读取预训练词向量
用于将字符转换成id，将id转换成字符
输入initial_tokens，返回embeddings
'''
class Vocab(object):

    config = Config()

    def __init__(self, initial_tokens=None, lower=False):
        # id->token 存放的是id:token
        self.id2token = {}
        # token->id 存放的是token:id
        self.token2id = {}
        # token的统计词数
        self.token_cnt = {}
        self.lower = lower

        # 填充标签，将question和passage填充到一定的长度
        self.pad_token = '<PAD>'
        # 如果在词表中不存在，返回UNK
        self.unk_token = '<UNK>'

        # 词向量维度、词表长度
        self.embed_size = self.config.get_default_params().embed_size
        self.voacb_size = self.config.get_default_params().vocab_size
        # self.embeddings = {}

        # 这个initial_tokens应该是初始传入的tokens
        # 如果为空就是空，不为空就是这个
        self.initial_tokens = initial_tokens if initial_tokens is not None else []
        # 添加补全和空
        self.initial_tokens.extend([self.pad_token, self.unk_token])
        # 所以只要输入initial_tokens，就能自己把tokens和ids存入词典中
        for token in self.initial_tokens:
            self.add(token)

    # 获取词典的大小
    def get_vocab_size(self):
        if len(self.id2token) == 0:
            return self.voacb_size
        else:
            return len(self.id2token)

    # 获取token的id
    def get_id(self, token):
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    # 获取id对应的token
    def get_token(self, idx):
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    # 添加token到词典
    def add(self, token, cnt=1):
        # 需要小写就小写
        token = token.lower() if self.lower else token
        # 如果有就查询出id
        if token in self.token2id:
            idx = self.token2id[token]
        # 没有就在词表中添加
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx
        # 如果词表中有这个token，那么就把这个次数加上去
        # 如果没有，赋值为1
        if cnt > 0:
            if token in self.token_cnt:
                self.token_cnt[token] += cnt
            else:
                self.token_cnt[token] = cnt
        return idx

    # 按计数过滤词汇中的标记
    def filter_tokens_by_cnt(self, min_cnt):
        filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)

    # 随机初始化token的词向量
    def randomly_init_embeddings(self, embedding_size=None):
        if embedding_size:
            self.embed_size = embedding_size
        self.embeddings = np.random.rand(self.get_vocab_size(), self.embed_size)
        # 当单词不在词典中
        for token in [self.pad_token, self.unk_token]:
            self.embeddings[self.get_id(token)] = np.zeros([self.embed_size])

    # 读取预训练的中文词向量
    def load_pretrained_embeddings(self, embedding_size=None):
        if embedding_size:
            self.embed_size = embedding_size
        trained_embeddings = {}
        with open(self.config.get_filepath().vector_file, 'r', encoding='utf-8') as f:
            # 跳过第一行，因为第一行是词表的元信息
            next(f)
            while True:
                line = f.readline()  # 逐行读取
                if not line:  # 如果读取到文件末尾
                    break

                token = line.split()[0]
                contents = line.split()[1:]
                if token not in self.token2id:
                    continue
                # 存入字典，token为key，词向量为value
                trained_embeddings[token] = contents
                filtered_tokens = trained_embeddings.keys()
                # 重新构建token到id的映射
                self.token2id = {}
                self.id2token = {}
                # 把initial_tokens里面的token放进词典中
                for token in self.initial_tokens:
                    self.add(token, cnt=0)
                # 把预训练词向量中的token也放入词表
                for token in filtered_tokens:
                    self.add(token, cnt=0)
                # embeddings
                self.embeddings = np.zeros([self.get_vocab_size(), self.embed_size])
                # 在词表中的token
                for token in self.token2id.keys():
                    # 如果token在预训练的词向量的token中
                    if token in trained_embeddings.keys():
                        # 获取词表中该token的id
                        # 把这个id和预训练词向量中的token的词向量存入embeddings中
                        self.embeddings[self.get_id(token)] = trained_embeddings[token]

    # 将tokens转换成id，如果token不在词典，使用unk
    def convert_to_ids(self, tokens):
        vec = [self.get_id(label) for label in tokens]
        return vec

    # 将id转换成token，如果遇到停止符就停止转换
    def recover_from_ids(self, ids, stop_id=None):
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens


