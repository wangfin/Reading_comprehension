#!/usr/bin/env python
# @Time    : 2019/4/7 16:49
# @Author  : wb
# @File    : pre_process.py

# 数据的预处理

import json
import numpy as np
from config import Config
import logging
from collections import Counter

# 读取百度知道开发集

'''
实现数据预处理
读取文件，对文件进行处理
切分batch
'''

class Propress(object):
    '''
    实现了用于加载和使用百度阅读理解数据
    输入：max_p_num, max_p_len, max_q_len,train_files=None, dev_files=None, test_files=None
    '''

    def __init__(self, max_p_num, max_p_len, max_q_len, max_char_len,
                 train_files=None, dev_files=None, test_files=None):

        # 获取名字为brc的记录器
        # logging.basicConfig(filename=self.df.get_filepath().log_file)
        self.logger = logging.getLogger("brc")
        # passage和question
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.max_char_len = max_char_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        # 获取三个数据集的内容
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_datasets(train_file, train=True)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_datasets(dev_file)
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_datasets(test_file)
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    '''
    载入文件
    输入：filename，是否train
    输出：处理完的内容
    '''
    def _load_datasets(self, filename, train=False):
        contents = []

        max_char_num = 0
        max_char_list = []

        with open(filename, 'r', encoding='utf8') as file:
            # 只能使用一行行的读取方式，不然会报错
            for line in file.readlines():
                sample = json.loads(line.strip())
                # 是否为训练文件
                if train:
                    # 不存在answer段落
                    if len(sample['answer_spans']) == 0:
                        continue
                    # 答案的长度大于我们设定的最大的passage的长度，这种也跳过
                    if sample['answer_spans'][0][1] >= self.max_p_len:
                        continue

                # 把里面的answer_docs选取出来，answer_docs放的是answer中真正的答案，也就是fake_answers的内容
                # 这个值是一个int
                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']

                # question_chars = [list(token) for token in dic['question_tokens']]
                # dic['question_chars'] = question_chars

                # 获取切分好的问句
                question_tokens = sample['segmented_question']
                sample['question_tokens'] = question_tokens
                question_chars = [list(token) for token in question_tokens]
                sample['question_chars'] = question_chars

                for char in sample['question_chars']:
                    if len(char) > max_char_num:
                        max_char_num = len(char)
                        max_char_list = char

                # 设置文档
                sample['passages'] = []

                # 获取documents中的数据
                # 这个是一个集合的迭代器，前面是id，后面是document的内容
                for doc_idx, doc in enumerate(sample['documents']):
                    # 进行训练
                    if train:
                        # 这个数据集很好的为我们找到了在para中与问题最相近的答案，减少了我们去考虑过多的para内容
                        # 这个值是一个int，标注了哪一段最相近
                        most_related_para = doc['most_related_para']

                        passage_tokens = doc['segmented_paragraphs'][most_related_para]
                        passage_chars = [list(token) for token in passage_tokens]

                        # passage_chars = [list(token) for token in doc['segmented_paragraphs'][most_related_para]]

                        for char in passage_chars:
                            if len(char) > max_char_num:
                                max_char_num = len(char)
                                max_char_list = char

                        sample['passages'].append({
                            'passage_tokens': passage_tokens,  # 选取了分词好的最相关段落，一段
                            'is_selected': doc['is_selected'],  # 是否被选取
                            'passage_chars': passage_chars
                        })
                    else:
                        # 不进行训练
                        para_infos = []
                        # 在分词后的段落列表中循环
                        for para_tokens in doc['segmented_paragraphs']:
                            para_tokens = para_tokens
                            # 获取分词的问题
                            question_tokens = sample['segmented_question']
                            # 先计数para_tokens，然后计数question_tokens，最后统计两个dict的交集
                            # 也就是获取段落tokens和问题tokens这两个中相同的tokens
                            common_with_question = Counter(para_tokens) & Counter(question_tokens)
                            # 对交集，也就是相同token出现次数进行求和
                            correct_preds = sum(common_with_question.values())
                            # 如果没有交集token

                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(correct_preds) / len(question_tokens)

                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))

                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:
                            fake_passage_tokens += para_info[0]

                        sample['passages'].append({'passage_tokens': fake_passage_tokens,
                                                   'passage_chars': [list(token) for token in fake_passage_tokens]})

                contents.append(sample)

            return contents

    '''
    获取一个mini batch
    输入：所有的数据，要选择的样本的索引，pad_id
    输出：一个batch数据
    '''
    def _one_mini_batch(self, data, indices, pad_id, pad_char_id):
        # 定义一个batch的数据
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_char_ids': [],
                      'passage_token_ids': [],
                      'passage_char_ids': [],
                      'start_id': [],
                      'end_id': []}
        # passages数组的长度，passages是之前处理完的数据，里面是存放的选择出来的passage（分词完的）
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        # 选择max_p_num和max_passage_num里面最小的值
        max_passage_num = min(self.max_p_num, max_passage_num)
        # 遍历batch_data中的raw_data
        for sidx, sample in enumerate(batch_data['raw_data']):
            # 使用max_passage_num
            for pidx in range(max_passage_num):
                # 如果这个idx小于passages的长度
                if pidx < len(sample['passages']):
                    # data 数据里面有question_tokens_ids
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_char_ids'].append(sample['question_char_ids'])
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    passage_char_ids = sample['passages'][pidx]['passage_char_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_char_ids'].append(passage_char_ids)
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_char_ids'].append([[]])
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_char_ids'].append([[]])

        # 填充完的数据
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id, pad_char_id)
        for sample in batch_data['raw_data']:
            # 如果有答案所在的段落标识并且长度大于0，也就是存在这个
            if 'answer_passages' in sample and len(sample['answer_passages']):
                # 填充段落长度 * 答案所在段落
                gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # 某些样品的假跨度，仅对测试有效
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        return batch_data

    '''
    使用pad_id动态填充batch_data
    输入：数据，pad_id
    输出：batch_data，passage的填充长度，question的填充长度
    '''
    def _dynamic_padding(self, batch_data, pad_id, pad_char_id):
        pad_char_len = self.max_char_len
        pad_p_len = self.max_p_len  # min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = self.max_q_len  # min(self.max_q_len, max(batch_data['question_length']))

        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]

        for index, char_list in enumerate(batch_data['passage_char_ids']):
            # print(batch_data['passage_char_ids'])
            for char_index in range(len(char_list)):
                if len(char_list[char_index]) >= pad_char_len:
                    char_list[char_index] = char_list[char_index][:self.max_char_len]
                else:
                    char_list[char_index] += [pad_char_id] * (pad_char_len - len(char_list[char_index]))
            batch_data['passage_char_ids'][index] = char_list
        batch_data['passage_char_ids'] = [(ids + [[pad_char_id] * pad_char_len] * (pad_p_len - len(ids)))[:pad_p_len]
                                          for ids in batch_data['passage_char_ids']]

        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]

        for index, char_list in enumerate(batch_data['question_char_ids']):
            for char_index in range(len(char_list)):
                if len(char_list[char_index]) >= pad_char_len:
                    char_list[char_index] = char_list[char_index][:self.max_char_len]
                else:
                    char_list[char_index] += [pad_char_id] * (pad_char_len - len(char_list[char_index]))
            batch_data['question_char_ids'][index] = char_list
        batch_data['question_char_ids'] = [(ids + [[pad_char_id] * pad_char_len] * (pad_q_len - len(ids)))[:pad_q_len]
                                           for ids in batch_data['question_char_ids']]

        return batch_data, pad_p_len, pad_q_len

    '''
    迭代数据集中所有的单词
    输入：set_name，如果设置就使用特殊的数据集
    输出：迭代器
    '''
    def word_iter(self, set_name=None):
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        yield token

    '''
    将原始数据集中的question和passage转换为ids
    输入：该数据集上的词汇表
    '''
    def convert_to_ids(self, vocab):
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_word_to_ids(sample['question_tokens'])
                sample["question_char_ids"] = vocab.convert_char_to_ids(sample['question_chars'])
                for passage in sample['passages']:
                    passage['passage_token_ids'] = vocab.convert_word_to_ids(passage['passage_tokens'])
                    passage['passage_char_ids'] = vocab.convert_char_to_ids(passage['passage_chars'])

    '''
    生成特定数据集的数据批次(train/dev/test)
    输入：数据集，batch的数量，pad_id，打乱数据集
    输出：batch的迭代器
    '''
    def next_batch(self, set_name, batch_size, pad_id, pad_char_id, shuffle=True):
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        # 在给定的时间间隔内返回均匀间隔的值。
        indices = np.arange(data_size)
        # 打乱数据集
        if shuffle:
            np.random.shuffle(indices)
        # 生成batch数据，开始0，结束data_size，步长batch_size
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            # 返回one_mini_batch
            yield self._one_mini_batch(data, batch_indices, pad_id, pad_char_id)


if __name__ == '__main__':
    df = Config()
    dev = df.get_filepath().zhidao_dev_file
    pro = Propress(100, 100, 100, 16, dev_files=[dev])
