#!/usr/bin/env python
# @Time    : 2019/4/7 16:49
# @Author  : wb
# @File    : pre_process.py

# 数据的预处理

'''
实现数据预处理
读取文件，对文件进行处理
切分batch
'''

import json
import numpy as np
from comprehension.config import Config
import logging
from collections import Counter

# 读取百度知道开发集

class Propress:
    df = Config()
    '''
    输入为  zhidao_dev,zhidao_train,zhidao_test ; search_dev,search_train,search_test  
    实现了用于加载和使用百度阅读理解数据
    输入：max_p_num, max_p_len, max_q_len,train_files=None, dev_files=None, test_files=None
    '''
    def __init__(self, max_p_num, max_p_len, max_q_len,
                 train_files=None, dev_files=None, test_files=None):

        # 获取名字为brc的记录器
        # logging.basicConfig(filename=self.df.get_filepath().log_file)
        self.logger = logging.getLogger("mcr")
        self.logger.setLevel(logging.DEBUG)
        # passage和question
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        # 获取三个数据集的内容
        if train_files:
            for train_file in train_files:
                self.train_set += self.load_datasets(train_file, train=True)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self.load_datasets(dev_file)
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self.load_datasets(test_file)
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

        # if filename == 'zhidao_dev':
        #     file = open(self.df.get_filepath().zhidao_dev_file, 'r', encoding='utf8')
        # elif filename == 'zhidao_train':
        #     file = open(self.df.get_filepath().zhidao_train_file, 'r', encoding='utf8')
        # elif filename == 'zhidao_test':
        #     file = open(self.df.get_filepath().zhidao_test_file, 'r', encoding='utf8')
        # elif filename == 'search_dev':
        #     file = open(self.df.get_filepath().search_dev_file, 'r', encoding='utf8')
        # elif filename == 'search_train':
        #     file = open(self.df.get_filepath().search_train_file, 'r', encoding='utf8')
        # elif filename == 'search_test':
        #     file = open(self.df.get_filepath().search_test_file, 'r', encoding='utf8')

    '''
    载入文件
    输入：filename，是否train
    输出：处理完的内容
    '''
    def load_datasets(self, filename, train=False):
        contents = []
        file = open(filename, 'r', encoding='utf8')

        # 只能使用一行行的读取方式，不然会报错
        for line in file.readlines():
            # dic = np.array(json.loads(line))
            # np.append(contents, dic)
            dic = json.loads(line.strip())
            # 是否为训练文件
            if train:
                # 不存在answer段落
                if len(dic['answer_spans']) == 0:
                    continue
                # 答案的长度大于我们设定的最大的passage的长度，这种也跳过
                if dic['answer_spans'][0][1] >= self.max_p_len:
                    continue

            # 把里面的answer_docs选取出来，answer_docs放的是answer中真正的答案，也就是fake_answers的内容
            # 这个值是一个int
            if 'answer_docs' in dic:
                dic['answer_passages'] = dic['answer_docs']

            # 获取切分好的问句
            dic['question_tokens'] = dic['segmented_question']

            # 设置文档
            dic['passages'] = []

            # 获取documents中的数据
            # 这个是一个集合的迭代器，前面是id，后面是document的内容
            for doc_idx, doc in enumerate(dic['documents']):
                # 进行训练
                if train:
                    # 这个数据集很好的为我们找到了在para中与问题最相近的答案，减少了我们去考虑过多的para内容
                    # 这个值是一个int，标注了哪一段最相近
                    most_related_para = doc['most_related_para']
                    dic['passages'].append({
                        'passage_tokens': doc['segmented_paragraphs'][most_related_para],  # 选取了分词好的最相关段落，一段
                        'is_selected': doc['is_selected']  # 是否被选取
                    })
                else:
                    # 不进行训练
                    para_infos = []
                    # 在分词后的段落列表中循环
                    for para_tokens in doc['segmented_paragraphs']:
                        # 获取分词的问题
                        question_tokens = dic['segmented_question']
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
                    dic['passages'].append({'passage_tokens': fake_passage_tokens})
            contents.append(dic)
        # print(contents[0])

        return contents

    '''
    获取一个mini batch
    输入：所有的数据，要选择的样本的索引，pad_id
    输出：一个batch数据
    '''
    def _one_mini_batch(self, data, indices, pad_id):
        # 定义一个batch的数据
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
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
                # 如果这个idx小于
                if pidx < len(sample['passages']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data, pad_p_len, pad_q_len



if __name__ == '__main__':
    df = Config()
    dev = df.get_filepath().zhidao_dev_file
    pro = Propress(100, 100, 100, dev_files=[dev])
