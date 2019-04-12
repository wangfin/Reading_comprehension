#!/usr/bin/env python
# @Time    : 2019/4/2 16:50
# @Author  : wb
# @File    : config.py

import os
import tensorflow as tf

'''
实现生成配置信息
'''
class Config:

    # 文件参数
    def get_filepath(self):
        '''
            所有的文件都在data文件夹中
            如zhidao的数据集就在zhidao文件夹中
        '''
        # 训练文件文件名
        data = os.path.expanduser("data")
        zhidao_train_file = os.path.join(data, "zhidao", "zhidao.train.json")
        zhidao_dev_file = os.path.join(data, "zhidao", "zhidao.dev.json")
        zhidao_test_file = os.path.join(data, "zhidao", "zhidao.test.json")

        search_train_file = os.path.join(data, "search", "search.train.json")
        search_dev_file = os.path.join(data, "search", "search.dev.json")
        search_test_file = os.path.join(data, "search", "search.test.json")

        # 输出文件
        output_file = os.path.join(data, "output", "output.txt")

        # 日志文件
        log_file = os.path.join(data, "log", "dev_log.txt")

        # 预训练的中文词向量
        vector_file = os.path.join(data, "vector", "sgns.merge.bigram")

        # tf.app.flags 是用于接受命令行传来的参数
        return tf.contrib.training.HParams(
            zhidao_train_file=zhidao_train_file,
            zhidao_dev_file=zhidao_dev_file,
            zhidao_test_file=zhidao_test_file,
            search_train_file=search_train_file,
            search_dev_file=search_dev_file,
            search_test_file=search_test_file,
            vector_file=vector_file,
            log_file=log_file
        )

    # 定义超参数
    def get_default_params(self):
        return tf.contrib.training.HParams(
            # 词向量维度
            embed_size=300,
            # 词典长度
            vocab_size=1285531
        )


