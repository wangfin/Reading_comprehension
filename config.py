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

        # 输出文件夹
        output_dir = os.path.join(data, "output")

        # 模型保存文件夹
        model_dir = os.path.join(data, "models")

        # 统计，tensorboard页面
        summary_dir = os.path.join(data, "summary")

        # vocab dir
        vocab_dir = os.path.join(data, "vocab")

        # 日志文件
        log_file = os.path.join(data, "log", "dev_log.txt")

        # 预训练的中文词向量
        vector_file = os.path.join(data, "vector", "sgns.merge.bigram")
        # 字向量的文件
        char_vector_file = os.path.join(data, "vector", "sgns.char.dim300.iter5")


        # tf.app.flags 是用于接受命令行传来的参数
        return tf.contrib.training.HParams(
            zhidao_train_file=zhidao_train_file,
            zhidao_dev_file=zhidao_dev_file,
            zhidao_test_file=zhidao_test_file,
            search_train_file=search_train_file,
            search_dev_file=search_dev_file,
            search_test_file=search_test_file,
            vector_file=vector_file,
            char_vector_file=char_vector_file,
            log_file=log_file,
            output_dir=output_dir,
            model_dir=model_dir,
            summary_dir=summary_dir,
            vocab_dir=vocab_dir
        )

    # 定义超参数
    def get_default_params(self):
        return tf.contrib.training.HParams(
            # 最大的passage数量
            max_p_num=5,
            # passage长度
            max_p_len=400,
            # question长度
            max_q_len=60,
            # answer长度
            max_a_len=200,
            # 单词的最大字符长度
            max_ch_len=20,
            # 词向量维度
            word_embed_size=300,
            # 字向量维度
            char_embed_size=300,
            # 词典长度
            vocab_size=1285531,
            # dropout
            keep_prob=0.2,
            # ptr_dropout
            ptr_keep_prob=0.7,
            # hidden size
            hidden_size=75,
            # char hidden
            char_hidden=100,
            # attention size
            attn_size=75,
            # batch size
            batch_size=64,
            # epoch
            epoch=100,
            # 优化函数
            opt_arg={'adadelta': {'learning_rate': 1, 'rho': 0.95, 'epsilon': 1e-6},
                     'adam': {'learning_rate': 1e-3, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8},
                     'gradientdescent': {'learning_rate': 1},
                     'adagrad': {'learning_rate': 1}},
            # 是否使用 cudnn
            use_cudnn=True,
            # 全局梯度削减速率
            grad_clip=5.0
        )


if __name__ == '__main__':
    con = Config()
    print(con.get_default_params().opt_arg['adam']['learning_rate'])




