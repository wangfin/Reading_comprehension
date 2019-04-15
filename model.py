#!/usr/bin/env python
# @Time    : 2019/3/26 8:52
# @Author  : wb
# @File    : model.py

# 模型文件

from comprehension.config import Config
import tensorflow as tf
from comprehension.layers import cudnn_gru, native_gru, dropout, dot_attention

'''
model文件
整个模型分为四个部分：
1.embedding编码部分，对passage和question进行word和char级别的编码
2.生成在question注意力下的passage编码，这一步主要是带着问题看文章
3.将question-aware-passage和passage进行匹配，选出其中的重要词语。同时也在全局收集匹配的证据和相关的段落
4.预测起始位和结束位，Pointer network
'''
class Model(object):

    # 引入文件、超参数等在config文件中
    config = Config()

    def __init__(self, batch, word_embeddings=None, char_embeddings=None, trainable=True, opt=True):
        # 定义属性
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        # 是否训练
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        # 词向量
        self.word_embeddings = tf.get_variable(
            "word_embeddings", initializer=tf.constant(word_embeddings, dtype=tf.float32), trainable=False)
        # 字向量
        self.char_embeddings = tf.get_variable(
            "char_embeddings", initializer=tf.constant(char_embeddings, dtype=tf.float32))
        # 字向量维度
        self.char_embed_size = self.config.get_default_params().char_embed_size
        # 词向量维度
        self.embed_size = self.config.get_default_params().embed_size
        # 字长度
        # self.char_len = self.config.get_default_params().

        self.batch_size = self.config.get_default_params().batch_size
        self.char_hidden = self.config.get_default_params().char_hidden
        self.hidden_size = self.config.get_default_params().hidden

        # size limit
        self.max_p_num = self.config.get_default_params().max_p_num
        self.max_p_len = self.config.get_default_params().max_p_len
        self.max_q_len = self.config.get_default_params().max_q_len
        self.max_a_len = self.config.get_default_params().max_a_len

        # gru单元，是否使用cundnn
        self.gru = cudnn_gru if self.config.get_default_params().use_cudnn else native_gru
        # keep_prob
        self.keep_prob = self.config.get_default_params().keep_prob

        # 保存模型
        self.saver = tf.train.Saver()

    '''
    定义placeholder
    '''
    def _set_placeholders(self):
        # passage的索引
        self.p = tf.placeholder(tf.int32, [None, None])
        # question索引
        self.q = tf.placeholder(tf.int32, [None, None])
        # passage的长度
        self.p_length = tf.placeholder(tf.int32, [None])
        # question的长度
        self.q_length = tf.placeholder(tf.int32, [None])
        # 开始
        self.start_label = tf.placeholder(tf.int32, [None])
        # 结束
        self.end_label = tf.placeholder(tf.int32, [None])

    '''
    下面是模型的每一层，里面包含了实际的模型
    '''

    '''
    embedding层
    '''
    def embedding(self):
        # N batch_size;PL c_maxlen;CL char_limit;dc char_dim
        with tf.variable_scope("emb"):
            # 字符向量
            with tf.variable_scope("char"):
                p_char_emb = tf.nn.embedding_lookup(self.char_embeddings, self.p)
                q_char_emb = tf.nn.embedding_lookup(self.char_embeddings, self.q)
                p_char_emb = dropout(
                    p_char_emb, keep_prob=self.keep_prob, is_train=self.is_train)
                q_char_emb = dropout(
                    q_char_emb, keep_prob=self.keep_prob, is_train=self.is_train)
                # 门控递归单元
                # 前向
                cell_fw = tf.contrib.rnn.GRUCell(self.char_hidden)
                # 后向
                cell_bw = tf.contrib.rnn.GRUCell(self.char_hidden)

                # 双向递归网络的动态版本
                # inputs必须是 [batch_size, max_time, ...]
                # 返回值output_fw [batch_size,max_time,cell_fw.output_size]
                # output_bw [batch_size,max_time,cell_bw.output_size]
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, p_char_emb, self.max_p_len, dtype=tf.float32)
                p_char_emb = tf.concat([state_fw, state_bw], axis=1)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, q_char_emb, self.max_q_len, dtype=tf.float32)
                q_char_emb = tf.concat([state_fw, state_bw], axis=1)
                q_char_emb = tf.reshape(q_char_emb, [self.batch_size, self.q_length, 2 * self.char_hidden])
                p_char_emb = tf.reshape(p_char_emb, [self.batch_size, self.p_length, 2 * self.char_hidden])

            # 词向量
            with tf.name_scope("word"):
                p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
                q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)

            p_embeddings = tf.concat([p_emb, p_char_emb], axis=2)
            q_embeddings = tf.concat([q_emb, q_char_emb], axis=2)

        return p_embeddings, q_embeddings

    '''
    encoding层
    '''
    def encode(self, passage_emb, question_emb):

        rnn = self.gru(num_layers=3,
                       num_units=self.hidden_size,
                       batch_size=self.batch_size,
                       input_size=passage_emb.get_shape().as_list()[-1],
                       keep_prob=self.keep_prob, is_train=self.is_train)
        pass_encoding = rnn(passage_emb, seq_len=self.p_length)
        ques_encoding = rnn(question_emb, seq_len=self.q_length)

    '''
    attention层
    '''
    def attention(self):
        with tf.variable_scope("attention"):
            qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d,
                                   keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            att = rnn(qc_att, seq_len=self.c_len)






