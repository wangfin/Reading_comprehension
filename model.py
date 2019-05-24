#!/usr/bin/env python
# @Time    : 2019/3/26 8:52
# @Author  : wb
# @File    : model.py

# 模型文件
import json
import logging
import os
import time
import tensorflow as tf
from tqdm import tqdm

from config import Config
from layers import cudnn_gru, native_gru, dropout, dot_attention, summ, ptr_net
from utils.dureader_eval import compute_bleu_rouge, normalize

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

    def __init__(self, vocab, trainable=True):

        # logger
        self.logger = logging.getLogger("brc")
        # vocabulary
        self.vocab = vocab
        self.trainable = trainable
        # 使用的优化函数
        self.optim_type = 'adam'

        # batch的size
        self.batch_size = self.config.get_default_params().batch_size
        # 隐藏单元
        self.char_hidden = self.config.get_default_params().char_hidden
        self.hidden_size = self.config.get_default_params().hidden_size
        self.attn_size = self.config.get_default_params().attn_size

        # size limit
        self.max_p_num = self.config.get_default_params().max_p_num
        self.max_p_len = self.config.get_default_params().max_p_len
        self.max_q_len = self.config.get_default_params().max_q_len
        self.max_a_len = self.config.get_default_params().max_a_len
        self.max_ch_len = self.config.get_default_params().max_ch_len

        # gru单元，是否使用cundnn
        self.gru = cudnn_gru if self.config.get_default_params().use_cudnn else native_gru
        # keep_prob
        self.keep_prob = self.config.get_default_params().keep_prob
        # ptr_keep_prob
        self.ptr_keep_prob = self.config.get_default_params().ptr_keep_prob

        # session info
        sess_config = tf.ConfigProto()
        # 程序按需申请内存
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        # 构建计算图
        self._build_graph()

        # 保存模型
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    '''
    定义placeholder
    '''
    def _set_placeholders(self):
        # 训练时创造的参数
        # if self.trainable:
        #     # passage的索引
        #     self.p = tf.placeholder(tf.int32, [self.batch_size*self.max_p_num, self.max_p_len], "passage")
        #     # question索引
        #     self.q = tf.placeholder(tf.int32, [self.batch_size*self.max_p_num, self.max_q_len], "question")
        #
        #     self.ph = tf.placeholder(tf.int32, [self.batch_size*self.max_p_num, self.max_p_len, self.max_ch_len],
        #                              "passage_char")
        #     self.qh = tf.placeholder(tf.int32, [self.batch_size*self.max_p_num, self.max_q_len, self.max_ch_len],
        #                              "question_char")
        #     # 开始
        #     self.start_label = tf.placeholder(tf.int32, [self.batch_size], "start_label")
        #     # 结束
        #     self.end_label = tf.placeholder(tf.int32, [self.batch_size], "end_label")
        #
        # # 不训练
        # else:
        #     # passage的索引
        #     self.p = tf.placeholder(tf.int32, [None, self.max_p_len], "passage")
        #     # question索引
        #     self.q = tf.placeholder(tf.int32, [None, self.max_q_len], "question")
        #
        #     self.ph = tf.placeholder(tf.int32, [None, self.max_p_len, self.max_ch_len], "passage_char")
        #     self.qh = tf.placeholder(tf.int32, [None, self.max_q_len, self.max_ch_len], "question_char")
        #
        #     # 开始
        #     self.start_label = tf.placeholder(tf.int32, [None], "start_label")
        #     # 结束
        #     self.end_label = tf.placeholder(tf.int32, [None], "end_label")

        # passage内容
        self.p = tf.placeholder(tf.int32, [None, self.max_p_len], "passage")
        # question内容
        self.q = tf.placeholder(tf.int32, [None, self.max_q_len], "question")

        # 字符内容
        self.ph = tf.placeholder(tf.int32, [None, self.max_p_len, self.max_ch_len], "passage_char")
        self.qh = tf.placeholder(tf.int32, [None, self.max_q_len, self.max_ch_len], "question_char")

        # 开始
        self.start_label = tf.placeholder(tf.int32, [None], "start_label")
        # 结束
        self.end_label = tf.placeholder(tf.int32, [None], "end_label")

        # mask矩阵是用于将一个batch中，长短不一的句子都能补齐到同一个长度，其中补上的数据为0，这样的话补齐的0就不会参与到后续的计算中
        self.p_mask = tf.cast(self.p, tf.bool)  # index 0 is padding symbol  N x self.max_p_num, max_p_len
        self.q_mask = tf.cast(self.q, tf.bool)
        # 对mask矩阵进行求和，得到passage和question的长度
        self.p_len = tf.reduce_sum(tf.cast(self.p_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
        # 是否训练
        self.is_train = self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=False)
        # 全局的训练步数
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        # print(self.p_mask.get_shape().as_list())
        # print(self.start_label.get_shape().as_list())

    '''
    下面是模型的每一层，里面包含了实际的模型
    '''
    '''
    构建计算图
    '''
    def _build_graph(self):
        start_t = time.time()
        self._set_placeholders()
        self._embed()
        self._encode()
        self._self_match()
        self._predict()
        self._compute_loss()
        self._create_train_op()

        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
    '''
    embedding层
    '''
    def _embed(self):
        with tf.variable_scope("emb"):
            # 字向量
            self.pretrained_char_mat = tf.get_variable(
                "char_embeddings",
                [self.vocab.get_char_size() - 2, self.vocab.char_embed_size],
                dtype=tf.float32,
                initializer=tf.constant_initializer(self.vocab.char_embeddings[2:], dtype=tf.float32),
                trainable=False)
            # 字向量的pad
            self.char_pad_unk_mat = tf.get_variable(
                "char_unk_pad",
                [2, self.pretrained_char_mat.get_shape().as_list()[1]],
                dtype=tf.float32,
                initializer=tf.constant_initializer(self.vocab.char_embeddings[:2], dtype=tf.float32),
                trainable=True)

            # 词向量
            self.pretrained_word_mat = tf.get_variable(
                "word_embeddings",
                [self.vocab.get_vocab_size() - 2, self.vocab.word_embed_size],
                initializer=tf.constant_initializer(self.vocab.word_embeddings[2:], dtype=tf.float32),
                trainable=False)
            # 词向量的pad
            self.word_pad_unk_mat = tf.get_variable(
                "word_unk_pad",
                [2, self.pretrained_word_mat.get_shape().as_list()[1]],
                dtype=tf.float32,
                initializer=tf.constant_initializer(self.vocab.word_embeddings[:2], dtype=tf.float32),
                trainable=True)

            self.char_embeddings = tf.concat([self.char_pad_unk_mat, self.pretrained_char_mat], axis=0)
            self.word_embeddings = tf.concat([self.word_pad_unk_mat, self.pretrained_word_mat], axis=0)

            # 字符长度，压缩了最后一层的字符emb_size
            self.ph_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.ph, tf.bool), tf.int32), axis=2), [-1])
            self.qh_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

            # print(self.word_embeddings.get_shape().as_list())

            # 字符向量
            with tf.variable_scope("char"):
                # embedding_lookup选取ph矩阵中的id对应在char_embeddings中的值
                p_char_emb = tf.reshape(tf.nn.embedding_lookup(self.char_embeddings, self.ph),
                    [self.batch_size*self.max_p_len*self.max_p_num, self.max_ch_len, self.vocab.char_embed_size])
                q_char_emb = tf.reshape(tf.nn.embedding_lookup(self.char_embeddings, self.qh),
                    [self.batch_size*self.max_q_len*self.max_p_num, self.max_ch_len, self.vocab.char_embed_size])

                # p_char_emb = tf.nn.embedding_lookup(self.char_embeddings, self.ph)
                # q_char_emb = tf.nn.embedding_lookup(self.char_embeddings, self.qh)

                p_char_emb = dropout(
                    p_char_emb, keep_prob=self.keep_prob, is_train=self.is_train)
                q_char_emb = dropout(
                    q_char_emb, keep_prob=self.keep_prob, is_train=self.is_train)

                # 门控递归单元
                # 前向
                cell_fw = tf.contrib.rnn.GRUCell(self.char_hidden)
                # 后向
                cell_bw = tf.contrib.rnn.GRUCell(self.char_hidden)
                '''
                    双向递归网络的动态版本
                    inputs必须是 [batch_size, max_time, ...]
                    返回值output_fw [batch_size,max_time,cell_fw.output_size]
                    output_bw [batch_size,max_time,cell_bw.output_size]
                '''
                # char-level向量是先输入预训练向量，然后输入双向RNN中
                # input的shape为[batch_size, max_len, depth]
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, p_char_emb, self.ph_len, dtype=tf.float32)
                p_char_emb = tf.concat([state_fw, state_bw], axis=1)

                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, q_char_emb, self.qh_len, dtype=tf.float32)
                q_char_emb = tf.concat([state_fw, state_bw], axis=1)

                # print(p_char_emb.get_shape().as_list())
                # print(q_char_emb.get_shape().as_list())

                p_char_emb = tf.reshape(p_char_emb,
                                        [self.batch_size*self.max_p_num, self.max_p_len, 2 * self.char_hidden])
                q_char_emb = tf.reshape(q_char_emb,
                                        [self.batch_size*self.max_p_num, self.max_q_len, 2 * self.char_hidden])

            # 词向量
            with tf.name_scope("word"):
                p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
                q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)

            # 最后得到passage和question的embeddings
            # 是由word-level 和 character-level组合成的
            self.p_embeddings = tf.concat([p_emb, p_char_emb], axis=2)
            self.q_embeddings = tf.concat([q_emb, q_char_emb], axis=2)

            '''
                200 + 300
                p_embeddings [320, 400, 500]
                q_embeddings [320, 60, 500]
            '''

    '''
    encoding层
    '''
    def _encode(self):

        # 3层的GRU单元
        rnn = self.gru(num_layers=3,
                       num_units=self.hidden_size,
                       batch_size=self.batch_size*self.max_p_num,
                       input_size=self.p_embeddings.get_shape().as_list()[-1],
                       keep_prob=self.keep_prob, is_train=self.is_train, scope='encode_rnn')

        self.pass_encoding = rnn(self.p_embeddings, seq_len=self.p_len)
        self.ques_encoding = rnn(self.q_embeddings, seq_len=self.q_len)

        '''
            pass_encoding [320, 400, 450]
            ques_encoding [320, 60, 450]
        '''

    '''
    自我的self_match匹配
    '''
    def _self_match(self):

        # 先计算attention
        with tf.variable_scope("gate_attention"):
            # question对于passage的attention
            ques_pass_att = dot_attention(self.pass_encoding, self.ques_encoding, mask=self.q_mask,
                                          hidden=self.attn_size, keep_prob=self.keep_prob, is_train=self.is_train)
            rnn = self.gru(num_layers=1, num_units=self.hidden_size, batch_size=self.batch_size*self.max_p_num,
                           input_size=ques_pass_att.get_shape().as_list()[-1],
                           keep_prob=self.keep_prob, is_train=self.is_train, scope='gate_attention_rnn')
            att = rnn(ques_pass_att, seq_len=self.p_len)
            '''
                att [320, 400, 150]
            '''

        # 进行match匹配
        with tf.variable_scope("self_match"):
            # 计算self_attention，在上一步的rnn得出的编码，在这里进一步计算self-attention
            self_att = dot_attention(
                att, att, mask=self.p_mask, hidden=self.attn_size, keep_prob=self.keep_prob, is_train=self.is_train)
            rnn = self.gru(num_layers=1, num_units=self.hidden_size, batch_size=self.batch_size*self.max_p_num,
                           input_size=self_att.get_shape().as_list()[-1],
                           keep_prob=self.keep_prob, is_train=self.is_train, scope='self_match_rnn')
            self.match = rnn(self_att, seq_len=self.q_len)
            '''
                match [320, 400, 150]
            '''

    '''
    预测函数，进行最终的预测
    '''
    def _predict(self):

        # pointer 指针网络
        # 指针网络就是softmax网络的特例
        with tf.variable_scope("pointer"):
            # 这里的init表示了question的attention-pooling
            init = summ(self.ques_encoding[:, :, -2*self.hidden_size:], self.hidden_size, mask=self.q_mask,
                        keep_prob=self.ptr_keep_prob, is_train=self.is_train)
            pointer = ptr_net(batch=self.batch_size*self.max_p_num,
                              hidden=init.get_shape().as_list()[-1],
                              keep_prob=self.ptr_keep_prob,
                              is_train=self.is_train)
            self.logits1, self.logits2 = pointer(init, self.match, self.hidden_size, self.p_mask)

            self.start_logits = tf.reshape(self.logits1, [self.batch_size, -1])
            self.end_logits = tf.reshape(self.logits2, [self.batch_size, -1])

        # 进行预测
        with tf.variable_scope("predict"):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(self.logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(self.logits2), axis=1))
            # 复制一个张量,将每个最内层矩阵中的所有中心区域外的所有内容设置为零.
            outer = tf.matrix_band_part(outer, 0, 15)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

    '''
    计算损失函数
    '''
    def _compute_loss(self):
        def sparse_nll_loss(probs, labels, scope=None):
            # negative log likelyhood loss
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=probs,
                    labels=tf.stop_gradient(labels)))
                # losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)

                # 当数据中，有的是数据项中不含有答案，可以使用零向量来表示缺失，并将损失函数改为
                # loss = tf.reduce_sum（label * - tf.log（tf.nn.softmax（logits）+  1e-6））
            return losses

        start_loss = sparse_nll_loss(probs=self.start_logits, labels=self.start_label)
        end_loss = sparse_nll_loss(probs=self.end_logits, labels=self.end_label)
        # self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(start_loss, end_loss))
        # if self.weight_decay > 0:
        #     with tf.variable_scope('l2_loss'):
        #         l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
        #     self.loss += self.weight_decay * l2_loss

    '''
    找到每个位置给定start_prob和end_prob的样本的最佳答案。这将调用find_best_answer_for_passage，因为示例中有多个段落
    '''
    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len):
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer

    '''
    使用单个段落中的最大start_prob * end_prob查找最佳答案
    '''
    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    '''
    优化函数
    '''
    def _create_train_op(self):
        if self.trainable:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, self.config.get_default_params().grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    #     opt_arg = self.config.get_default_params().opt_arg
    #
    #     if self.optim_type == 'adagrad':
    #         self.optimizer = tf.train.AdagradOptimizer(
    #             learning_rate=opt_arg['adagrad']['learning_rate'])
    #     elif self.optim_type == 'adam':
    #         self.optimizer = tf.train.AdamOptimizer(
    #             learning_rate=opt_arg['adam']['learning_rate'],
    #             beta1=opt_arg['adam']['beta1'],
    #             beta2=opt_arg['adam']['beta2'],
    #             epsilon=opt_arg['adam']['epsilon'])
    #     elif self.optim_type == 'adadelta':
    #         self.optimizer = tf.train.AdadeltaOptimizer(
    #             learning_rate=opt_arg['adadelta']['learning_rate'],
    #             rho=opt_arg['adadelta']['rho'],
    #             epsilon=opt_arg['adadelta']['epsilon'])
    #     elif self.optim_type == 'gd':
    #         self.optimizer = tf.train.GradientDescentOptimizer(
    #             learning_rate=opt_arg['gradientdescent']['learning_rate'])
    #     else:
    #         raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
    #
    #     self.logger.info("applying optimize %s" % self.optim_type)
    #     # 返回使用trainable = True创建的所有变量
    #     if self.clip_weight:
    #         # 削减梯度
    #         tvars = tf.trainable_variables()
    #         grads = tf.gradients(self.loss, tvars)
    #         grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.config.get_default_params().grad_clip)
    #         grad_var_pairs = zip(grads, tvars)
    #         # 最小化loss
    #         self.train_op = self.optimizer.apply_gradients(grad_var_pairs, global_step=self.global_step, name='apply_grad')
    #     else:
    #         self.train_op = self.optimizer.minimize(self.loss)

    '''
    训练每个epoch
    '''
    def _train_epoch(self, train_batches):
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 100, 0

        for bitx, batch in enumerate(train_batches, 1):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.qh: batch['question_char_ids'],
                         self.ph: batch['passage_char_ids'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         }
            try:
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
                total_loss += loss * len(batch['raw_data'])
                total_num += len(batch['raw_data'])
                n_batch_loss += loss
            except Exception as e:
                continue

            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        print("total_num", total_num)
        return 1.0 * total_loss / total_num

    '''
    训练函数
    '''
    def train(self, data, epochs, batch_size, save_dir, save_prefix, evaluate=True):
        pad_id = self.vocab.get_id_byword(self.vocab.pad_token)
        pad_char_id = self.vocab.get_id_bychar(self.vocab.pad_token)
        max_rouge_l = 0
        # 保存summary
        writer = tf.summary.FileWriter(self.config.get_filepath().summary_dir, self.sess.graph)

        lr = self.config.get_default_params().init_lr
        self.sess.run(tf.assign(self.is_train, tf.constant(True, dtype=tf.bool)))
        self.sess.run(tf.assign(self.lr, tf.constant(lr, dtype=tf.float32)))

        for epoch in tqdm(range(1, epochs + 1)):
            global_step = self.sess.run(self.global_step) + 1
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.next_batch('train', batch_size, pad_id, pad_char_id, shuffle=True)
            train_loss = self._train_epoch(train_batches)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))
            # 保存到tensorboard
            if global_step % self.config.get_default_params().period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=train_loss), ])
                writer.add_summary(loss_sum, global_step)

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                self.sess.run(tf.assign(self.is_train, tf.constant(False, dtype=tf.bool)))
                if data.dev_set is not None:
                    eval_batches = data.next_batch('dev', batch_size, pad_id, pad_char_id, shuffle=False)
                    eval_loss, bleu_rouge, summ = self.evaluate(eval_batches, data_type='dev')
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    for s in summ:
                        writer.add_summary(s, global_step)

                    if bleu_rouge['Rouge-L'] > max_rouge_l:
                        self.save(save_dir, save_prefix)
                        max_rouge_l = bleu_rouge['Rouge-L']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

            self.sess.run(tf.assign(self.is_train, tf.constant(True, dtype=tf.bool)))

    '''
    评价函数
    '''
    def evaluate(self, eval_batches, data_type, result_dir=None, result_prefix=None, save_full_info=False):
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):

            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.qh: batch['question_char_ids'],
                         self.ph: batch["passage_char_ids"],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         }
            try:
                start_probs, end_probs, loss = self.sess.run([self.logits1, self.logits2, self.loss], feed_dict)
                total_loss += loss * len(batch['raw_data'])
                total_num += len(batch['raw_data'])

                padded_p_len = len(batch['passage_token_ids'][0])
                for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):

                    best_answer = self.find_best_answer(sample, start_prob, end_prob, padded_p_len)
                    if save_full_info:
                        sample['pred_answers'] = [best_answer]
                        pred_answers.append(sample)
                    else:
                        pred_answers.append({'question_id': sample['question_id'],
                                             'question_type': sample['question_type'],
                                             'answers': [best_answer],
                                             'entity_answers': [[]],
                                             'yesno_answers': []})
                    if 'answers' in sample:
                        ref_answers.append({'question_id': sample['question_id'],
                                            'question_type': sample['question_type'],
                                            'answers': sample['answers'],
                                            'entity_answers': [[]],
                                            'yesno_answers': []})
            except:
                print('evaluate 异常')
                continue

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # 这个平均损失在测试集上是无效的，因为我们没有真正的start_id和end_id
        ave_loss = 1.0 * total_loss / total_num
        # 如果提供了参考答案，则计算bleu和rouge分数
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        # 存储
        ave_loss_sum = tf.Summary(value=[tf.Summary.Value(
            tag="{}/loss".format(data_type), simple_value=ave_loss), ])
        bleu_4_sum = tf.Summary(value=[tf.Summary.Value(
            tag="{}/bleu_4".format(data_type), simple_value=bleu_rouge['Bleu-4']), ])
        rougeL_sum = tf.Summary(value=[tf.Summary.Value(
            tag="{}/rouge-L".format(data_type), simple_value=bleu_rouge['Rouge-L']), ])
        return ave_loss, bleu_rouge, [ave_loss_sum, bleu_4_sum, rougeL_sum]

    '''
    存储模型
    '''
    def save(self, model_dir, model_prefix):

        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    '''
    将模型从model_prefix恢复为model_dir作为模型指示符
    '''
    def restore(self, model_dir, model_prefix):

        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))


