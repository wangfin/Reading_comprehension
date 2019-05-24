#!/usr/bin/env python
# @Time    : 2019/4/13 10:16
# @Author  : wb
# @File    : layers.py

import tensorflow as tf
'''
function文件，里面主要是功能函数
用于model.py文件
'''

'''
cudnn的GRU
'''
class cudnn_gru:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        # 输入的层数是3层
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            # 第一层500，后两层 2*75
            input_size_ = input_size if layer == 0 else 2 * num_units
            # 模型层数1层，单元数75
            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            # tile 复制修改过的mask，在mask的第2维重复了batch_size遍
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    # 实现了call方法，可以像函数一样调用他 如 a=A() a()
    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        # 这一步是将数据的第二维和第一维对调，从维度上看就是第二维和第一维对换
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope("fw_{}".format(self.scope + str(layer))):
                # outputs[-1] * mask_fw shape [1, 320, 500]
                out_fw, _ = gru_fw(
                    outputs[-1] * mask_fw, initial_state=(init_fw, ))
            with tf.variable_scope("bw_{}".format(self.scope + str(layer))):
                # 反转可变长度切片
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = gru_bw(inputs_bw, initial_state=(init_bw, ))
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
            # print('outputs' + str(layer) + str(tf.shape(outputs)))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res

'''
不使用cudnn的GRU单元
'''
class native_gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="native_gru"):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.rnn.GRUCell(num_units)
            gru_bw = tf.contrib.rnn.GRUCell(num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, _ = tf.nn.dynamic_rnn(
                        gru_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    out_bw, _ = tf.nn.dynamic_rnn(
                        gru_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        return res

# pointer net
class ptr_net:
    def __init__(self, batch, hidden, keep_prob=1.0, is_train=None, scope="ptr_net"):
        self.gru = tf.contrib.rnn.GRUCell(hidden)
        self.batch = batch
        self.scope = scope
        self.keep_prob = keep_prob
        self.is_train = is_train
        self.dropout_mask = dropout(tf.ones(
            [batch, hidden], dtype=tf.float32), keep_prob=keep_prob, is_train=is_train)

    def __call__(self, init, match, d, mask):
        with tf.variable_scope(self.scope):
            # 输入的match，先进行Dropout
            d_match = dropout(match, keep_prob=self.keep_prob,
                              is_train=self.is_train)
            inp, logits1 = pointer(d_match, init * self.dropout_mask, d, mask)
            d_inp = dropout(inp, keep_prob=self.keep_prob,
                            is_train=self.is_train)
            # 从给定的init状态开始运行此GRU单元
            _, state = self.gru(d_inp, init)
            tf.get_variable_scope().reuse_variables()
            _, logits2 = pointer(d_match, state * self.dropout_mask, d, mask)
            return logits1, logits2

# 指针
def pointer(inputs, state, hidden, mask, scope="pointer"):
    with tf.variable_scope(scope):
        # expand_dims 在第一维插入一维
        u = tf.concat([tf.tile(tf.expand_dims(state, axis=1),
                               [1, tf.shape(inputs)[1], 1]), inputs], axis=2)
        s0 = tf.nn.tanh(dense(u, hidden, use_bias=False, scope="pointer_s0"))
        s = dense(s0, 1, use_bias=False, scope="pointer_s")
        # 计算概率
        # squeeze删除1的维度，也就是删除了s中的第三维
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        # 补全第三维
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        # 按第一维求和，a*inputs的shape为[320, 400, 150]
        res = tf.reduce_sum(a * inputs, axis=1)
        return res, s1

# 求和，这个应该是attention-pooling
def summ(memory, hidden, mask, keep_prob=1.0, is_train=None, scope="summ"):
    with tf.variable_scope(scope):
        # drop层
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        s0 = tf.nn.tanh(dense(d_memory, hidden, scope="att_pool_s0"))
        s = dense(s0, 1, use_bias=False, scope="att_pool_s")
        # squeeze删除s里面 第0个到第3个里面的1
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        # 这里是论文中的rQ
        res = tf.reduce_sum(a * memory, axis=1)
        return res

# dropout层
def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        # 噪声
        # noise是一个一维的向量
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        # 如果mode是embedding层
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        # 输入是3维的
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        # 如果is_trian为True，就运行第一个函数，反则就是第二个
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args

# 对一个batch中填充的0（pad）进行处理，在计算loss是需要将填充部分置0
# 在attention中，是把填充部分减去一个大整数，保证接近于0
def softmax_mask(val, mask):
    INF = 1e30
    return -INF * (1 - tf.cast(mask, tf.float32)) + val

'''
在计算attention的过程中，需要进行mask操作。有两种需要进行mask的情况：

1.所有计算权重向量的地方都要对补全（padding）的数据进行屏蔽
2.decoder中的self-attention层在计算词语的权向量时还要屏蔽掉该词语的后续词语

对于第一种情况，在nlp中对文本进行padding是一项基本的预处理操作，为的是使序列的长度相同，从而可以进行矩阵运算。
这些补全的数据对于其他词语的信息获取是无用的，因此在attention机制中计算词语的权重向量时，应该将所有padding数据对应的权重置为0。

对于第二种情况，在机器翻译任务中，我们在获取输出序列中某一个词语的向量表达时，通常假设该词语的后续词语是未知的，即仅根据已有的信息计算词向量。
因此在采用self-attention机制计算词语的权重向量时，我们应将目标词语的后续词语所对应的权重置为0。
'''

# 点注意力
def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    with tf.variable_scope(scope):
        # 进行dropout
        # 根据drop，说明inputs也就是3维输入
        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        # 第2维
        JX = tf.shape(inputs)[1]

        # 计算attention attention(K,Q,V)
        with tf.variable_scope("attention"):
            # 嵌套 relu
            inputs_ = tf.nn.relu(
                dense(d_inputs, hidden, use_bias=False, scope="inputs"))
            memory_ = tf.nn.relu(
                dense(d_memory, hidden, use_bias=False, scope="memory"))
            # 对memory_进行转置，与inputs_相乘，除以根号hidden
            outputs = tf.matmul(inputs_, tf.transpose(
                memory_, [0, 2, 1])) / (hidden ** 0.5)
            # expand_dims 现在mask的第2个位置，插入一个1
            # tile 复制修改过的mask，在mask的第2维重复了JX遍
            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
            # 对outputs做softmax，也就是进行计算attention的最后一步
            # softmax = tf.exp(outputs) / tf.reduce_sum(tf.exp(outputs), mask)
            # 将在mask维度，对outputs进行softmax
            logits = tf.nn.softmax(softmax_mask(outputs, mask))
            # 这里的outputs也就是attention了
            outputs = tf.matmul(logits, memory)
            # 连接inputs，outputs，在第三维
            # 这里就是论文中的[up,ct]
            res = tf.concat([inputs, outputs], axis=2)

        # 门控单元，用于进一步的把控输入
        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            # y = 1 / (1 + exp(-x))
            gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
            return res * gate

# 全连接层
def dense(inputs, hidden, use_bias=True, scope="dense"):
    with tf.variable_scope(scope):
        # 获取输入的shape
        shape = tf.shape(inputs)
        # as_list 返回每个维度的整数列表
        # 获取最后一个维度的长度，也就是dim
        dim = inputs.get_shape().as_list()[-1]
        # 获取输入的-1维 加上 hidden
        # hidden是一维的数字，表示了隐藏单元的数量
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        # 这个inputs把原来的input前两维组合在一起
        flat_inputs = tf.reshape(inputs, [-1, dim])
        # 创建W
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        # res的shape是inputs的最后一维换成hidden
        return res

if __name__ == '__main__':
    # args1 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    args1 = [[1, 2, 3], [3, 4, 5]]
    args1 = tf.cast(args1, tf.float32)
    args2 = 75 #[[9, 10], [11, 12]]
    # args2 = [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
    args2 = tf.cast(args2, tf.int32)

    # shape = tf.shape(args2)
    # out_shape = [shape[idx] for idx in range(len(args2.get_shape().as_list()) - 1)]
    # dim = args1.get_shape().as_list()[-1]
    den = dense(args1, args2)#tf.reshape(args1, [-1, dim])#dense(args1, args2)
    # print(len(args.get_shape().as_list()))
    # for i in range(len(args.get_shape().as_list()) - 1):
    #     print(i)
    # is_training = tf.cast(True, tf.bool)
    # drop = dropout(args, keep_prob=0.7, is_train=is_training, mode='embedding')
    sess = tf.Session()
    print(sess.run(den))