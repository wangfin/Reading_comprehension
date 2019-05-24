#!/usr/bin/env python
# @Time    : 2019/4/17 9:13
# @Author  : wb
# @File    : run.py

import logging
import os
import pickle

from pre_process import Propress
from vocab import Vocab
from config import Config
from model import Model

'''
运行函数，用于运行整个模型
'''

class Run():
    config = Config()

    # 准备数据
    train_files = [config.get_filepath().zhidao_train_file]#, config.get_filepath().search_train_file]
    dev_files = [config.get_filepath().zhidao_dev_file]#, config.get_filepath().search_dev_file]
    test_files = [config.get_filepath().zhidao_test_file]#, config.get_filepath().search_test_file]

    algo = 'r-net'

    '''
    检查数据，创建目录，准备词汇表和嵌入
    '''
    def prepare(self):
        logger = logging.getLogger("brc")
        logger.info("====== preprocessing ======")
        logger.info('Checking the data files...')

        for data_path in self.train_files + self.dev_files + self.test_files:
            assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
        logger.info('Preparing the directories...')
        for dir_path in [self.config.get_filepath().vocab_dir,
                         self.config.get_filepath().model_dir,
                         self.config.get_filepath().output_dir,
                         self.config.get_filepath().summary_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        logger.info('Building vocabulary...')

        brc_data = Propress(self.config.get_default_params().max_p_num,
                            self.config.get_default_params().max_p_len,
                            self.config.get_default_params().max_q_len,
                            self.config.get_default_params().max_ch_len,
                            train_files=self.train_files, dev_files=self.dev_files, test_files=self.test_files)

        vocab = Vocab(lower=True)
        # 遍历 question_tokens 和 passage里面的passage_token
        for word in brc_data.word_iter('train'):
            vocab.add_word(word)
            [vocab.add_char(ch) for ch in word]

        unfiltered_vocab_size = vocab.get_vocab_size()
        vocab.filter_words_by_cnt(min_cnt=2)
        filtered_num = unfiltered_vocab_size - vocab.get_vocab_size()
        logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num, vocab.get_vocab_size()))

        unfiltered_vocab_char_size = vocab.get_char_size()
        vocab.filter_chars_by_cnt(min_cnt=2)
        filtered_char_num = unfiltered_vocab_char_size - vocab.get_char_size()
        logger.info('After filter {} chars, the final char vocab size is {}'.format(filtered_char_num,
                                                                                    vocab.get_char_size()))

        logger.info('Assigning embeddings...')
        pretrained_word_path = self.config.get_filepath().vector_file
        pretrained_char_path = self.config.get_filepath().char_vector_file

        # 读入预训练的词向量
        if pretrained_word_path is not None:
            vocab.load_pretrained_word_embeddings(pretrained_word_path)
        else:
            vocab.randomly_init_word_embeddings(self.config.get_default_params().word_embed_size)

        # 读入预训练的字向量
        if pretrained_char_path is not None:
            vocab.load_pretrained_char_embeddings(pretrained_char_path)
        else:
            vocab.randomly_init_char_embeddings(self.config.get_default_params().char_embed_size)

        logger.info('save word vocab size is {}'.format(vocab.get_vocab_size()))
        logger.info('save char vocab size is {}'.format(vocab.get_char_size()))

        logger.info('Saving vocab...')
        with open(os.path.join(self.config.get_filepath().vocab_dir, 'vocab.data'), 'ab') as fout:
            pickle.dump(vocab, fout)

        logger.info('====== Done with preparing! ======')

    '''
    训练RC模型
    '''
    def train(self):
        logger = logging.getLogger("brc")
        logger.info("====== training ======")
        logger.info('Load data_set and vocab...')
        with open(os.path.join(self.config.get_filepath().vocab_dir, 'vocab.data'), 'rb') as fin:
            vocab = pickle.load(fin)

        # print(vocab.get_char_size())

        brc_data = Propress(self.config.get_default_params().max_p_num,
                            self.config.get_default_params().max_p_len,
                            self.config.get_default_params().max_q_len,
                            self.config.get_default_params().max_ch_len,
                            train_files=self.dev_files, dev_files=self.dev_files)
        logger.info('Converting text into ids...')
        brc_data.convert_to_ids(vocab)
        logger.info('Initialize the model...')
        rc_model = Model(vocab, trainable=True)
        logger.info('Training the model...')
        rc_model.train(brc_data,
                       self.config.get_default_params().epoch,
                       self.config.get_default_params().batch_size,
                       save_dir=self.config.get_filepath().model_dir,
                       save_prefix=self.algo)
        logger.info('====== Done with model training! ======')

    '''
    评估测试数据
    '''
    def evaluate(self):
        logger = logging.getLogger("brc")
        logger.info("====== evaluating ======")
        logger.info('Load data_set and vocab...')
        with open(os.path.join(self.config.get_filepath().vocab_dir, 'vocab.data'), 'rb') as fin:
            vocab = pickle.load(fin)

        assert len(self.dev_files) > 0, 'No dev files are provided.'
        dataloader = Propress(self.config.get_default_params().max_p_num,
                              self.config.get_default_params().max_p_len,
                              self.config.get_default_params().max_q_len,
                              self.config.get_default_params().max_ch_len,
                              dev_files=self.dev_files)

        logger.info('Converting text into ids...')
        dataloader.convert_to_ids(vocab)

        logger.info('Restoring the model...')
        model = Model(vocab, trainable=False)
        model.restore(self.config.get_filepath().model_dir, self.algo)
        logger.info('Evaluating the model on dev set...')
        dev_batches = dataloader.next_batch('dev',
                                            self.config.get_default_params().batch_size,
                                            vocab.get_id_byword(vocab.pad_token),
                                            vocab.get_id_bychar(vocab.pad_token),
                                            shuffle=False)

        dev_loss, dev_bleu_rouge, summ = model.evaluate(
            dev_batches, 'dev', result_dir=self.config.get_filepath().output_dir, result_prefix='dev.predicted')

        logger.info('Loss on dev set: {}'.format(dev_loss))
        logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
        logger.info('Predicted answers are saved to {}'.format(os.path.join(self.config.get_filepath().output_dir)))

    '''
    预测结果
    '''
    def predict(self):
        logger = logging.getLogger("brc")

        logger.info('Load data_set and vocab...')
        with open(os.path.join(self.config.get_filepath().vocab_dir, 'vocab.data'), 'rb') as fin:
            vocab = pickle.load(fin)

        assert len(self.test_files) > 0, 'No test files are provided.'
        dataloader = Propress(self.config.get_default_params().max_p_num,
                              self.config.get_default_params().max_p_len,
                              self.config.get_default_params().max_q_len,
                              self.config.get_default_params().max_ch_len,
                              test_files=self.test_files)

        logger.info('Converting text into ids...')
        dataloader.convert_to_ids(vocab)
        logger.info('Restoring the model...')

        model = Model(vocab, trainable=False)
        model.restore(self.config.get_filepath().model_dir, self.algo)
        logger.info('Predicting answers for test set...')
        test_batches = dataloader.next_batch('test',
                                             self.config.get_default_params().batch_size,
                                             vocab.get_word_id(vocab.pad_token),
                                             vocab.get_char_id(vocab.pad_token),
                                             shuffle=False)

        model.evaluate(test_batches, 'test', result_dir=self.config.get_filepath().output_dir, result_prefix='test.predicted')

    def run(self):

        logger = logging.getLogger("brc")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(self.config.get_filepath().log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # self.prepare()
        # self.train()
        # self.evaluate()
        # self.predict()

if __name__ == '__main__':

    brc = Run()
    brc.run()
