# -*- coding: utf-8 -*-
# @Time    : 2018/4/3 20:40
# @Author  : yzhao
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import numpy as np

from document import create_documents, build_vocabulary, create_document
from model import format_k, lstm
from resource import load_trains_cut, load_evaluations_cut, load_tests_cut, load_trains_c, load_evaluations_c
from rouge import rouge_score
from text_utils import generate_result

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


class Config(object):
    vocab_size = 10000  # for word level
    embedding_size = 200
    hidden_size = 200
    batch_size = 500
    epochs = 10
    input_length = 200  # each sentence length
    learning_rate = 0.0001
    dropout_p = 0.2

    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4


if __name__ == '__main__':
    opt = Config()
    pos, neg = create_documents(load_trains_cut(), balance_length=50000)
    print('trains length:', len(pos), len(neg))
    docs_train = pos[:50000] + neg[:50000]  # 选取摘要和非摘要各取n条训练
    w2ix, ix2w = build_vocabulary(docs_train, opt.vocab_size)
    print('vocab size:', len(w2ix))
    X, y = format_k(docs_train, w2ix, opt.input_length)

    model = lstm(X, y, opt)

    summarizations = []
    references = []

    for i, e in enumerate(load_evaluations_cut()):
        _, neg = create_document(e)

        X, y = format_k(neg, w2ix, opt.input_length)
        X_pred = model.predict(X)

        summary = [neg[i].text for i in np.argsort(-X_pred.squeeze(1)[:20])]
        print('-------------------------')
        print(i, e['summarization'])
        print('-------------------------')
        print('\n'.join(summary))
        summarizations.append(summary)
        references.append(e['summarization'])

    # write2file(summarizations, references)
    rouge_score(summarizations, references)
    # generate_result(summarizations)
