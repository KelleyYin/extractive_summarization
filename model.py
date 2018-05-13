# -*- coding: utf-8 -*-
# @Time    : 2018/5/10 9:38
# @Author  : yzhao
from keras import Input
from keras.optimizers import Adam
import numpy as np
from keras.layers import Embedding, Bidirectional, Conv1D, MaxPooling1D, GlobalAveragePooling1D, concatenate
from keras.layers.core import Dense, Dropout
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence


def format_k(docs, vocab, max_len):
    """
    转换成numpy格式的index数组
    :param docs:
    :param vocab:
    :param max_len: 每篇文档的最大长度
    :return:
    """
    X, y = [], []
    for doc in docs:
        x = []  # 存放每个document中的词在词汇表中的索引
        for w in doc.features:
            if w in vocab:
                x.append(vocab[w])
        X.append(x)
        y.append(doc.label)

    X = sequence.pad_sequences(X, maxlen=max_len)
    y = np.array(y)
    return X, y


def lstm(X_train, y_train, opt):
    model = Sequential()
    model.add(Embedding(opt.vocab_size, opt.embedding_size, input_length=opt.input_length))
    model.add(Dropout(opt.dropout_p))
    model.add(Conv1D(opt.filters, opt.kernel_size, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=opt.pool_size))
    model.add(LSTM(opt.hidden_size, dropout=opt.dropout_p, recurrent_dropout=opt.dropout_p))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=opt.learning_rate))
    model.fit(X_train, y_train, batch_size=opt.batch_size, epochs=opt.epochs)
    return model


def lstm_concat(X, X_c, y, opt):
    input_w = Input(shape=(opt.input_length,))
    x11 = Embedding(input_dim=opt.vocab_size, output_dim=opt.embedding_size)(input_w)
    x12 = Dropout(opt.dropout_p)(x11)
    x13 = Conv1D(opt.filters, opt.kernel_size, padding='valid', activation='relu', strides=1)(x12)
    x14 = MaxPooling1D(pool_size=opt.pool_size)(x13)
    lstm_w = LSTM(opt.hidden_size, dropout=opt.dropout_p, recurrent_dropout=opt.dropout_p)(x14)

    input_c = Input(shape=(opt.input_length,))
    x21 = Embedding(input_dim=opt.vocab_size, output_dim=opt.embedding_size)(input_c)
    x22 = Dropout(opt.dropout_p)(x21)
    x23 = Conv1D(opt.filters, opt.kernel_size, padding='valid', activation='relu', strides=1)(x22)
    x24 = MaxPooling1D(pool_size=opt.pool_size)(x23)
    lstm_c = LSTM(opt.hidden_size, dropout=opt.dropout_p, recurrent_dropout=opt.dropout_p)(x24)

    lstm = concatenate([lstm_w, lstm_c])
    outputs = Dense(1, activation='sigmoid')(lstm)
    model = Model([input_w, input_c], outputs)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=opt.learning_rate))
    model.fit([X, X_c], y, batch_size=opt.batch_size, epochs=opt.epochs)
    return model
