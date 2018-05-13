# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 0011 23:59
# @Author  : yzhao

import os
import sys

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from scipy.sparse import lil_matrix
import numpy as np

from document import create_documents, build_vocabulary
from resource import load_trains_cut, load_evaluations_cut
from rouge import rouge_score
from text_utils import split_sentence, write2file


def build_vector(docs, vocab):
    """
    将document对象转换成one-hot的形式，由于词典比较大，采用稀疏矩阵存储
    :param docs: document列表
    :param vocab: 词汇表
    :return: X,y
    """
    m = len(docs)
    n = len(vocab)
    X = lil_matrix((m, n))  # 构建稀疏矩阵
    y = np.zeros((m,))
    for i in range(m):
        vec = [0] * n
        for feature, value in docs[i].features.items():
            if feature in vocab:
                vec[vocab[feature]] = 1  # 这里设置为1效果会好一点
        X[i] = vec
        y[i] = docs[i].label
    return X, y


if __name__ == '__main__':
    num_train = 10000
    vocab_size = 10000
    pos, neg = create_documents(load_trains_cut(), balance_length=num_train)
    print(len(pos), len(neg))
    doc_train = pos[:num_train] + neg[:num_train]  # 选取摘要和非摘要各取n条训练
    w2ix, ix2w = build_vocabulary(doc_train, vocab_size)

    transformer = TfidfTransformer()  # 使用tf-idf效果会好一点
    X, y = build_vector(doc_train, w2ix)
    X = transformer.fit_transform(X)

    clf = CalibratedClassifierCV(LinearSVC(random_state=0))  # 要让它输出概率[0.8,0.2]
    clf.fit(X, y)

    summarizations = []
    references = []
    for e in load_evaluations_cut():

        _, neg = create_documents([e])
        X, _ = build_vector(neg, w2ix)
        X = transformer.transform(X)
        proba = clf.predict_proba(X)
        sentences = split_sentence(e['article'])
        summary = []
        d = {i: s for i, s in enumerate([x[1] for x in proba])}
        for x in sorted(d.items(), key=lambda x: x[1], reverse=True)[:20]:
            summary.append(sentences[x[0]])

        summarizations.append(summary)
        print('\n'.join(summary))
        print('--------------------------')
        references.append(e['summarization'])

    # write2file(summarizations, references)
    rouge_score(summarizations, references)
