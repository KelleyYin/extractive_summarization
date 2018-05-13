# -*- coding: utf-8 -*-
# @Time    : 2018/5/13 15:17
# @Author  : yzhao
from sklearn.preprocessing import normalize

import rouge
from resource import *
from text_utils import *
import numpy as np


def similarity_matrix(sentences):
    n = len(sentences)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = similarity(sentences[i], sentences[j])
            m[j, i] = m[i, j]
    return np.mat(m)


def hits_summary(text, n=10):
    sentences = split_sentence(text)
    m = similarity_matrix(sentences)
    a = np.ones((m.shape[0], 1))
    h = np.ones((m.shape[0], 1))
    for _ in range(10):
        a = m * h
        a = normalize(a, 'l1', axis=0)
        h = m * a
        h = normalize(h, 'l1', axis=0)
    summary = []
    for i in np.argsort(-a, axis=0).squeeze(1):
        if len(summary) == n:
            break
        if sentences[i] not in summary:
            summary.append(sentences[i])
    return summary


if __name__ == '__main__':
    summarizations = []
    references = []
    for e in load_evaluations_cut():
        summary = hits_summary(e['article'])

        print('------------------')
        print(e['summarization'])
        print('------------------')
        print('\n'.join(summary))

        summarizations.append(summary)
        references.append(e['summarization'])

    rouge.rouge_score(summarizations, references)

