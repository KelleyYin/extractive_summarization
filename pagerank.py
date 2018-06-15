# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 0011 20:41
# @Author  : yzhao
import random
import numpy as np

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import rouge
from resource import load_evaluations_cut
from text_utils import split_sentence, write2file

random.seed(42)


def mrw_transition_probability(vec_arr):
    n = vec_arr.shape[0]
    m = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = cosine_similarity(vec_arr[i].reshape(1, -1),
                                        vec_arr[j].reshape(1, -1))[0][0]
            m[j, i] = m[i, j]
    for i in range(n):
        if not np.count_nonzero(m[i]):
            m[i] = np.full((1, n), 1 / n)
    return preprocessing.normalize(m, norm='l1')


def mrw_score(m, mu=0.85, epsilon=0.0001):
    score = np.full((m.shape[0], 1), 1)
    for _ in range(500):
        temp = score.copy()
        score = mu * np.mat(m).T * score + (1 - mu) / m.shape[0]
        if max(abs(temp - score)) < epsilon:
            break
    return score


def rand_summary(sentences, n=20):
    summary = []
    n = n if n < len(sentences) else len(sentences)
    # take n sentences randomly
    for index in random.sample(range(len(sentences)), n):
        summary.append(sentences[index])
    return summary


def first_summary(sentences, n=20):
    summary = []
    n = n if n < len(sentences) else len(sentences)
    # take the first n sentences in sequence
    for index in range(n):
        summary.append(sentences[index])
    return summary


def mrw_summary(sentences, n=20):
    summary = []
    n = n if n < len(sentences) else len(sentences)
    # take the top n sentences in scores
    vectorizer = CountVectorizer(binary=True)
    x = vectorizer.fit_transform(sentences)
    m = mrw_transition_probability(x.toarray())
    scores = mrw_score(m).A
    for i in np.argsort(-scores, axis=0).flatten()[:n]:
        summary.append(sentences[i])
    return summary


if __name__ == '__main__':
    summarizations = []
    references = []
    for e in load_evaluations_cut():
        sentences = split_sentence(e['article'])
        summary = rand_summary(sentences)
        summarizations.append(summary)
        references.append(e['summarization'])
        print(summary)

    # write2file(summarizations, references)
    rouge.rouge_score(summarizations, references)
