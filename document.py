# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 0011 23:59
# @Author  : yzhao
import numpy as np

from mmr import mmr_summary
from text_utils import split_sentence


class Document(object):
    def __init__(self, text, label, **kwargs):
        self.text = text
        self.label = label
        self.features = self._add_features(kwargs)

    def _add_features(self, kwargs):
        #  添加词特征
        feature_dict = {}
        for word in self.text.split():
            feature_dict[word] = feature_dict.get(word, 0) + 1
        #  添加其他特征
        for k, v in kwargs.items():
            feature_dict[str(k) + str(v)] = 1
        return feature_dict


def create_documents(dict_list, balance_length=5000):
    """
    创建文档对象，包含所有摘要和非摘要的列表
    :param dict_list:
    :param balance_length: 各取多少条
    :return:
    """
    pos, neg = [], []
    for d in dict_list:
        if len(pos) >= balance_length:
            break

        sentences = split_sentence(d['summarization'])
        for i, s in enumerate(sentences):
            pos.append(Document(s, True,
                                abs_pos=i,
                                # sen_len=int(np.round(len(s) * 9 / (len(max(s, key=len)))))
                                # rel_pos=np.round(i * 9 / len(sentences1))
                                ))  # i 代表文章中的位置
            if len(pos) >= balance_length:
                break

        sentences = split_sentence(d['article'])
        for i, s in enumerate(sentences):
            neg.append(Document(s, False,
                                abs_pos=i,
                                # sen_len=int(np.round(len(s) * 9 / (len(max(s, key=len)))))
                                # rel_pos=np.round(i * 9 / len(sentences1))
                                ))
            if len(neg) >= balance_length:
                break

    return pos, neg


def create_document(d, preprocess=None):
    pos, neg = [], []
    summary_sentences = split_sentence(d['summarization'])
    article_sentences = None
    if preprocess is None:
        article_sentences = split_sentence(d['article'])
    elif preprocess == 'mmr':
        summary = mmr_summary(d['article'], n=40)
        article_sentences = split_sentence(summary)

    for i, s in enumerate(summary_sentences):
        pos.append(Document(s, True, abs_pos=i))
    for i, s in enumerate(article_sentences):
        neg.append(Document(s, True, abs_pos=i))
    return pos, neg


def build_vocabulary(train, count=None):
    """
    根据document对象，取出频率最高的若干词
    :param train: document列表
    :param count: 频率最高的前N条
    :return:
    """
    words = {}
    for doc in train:
        for w, c in doc.features.items():
            words[w] = words.get(w, 0) + c
    if count is not None:
        words = dict(sorted(words.items(), key=lambda d: d[1], reverse=True)[:count])
    w2ix = {w: i for i, (w, c) in enumerate(words.items())}
    ix2w = {i: w for w, i in w2ix.items()}
    return w2ix, ix2w


if __name__ == '__main__':
    pass
