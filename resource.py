# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 0011 21:16
# @Author  : yzhao
import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load(file_path):
    with open(file_path, mode='rb') as f:
        file = pickle.load(f)
    return file


def save(path, file):
    with open(path, mode='wb') as f:
        pickle.dump(file, f)


def load_evaluations_cut():
    """
    加载一个字典列表
    :return:
    """
    return load(BASE_DIR + '/data/evaluations_cut.pkl')


def load_trains_cut():
    """
    加载一个字典列表，作为训练数据
    @格式：d{'summarization':'xxx', 'article':'xxx', 'index':'0'}
    :return:
    """
    return load(BASE_DIR + '/data/trains_cut.pkl')


def load_trains():
    """未分词的训练数据"""
    return load(BASE_DIR + '/data/trains.pkl')


def load_evaluations():
    return load(BASE_DIR + '/data/evaluations.pkl')


def load_trains_c():
    """未分词的训练数据"""
    return load(BASE_DIR + '/data/trains_c.pkl')


def load_evaluations_c():
    return load(BASE_DIR + '/data/evaluations_c.pkl')


def load_tests_cut():
    return load(BASE_DIR + '/data/tests_cut.pkl')


def load_stop_words():
    stop_words = []
    for line in open(BASE_DIR + '/data/stopwords.txt', 'rt', encoding='utf-8'):
        stop_words.append(line.strip())
    return stop_words


def load_char_dict(from_dict=True):
    if from_dict:
        char_dict = {}
        for line in open(BASE_DIR + '/data/char_dict.txt', 'rt', encoding='utf-8'):
            char = line.strip().split('\t')[0]
            if char not in char_dict:
                char_dict[char] = len(char_dict)
        return char_dict
    else:
        if not os.path.exists(BASE_DIR + '/data/char_dict.pkl'):
            char_dict = {}
            for i in range(0x4e00, 0x9fa6):
                if chr(i) not in char_dict:
                    char_dict[chr(i)] = len(char_dict)
            print(len(char_dict))
            save(BASE_DIR + '/data/char_dict.pkl', char_dict)
        return load(BASE_DIR + '/data/char_dict.pkl')


if __name__ == '__main__':
    print(BASE_DIR)
