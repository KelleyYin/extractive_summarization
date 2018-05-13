# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 0011 21:23
# @Author  : uhauha2929
import re
import os
import shutil

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resource import load_stop_words, load_char_dict, load_evaluations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def similarity(s1, s2):
    """
    计算两个句子的余弦相似度
    :param s1:
    :param s2:
    :return:
    """
    cv = CountVectorizer(vocabulary=set(s1.split() + s2.split()))
    v1 = cv.fit_transform([s1])
    v2 = cv.fit_transform([s2])
    return cosine_similarity(v1, v2)[0][0]


def is_sent(s):
    """
    判断一个字符串可能不是一个句子
    :param s:
    :return:
    """
    cnt = 0
    s = re.sub('[ \t\r\n]', '', s)  # 注意空格
    for c in s.strip():
        if u'\u4e00' <= c <= u'\u9fa5':
            cnt += 1
    if cnt < len(s) / 2 or cnt <= 4:
        return False
    return True


def cut_sentence(sentence):
    """
    将没有分词的句子分词后用空格相连
    :param sentence:
    :return:
    """
    import jieba
    words = []
    for word in jieba.cut(sentence):
        words.append(word)
    return ' '.join(words)


def remove_stop_words(sentence):
    """
    去除句子中的停止词（句子已分词）
    :param sentence:
    :return:
    """
    stop_words = load_stop_words()
    pieces = [word for word in sentence.split() if word not in stop_words]
    return ' '.join(pieces)


def split_sentence(text):
    """适用于[中文]摘要的分句"""
    cut_list = "。！!?？；;，,【】[]：:" \
               "◆●〖〗■…①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳" \
               "⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇" \
               "⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛" \
               "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ" \
               "❶❷❸❹❺❻❼❽❾❿㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩"
    tmp = []
    ss = []
    for i in range(len(text)):
        tmp.append(text[i])  # 保留标点符号，作为特征
        if cut_list.__contains__(text[i]):
            s = ''.join(tmp).strip()
            if is_sent(s) and s[:-1] not in ss:  # remove duplicates
                ss.append(s)
            tmp = []
    if len(ss) == 0:
        ss.append(text)
    return ss


def ix_str(texts, limit=None):
    """
    将一串文本转换成字典中的下标表示
    :param texts:
    :param limit: 限制转换的字符的长度
    :return:
    """
    char_dict = load_char_dict()  # 只取字典中的字符，不包括标点符
    stop_words = load_stop_words()
    chars = []
    for text in texts:
        for char in re.sub('[ \t\r\n]', '', text):
            if len(chars) == limit:
                break
            if char in char_dict and char not in stop_words:
                chars.append(str(char_dict[char]))
    if len(chars) == 0:
        chars.append('null')
    return ' '.join(chars)


def write2file(summarizations, references, transform=True, length=60):
    """
    将摘要和参考摘要写到文件中，便于pyrouge计算得分
    :param summarizations: 自己的摘要, 所有摘要的列表，每一项可以是整个摘要，也可以是摘要句子列表
    :param references: 参考摘要，格式同上，且要一一对应
    :param transform: 是否转换成ID格式（处理中文）
    :param length: 限制摘要的长度
    :return:
    """
    gold_dir = BASE_DIR + '/gold'
    system_dir = BASE_DIR + '/system'
    if os.path.exists(gold_dir):
        shutil.rmtree(gold_dir)
    os.mkdir(gold_dir)
    if os.path.exists(system_dir):
        shutil.rmtree(system_dir)
    os.mkdir(system_dir)
    ix = 0
    for s, r in zip(summarizations, references):
        if isinstance(s, list):
            s = ' '.join(s)
        if isinstance(r, list):
            r = ' '.join(r)
        assert type(s) == type(r)
        file_path = '{}/system.{}.txt'.format(system_dir, ix)
        with open(file_path, 'w', encoding='utf-8') as f:
            if transform:
                f.write(ix_str(s, limit=length))
            else:
                f.write(s)

        file_path = '{}/gold.A.{}.txt'.format(gold_dir, ix)
        with open(file_path, 'w', encoding='utf-8') as f:
            if transform:
                f.write(ix_str(r, limit=length))
            else:
                f.write(r)
        ix += 1


def generate_result(summarizations):
    if os.path.exists('result.txt'):
        os.remove('result.txt')
    file = open('result.txt', 'at', encoding='utf-8')
    stop_words = load_stop_words()
    for i, s in enumerate(summarizations):
        if isinstance(s, list):
            s = ' '.join(s)
        s = ''.join([w for w in s.split() if w not in stop_words])
        s = re.sub('[ \t\r\n]', '', s).replace('\"', '\\\"')  # adapt to json format
        file.write('{{"summarization":"{}", "index":{}}}\n'.format(s, i))
    file.close()


