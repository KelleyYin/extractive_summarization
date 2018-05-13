# -*- coding: utf-8 -*-
# @Time    : 2018/5/10 16:19
# @Author  : yzhao

import json
import pickle
import jieba

tests = []
with open('data/tasktestdata03.txt', 'rt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        d = json.loads(line.strip())
        d['article'] = d['article'].replace('<Paragraph>', '\n')
        d['article'] = ' '.join([w for w in jieba.cut(d['article'])])
        d['index'] = i
        tests.append(d)

with open('data/tests_cut.pkl', 'wb') as f:
    pickle.dump(tests, f)
