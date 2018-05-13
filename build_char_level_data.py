# -*- coding: utf-8 -*-
# @Time    : 2018/5/10 20:25
# @Author  : yzhao

from resource import *

trains_c = []

for train in load_trains():
    train['summarization'] = ' '.join([c for c in train['summarization']])
    train['article'] = ' '.join([c for c in train['article']])
    trains_c.append(train)

evaluations_c = []
for dev in load_evaluations():
    dev['summarization'] = ' '.join([c for c in dev['summarization']])
    dev['article'] = ' '.join([c for c in dev['article']])
    evaluations_c.append(dev)

save('data/trains_c.pkl', trains_c)
save('data/evaluations_c.pkl', evaluations_c)

