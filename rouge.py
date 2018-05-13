# -*- coding: utf-8 -*-
# @Time    : 2018/5/10 12:18
# @Author  : yzhao
from sumeval.metrics.rouge import RougeCalculator

from text_utils import ix_str


def rouge_score(summarizations, references, length=60, transform=True):
    assert len(summarizations) == len(references)
    rouge = RougeCalculator(stopwords=True, lang="en")
    total_rouge_1 = 0
    total_rouge_2 = 0
    total_rouge_l = 0
    count = len(summarizations)
    for s, r in zip(summarizations, references):
        if isinstance(s, list):
            s = ' '.join(s)
        if isinstance(r, list):
            r = ' '.join(r)
        assert type(s) == type(r)
        if transform:
            s = ix_str(s, length)
            r = ix_str(r, length)

        rouge_1 = rouge.rouge_n(summary=s, references=r, n=1)
        rouge_2 = rouge.rouge_n(summary=s, references=r, n=2)
        rouge_l = rouge.rouge_l(summary=s, references=r)
        total_rouge_1 += rouge_1
        total_rouge_2 += rouge_2
        total_rouge_l += rouge_l

    print('rouge_1:{}\nrouge_2:{}\nrouge_l:{}'
          .format(total_rouge_1 / count, total_rouge_2 / count, total_rouge_l / count))


def pyrouge_socre():
    from pyrouge import Rouge155
    system_dir = 'system'
    gold_dir = 'gold'

    r = Rouge155('/home/yzhao/soft/ROUGE-1.5.5/RELEASE-1.5.5/')
    r.system_dir = system_dir
    r.model_dir = gold_dir
    r.system_filename_pattern = 'system.(\d+).txt'
    r.model_filename_pattern = 'gold.[A-Z].#ID#.txt'
    output = r.convert_and_evaluate()
    print(output)
    output_dict = r.output_to_dict(output)


if __name__ == '__main__':
    pyrouge_socre()

