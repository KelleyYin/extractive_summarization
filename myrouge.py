from pyrouge import Rouge155
import shutil
import os

rouge_path = '/home/yzhao/soft/RELEASE-1.5.5'


def clear_dir(dpath):
    if os.path.exists(dpath):
        shutil.rmtree(dpath)
    os.mkdir(dpath)


def eval_rouge(golds, results):

    assert len(golds) == len(results)

    clear_dir('gold_summaries')
    clear_dir('result_summaries')

    r = Rouge155(rouge_path)
    r.system_dir = 'result_summaries'
    r.model_dir = 'gold_summaries'
    r.system_filename_pattern = 'result.(\d+).txt'
    r.model_filename_pattern = 'gold.[A-Z].#ID#.txt'

    for i in range(len(golds)):
        output_gold = open('gold_summaries/gold.A.%d.txt' % i, 'w')
        output_result = open('result_summaries/result.%d.txt' % i, 'w')

        output_gold.write(golds[i])
        output_result.write(results[i])

        output_gold.close()
        output_result.close()

    output = r.convert_and_evaluate()
    print(output)
    return r.output_to_dict(output)


