from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator
from text_utils import split_sentence, remove_stop_words, write2file


def cal_sim(sentence, doc):
    """
    计算一个句子和一组句子的相似度
    :param sentence: 一个句子字符串
    :param doc: 其他字符串列表
    :return: 返回余弦相似度
    """
    if len(doc) == 0:
        return 0
    features = sentence.split()  # 词汇特征
    doc_2_one_sentence = ' '.join(doc)
    features.extend(doc_2_one_sentence.split())
    cv = CountVectorizer(vocabulary=set(features))
    doc_vector = cv.fit_transform([doc_2_one_sentence])
    sentence_vector = cv.fit_transform([sentence])
    return cosine_similarity(doc_vector, sentence_vector)[0][0]


def get_similarity_score(texts):
    """
    计算文本中一个句子和其他句子的相似度
    :param texts: 整个文本，需要分句
    :return: 每个句子分数组成的字典，和去除停用词后的句子对应的字典
    """
    sentences = []
    clean = []
    original_sentences = {}

    for i, s in enumerate(split_sentence(texts)):
        cl = remove_stop_words(s)
        sentences.append(s)  # 原本的句子
        clean.append(cl)  # 干净有重复的句子
        original_sentences[cl] = s, i  # 字典格式
    clean_set = set(clean)  # 干净无重复的句子
    scores = {}
    for data in clean:
        other_doc = clean_set - set(data)  # 在除了当前句子的剩余所有句子
        score = cal_sim(data, list(other_doc))  # 计算当前句子与剩余所有句子的相似度
        scores[data] = score

    return scores, original_sentences


def get_original_sentences(summary_set, original_sentences):
    selected = []
    for sentence in summary_set:
        selected.append(original_sentences[sentence])
    selected.sort(key=lambda x: x[1])
    summary = [s[0].strip() for s in selected]
    return '\n'.join(summary)


def mmr_summary(text, n=20, alpha=0.8):
    scores, original_sentences = get_similarity_score(text)
    summary_set = set()
    for _ in range(n):
        mmr = {}
        for sentence in scores.keys():
            if sentence not in summary_set:
                # sim2 = [cal_sim(sentence, [summary]) for summary in summary_set]
                # (0 if len(sim2) == 0 else max(sim2)
                # 这里的查询值直接用其他分数代替
                mmr[sentence] = alpha * scores[sentence] - \
                                (1 - alpha) * cal_sim(sentence, list(summary_set))
        if len(mmr) == 0:
            break
        selected = max(mmr.items(), key=operator.itemgetter(1))[0]
        summary_set.add(selected)
    summary = get_original_sentences(summary_set, original_sentences)
    return summary


if __name__ == '__main__':
    # summarizations = []
    # references = []
    # for e in load_evaluations_cut():
    #     references.append(e['summarization'])
    #     summarizations.append(mmr_summary(e['article']))
    # write2file(summarizations, references)

    text = '''24岁的男子小罗（化名）因患有精神病被送进医院，医院用保护带将其绑在床上。护士发现小罗被同室病人掐死。
    小罗母亲提起刑事自诉，市一中院终审判处当班护士杨某犯医疗事故罪，但由于医院已与家属达成协议，且杨某已被医院开除，
    故对其免予刑事处罚。24岁的被害人小罗因精神受到刺激被送往本市海淀区北京大学第六医院治疗。医院用保护带将其绑在床上，
    小罗被发现已死亡。杨某在本市海淀区北京医科大学附属第六医院住院部，对已约束的病人未按规定定时松解保护带。
    北京医科大学附属第六医院与被害人家属达成协议，赔偿被害人家属50万元并承担其他损失费用。医院对护士杨某和郝某予以行政处分，
    供述当班护士认为“应由医院担责”杨某供述称，护士站急收了病人小罗，小罗比较兴奋吵闹，他和另外一名实习护士郝某与前班护士办理交接。
    杨某首先隔着门玻璃查看清点了一下人数，杨某发现小罗的病房里又新进了一个病人唐某。杨某在交接班之后，没有进病房查看，
    之后就再没有巡视查看过。郝某找到杨某说小罗有些不对劲。发现小罗还被捆着，杨某认为自己虽有过错，但尚不构成犯罪，
    受害人小罗的死亡结果应当由医院承担责任。杨某辩护人认为，也有医院管理混乱，上一班医护人员错误安排病房、
    没有及时松解保护带等方面原因。杨某未按规定巡视并不必然导致被害人小罗死亡的结果发生。
    追责家属向护士提起刑事自诉郝某称，应该对病人每隔10分钟巡视1次，对于小罗这样有被约束病人的病房里，
    是不能再入住没有被约束的病人的。医院出具的特护记录显示，杨某的前一班护士对小罗都有详细且明确的护理记录，
    就没有任何关于对小罗的护理记录，杨某一晚上都没有对小罗进行过巡查，也没有按规定护理。检察机关认定被告人杨某涉嫌医疗事故罪，
    但情节轻微，小罗的母亲赵某向海淀法院提起刑事自诉。判决构成医疗事故罪免予刑事处罚海淀法院认为，
    虽然被害人小罗的死亡是由第三人病态行为直接造成，杨某身为当晚值班副班护士，是事发当时代表医院承担巡视、护理职责，
    其严重不负责的行为是医院没有履行好保护患者安全职责的表现之一，医院多方面的过失并非仅限于杨某单个因素。故杨某虽构成犯罪，
    但情节相对轻微，同时事后杨某已被医院处以开除的行政处分，故法院认为可以对其免予刑事处罚。免予刑事处罚。杨某也提起了上诉，
    杨某认为被害人的死亡后果并非其造成的，其不构成犯罪。驳回杨某和赵女士的上诉'''
    print(mmr_summary(text))
