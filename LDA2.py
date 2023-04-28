import os
import jieba
from gensim import corpora, models
import re
import numpy as np

f = open("D:/课程相关/001_NLP/第三次作业/baidu_stopwords.txt", 'r', encoding='UTF-8', errors='ignore')
stop = f.read()
stop = stop.split()


def content_deal(content):  # 语料预处理，进行断句，去除一些广告和无意义内容
    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com',
          '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '='
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
    for a in ad:
        content = content.replace(a, '')
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', content)
    return chinese


path = "D:/课程相关/001_NLP/第一次作业/文本"
content = []
test = []
names = os.listdir(path)
for name in names:
    print(name)
    con_temp = []
    test_temp = []
    novel_name = path + '\\' + name
    with open(novel_name, 'r', encoding='ANSI') as f:
        con = f.read()
        con = content_deal(con)
        con1 = jieba.cut(con)  # 结巴分词  以词为单位进行分类
        # con1=con             #    以字为单位进行分类
        con = [word for word in con1 if word not in stop]
        con_list = list(con)
        pos = int(len(con) // 13)  ####16篇文章，分词后，每篇均匀选取13个500词段落进行建模
        for i in range(13):
            con_temp = con_temp + con_list[i * pos:i * pos + 500]
        for i in range(5):
            test_temp = test_temp + con_list[i * pos + 500:i * pos + 1000]
        content.append(con_temp)
        test.append(test_temp)
    f.close()

dictionary = corpora.Dictionary(content)
dictionary.filter_n_most_frequent(100)
corpus = [dictionary.doc2bow(text) for text in content]
lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=16)

topic_list = lda.print_topics(16)
print(topic_list)
corpus_test = [dictionary.doc2bow(text) for text in test]
topics_test = lda.get_document_topics(corpus_test)
labels = list(names)
for i in range(208):
    label = labels[int(i // 13)].replace('.txt', '')
    print(label + '的段落的主题分布为：\n')
    print(topics_test[int(i // 13)], '\n')

right_index = 0
rright = 0
for i in range(208):
    new = np.array(topics_test[int(i // 13)])
    max_index = np.argmax(new[:, 1])
    if int(i // 13) == right_index:
        right = new[max_index, 0]
        right_index += 1
    else:
        if right == new[max_index, 0]:
            rright += 1
print(rright / 208)
