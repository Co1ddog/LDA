import gensim
import nltk
import re
import matplotlib
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim

matplotlib.use('TkAgg')

from pylab import xticks, yticks, np
from gensim.topic_coherence import direct_confirmation_measure
from my_custom_module import custom_log_ratio_measure
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import TfidfModel


# gensim模型的一致性计算有问题，在其issue里提出的解决方案如下
direct_confirmation_measure.log_ratio_measure = custom_log_ratio_measure


# plt.switch_backend('agg')


# 模型生成函数
def lda_model_values(num_topics, corpus, dictionary, text):
    x = []  # x轴
    perplexity_values = []  # 困惑度
    coherence_values = []  # 一致性
    model_list = []  # 存储对应主题数量下的lda模型,便于生成可视化网页

    for topic in range(num_topics):
        print("主题数量：", topic + 1)
        lda_model = gensim.models.LdaModel(corpus=corpus, num_topics=topic + 1, id2word=dictionary, chunksize=2000,
                                           passes=20, iterations=400)
        model_list.append(lda_model)
        x.append(topic + 1)
        perplexity_values.append(lda_model.log_perplexity(corpus))

        coherencemodel = gensim.models.CoherenceModel(model=lda_model, texts=text, dictionary=dictionary,
                                                      coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print('该模型的困惑度为：{},一致性为：{}'.format(perplexity_values, coherence_values))
        print("该主题评价完成\n")

    return model_list, x, perplexity_values, coherence_values


if __name__ == '__main__':

    # 加载英文停用词列表
    stop_words_list = stopwords.words('english')

    # 给英文停词表新增内容
    custom_stopwords = ['could', 'going', 'went', 'gone', 'seems', 'gives', 'takes', 'asks', 'sees', 'thinks',
                        'looks', 'gets', 'says', 'using', 'proposed', 'based', 'like', 'okay', 'oh', 'yes', 'hey',
                        'hmm', 'ah', 'um', 'thing', 'something', 'anything', 'someone', 'somewhere', 'one', 'really',
                        'quite', 'basically', 'actually', 'probably', 'literally', 'completely',
                        'generally', 'mostly', 'day', 'week', 'month', 'year', 'time', 'minute', 'hour', 'one', 'two',
                        'three', 'four', 'first', 'second', ' third', 'firstly', 'secondly', 'thirdly', 'background',
                        'results', 'methods', 'conclusions', 'objective', 'result', 'method',
                        'conclusion', 'pt', 'ab', 'data', 'er', 'data',
                        'system', 'technology', 'technologies', 'systems', 'paper', 'information', 'j', 'c', 'ti',
                        'people', 'peoples', 'study', 'things', 'enable', 'enabled', 'enables']

    stop_words_list.extend(custom_stopwords)

    # 转换为列表避免重复
    stop_words_list = set(stop_words_list)

    # lemmatizer = WordNetLemmatizer()

    # 数据文本路径
    path_to_text = '/Users/colddog/Desktop/pythonprojects/lda-test/paper_abstract_2023'
    # path_to_text = '/paper_abstract_3000+'
    # path_to_text = '/Users/colddog/Desktop/pythonprojects/lda-test/paper_title'
    # path_to_text = '/Users/colddog/Desktop/pythonprojects/lda-test/content_test'

    # 读取英文文本数据
    with open(path_to_text, 'r', encoding='utf-8') as file:
        sentence = file.read()

    # 清理文本，移除非字母字符，并转换为小写
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence).lower()

    # 使用nltk进行分词
    words = word_tokenize(sentence)

    # 移除停用词
    text = [word for word in words if word not in stop_words_list]

    # 生成词袋等模型需要的数据
    dictionary = gensim.corpora.Dictionary([text])

    # 过滤过于常见或者罕见的词汇
    # dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    corpus = [dictionary.doc2bow(tmp) for tmp in [text]]

    """下面为可视化绘画代码"""
    # 从1开始，循环到主题20，测试1-20的主题数的困惑度和一致性
    num_topics = 10
    # 调用模型生成函数
    model_list, x, perplexity_values, coherence_values = lda_model_values(num_topics, corpus, dictionary, sentence)

    # 输出最佳模型的主题分布，model_list[15]为生成的第16个模型
    for i in range(len(model_list)):
        print(model_list[i].print_topics(10, i + 1))
    # i默认为上面最后一次循环最大的序号

    # 绘制困惑度和一致性折线图
    fig = plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False

    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(x, perplexity_values, marker="o")
    plt.title("主题建模-困惑度")
    plt.xlabel('主题数目')
    plt.ylabel('困惑度大小')
    xticks(np.linspace(1, num_topics, num_topics, endpoint=True))  # 保证x轴刻度为1

    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(x, coherence_values, marker="o")
    plt.title("主题建模-一致性")
    plt.xlabel("主题数目")
    plt.ylabel("一致性大小")
    xticks(np.linspace(1, num_topics, num_topics, endpoint=True))

    plt.show()

    vis_data = pyLDAvis.gensim.prepare(model_list[i], corpus=corpus, dictionary=dictionary, mds='mmds')
    pyLDAvis.show(vis_data)
