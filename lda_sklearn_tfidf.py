import matplotlib
import matplotlib.pyplot as plt
import re
import pyLDAvis.lda_model

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

matplotlib.use('TkAgg')

custom_stopwords = ['could', 'going', 'went', 'gone', 'seems', 'gives', 'takes', 'asks', 'sees', 'thinks',
                    'looks', 'gets', 'says', 'using', 'proposed', 'based', 'like', 'okay', 'oh', 'yes', 'hey',
                    'hmm', 'ah', 'um', 'thing', 'something', 'anything', 'someone', 'somewhere', 'one', 'really',
                    'quite', 'basically', 'actually', 'probably', 'literally', 'completely',
                    'generally', 'mostly', 'day', 'week', 'month', 'year', 'time', 'minute', 'hour', 'one', 'two',
                    'three', 'four', 'first', 'second', ' third', 'firstly', 'secondly', 'thirdly', 'background',
                    'results', 'methods', 'conclusions', 'objective', 'result', 'method',
                    'conclusion', 'pt', 'ab', 'data', 'er', 'data',
                    'system', 'technology', 'technologies', 'systems', 'paper', 'information', 'j', 'c', 'ti',
                    'people', 'peoples', 'study', 'things', 'threshold', 'discover', 'discovered', 'equal', 'gender',
                    'used', 'uses', 'use', 'easily', 'devoted', 'devote', 'january', 'february', 'march', 'april', 'may',
                    'june', 'july', 'august', 'september', 'october', 'november', 'december', 'zero', 'blockchain']


def load_data():
    path_to_text = '/Users/colddog/Desktop/pythonprojects/lda-test/paper_abstract_3000+'
    documents = []
    with open(path_to_text, 'r', encoding='utf-8') as file:
        # 分割文档
        content = file.read()
        raw_documents = content.split('ER\n\n')

    for doc in raw_documents:
        # 移除文档开始的标记（PT J或PT C），如果有的话
        doc_cleaned = re.sub(r'^PT [JC]\nAB ', '', doc)
        # 移除所有数字
        doc_cleaned = re.sub(r'\d+', '', doc_cleaned)
        documents.append(doc_cleaned.strip())

    # 移除可能的空字符串文档
    documents = [doc for doc in documents if doc]
    return documents


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("主题 %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def display_topics_with_weights(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("主题 %d:" % (topic_idx))
        top_words = [(feature_names[i], topic[i]) for i in topic.argsort()[:-no_top_words - 1:-1]]
        for word, weight in top_words:
            print(f"{word}: {weight:.4f}")
        print("\n" + "-" * 40 + "\n")

def main():
    documents = load_data()

    # 创建文档-词频矩阵
    combined_stopwords = list(ENGLISH_STOP_WORDS.union(custom_stopwords))
    tfidf_vectorizer = TfidfVectorizer(max_df=0.25, min_df=3, stop_words=combined_stopwords)
    dtm = tfidf_vectorizer.fit_transform(documents)

    # 设置主题数并训练LDA模型
    topic_nums = list(range(1, 15, 1))
    perplexities = []

    for n_topics in topic_nums:
        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=25, learning_method='online',
                                        learning_decay=0.5, learning_offset=50., random_state=0,
                                        doc_topic_prior=0.1, topic_word_prior=0.1)
        lda.fit(dtm)
        # 计算困惑度
        perplexity = lda.perplexity(dtm)
        perplexities.append(perplexity)
        print(f"主题数: {n_topics}, 困惑度: {perplexity}")
        print(f"\n主题数为{n_topics}的各主题对应特征词")
        display_topics(lda, tfidf_vectorizer.get_feature_names_out(), no_top_words=20)

        # 每个模型的主题和权重
        # print(f"\n")
        # display_topics_with_weights(lda, count_vectorizer.get_feature_names_out(), no_top_words=10)
        print("\n" + "-" * 40 + "\n")

    # 选择展示具有最低困惑度的主题
    best_topic_idx = perplexities.index(min(perplexities))
    best_n_topics = topic_nums[best_topic_idx]
    best_perplexity = min(perplexities)
    print(f"最佳主题数: {best_n_topics}, 困惑度: {best_perplexity}")

    # 展示最佳主题数的主题
    lda_best = LatentDirichletAllocation(n_components=best_n_topics, max_iter=25, learning_method='online',
                                         learning_decay=0.5, learning_offset=50., random_state=0,
                                         doc_topic_prior=0.1, topic_word_prior=0.1)
    lda_best.fit(dtm)
    display_topics(lda_best, tfidf_vectorizer.get_feature_names_out(), no_top_words=20)

    # 使用pyLDAvis生成LDA可视化并保存为HTML
    data = pyLDAvis.lda_model.prepare(lda_best, dtm, tfidf_vectorizer)
    # 指定输出文件路径
    output_file_path = '/Users/colddog/Desktop/pythonprojects/lda-test/lda_vis_tfidf.html'
    pyLDAvis.save_html(data, output_file_path)
    print(f"最佳主题可视化已保存至: {output_file_path}")

    # 可视化困惑度
    plt.figure(figsize=(12, 6))
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.plot(topic_nums, perplexities, marker='o')
    plt.title('困惑度得分')
    plt.xlabel('主题个数')
    plt.ylabel('困惑度')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
