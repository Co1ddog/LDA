import matplotlib
import matplotlib.pyplot as plt
import re
import pyLDAvis.lda_model
import spacy
from joblib import dump
from spacy.matcher import PhraseMatcher

# 加载英语模型
nlp = spacy.load("en_core_web_sm")

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation

matplotlib.use('TkAgg')

custom_stopwords = ['could', 'going', 'went', 'gone', 'seems', 'gives', 'takes', 'asks', 'sees', 'thinks',
                    'looks', 'gets', 'says', 'using', 'proposed', 'based', 'like', 'okay', 'oh', 'yes', 'hey',
                    'hmm', 'ah', 'um', 'thing', 'something', 'anything', 'someone', 'somewhere', 'one', 'really',
                    'quite', 'basically', 'actually', 'probably', 'literally', 'completely',
                    'generally', 'mostly', 'day', 'week', 'month', 'year', 'time', 'minute', 'hour', 'one', 'two',
                    'three', 'four', 'first', 'second', ' third', 'firstly', 'secondly', 'thirdly', 'background',
                    'results', 'methods', 'conclusions', 'objective', 'result', 'method',
                    'conclusion', 'pt', 'ab', 'data', 'er',
                    'system', 'technology', 'technologies', 'systems', 'paper', 'information', 'j', 'c', 'ti',
                    'people', 'peoples', 'study', 'things', 'threshold', 'discover', 'discovered', 'equal', 'gender',
                    'used', 'uses', 'use', 'easily', 'devoted', 'devote', 'january', 'february', 'march', 'april',
                    'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'zero',
                    'blockchain',
                    'new', 'high', 'low', 'research', 'model', 'analysis', 'approach', 'process', 'technique',
                    'framework', 'different', 'various', 'smart', 'important', 'access', 'control', 'user', 'users',
                    'system', 'systems', 'method', 'approach', 'technique', 'model', 'models', 'review', 'study',
                    'studies', 'attribute', 'attributes', 'search', 'number', 'key', 'attack', 'attacks', 'improve',
                    'human', 'big', 'small', 'domain', 'role', 'rate', 'internet', 'make', 'makes', 'making', 'single',
                    'medical', 'health', 'solution', 'propose', 'issue', 'issues', 'include', 'includes', 'including',
                    'work', 'provide', 'compare', 'compared', 'comparing', 'existing', 'exist', 'exists', 'existed',
                    'scheme', 'schemes', 'finally', 'effectively', 'impact', 'knowledge', 'problem', 'problems',
                    'solutions', 'need', 'needed', 'needs', 'end', 'range', 'wide', 'allow', 'allows', 'enables',
                    'enable', 'enabled']

matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
phrases = ['supply chain', 'machine learning', 'deep learning']
patterns = [nlp(text) for text in phrases]
matcher.add("TerminologyList", patterns)


def load_data():
    path_to_text = '/Users/colddog/Desktop/pythonprojects/lda-test/new_reference'
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


def custom_preprocessor(doc):
    spacy_doc = nlp(doc.lower())
    matches = matcher(spacy_doc)
    with spacy_doc.retokenize() as retokenizer:
        for _, start, end in matches:
            retokenizer.merge(spacy_doc[start:end])
    tokens = [token.text for token in spacy_doc if
              not token.is_stop and not token.is_punct and token.text in nlp.vocab]
    return " ".join(tokens)


def custom_preprocessor_2(text):
    # 先替换文本
    text = text.replace('supply chain', 'supply_chain')
    text = text.replace('machine learning', 'machine_learning')
    text = text.replace('deep learning', 'deep_learning')
    text = text.replace('federated learning', 'federated_learning')
    text = text.replace('decentralized learning', 'decentralized_learning')
    text = text.replace('case study', 'case_study')
    text = text.replace('state of the art', 'state_of_the_art')
    # 然后进行分词
    spacy_doc = nlp(text)
    tokens = [token.lemma_ for token in spacy_doc if
              not token.is_stop and not token.is_punct and token.text in nlp.vocab]
    return " ".join(tokens)


def custom_preprocessor_3(text):
    # 先替换特定短语为单一token
    replacements = {
        'supply chain': 'supply_chain',
        'machine learning': 'machine_learning',
        'deep learning': 'deep_learning',
        'federated learning': 'federated_learning',
        'decentralized learning': 'decentralized_learning',
        'case study': 'case_study',
        'state of the art': 'state_of_the_art'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 进行分词和词性标注
    spacy_doc = nlp(text)

    # 定义想要保留的词性列表
    allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']

    # 定义不想保留的词形
    unwanted_token_morphs = ['VerbForm=Inf', 'VerbForm=Part', 'Number=Plur']

    # 过滤掉不在允许词性中或者具有不想要的词形的词
    tokens = []
    for token in spacy_doc:
        # 检查词性和词形
        if token.pos_ in allowed_postags and not any(morph in token.morph for morph in unwanted_token_morphs):
            # 检查是否停用词或标点符号
            if not token.is_stop and not token.is_punct and token.text in nlp.vocab:
                tokens.append(token.text)

    return " ".join(tokens)


def main():
    documents = load_data()
    processed_docs = [custom_preprocessor_2(doc) for doc in documents]

    # 创建文档-词频矩阵
    combined_stopwords = list(ENGLISH_STOP_WORDS.union(custom_stopwords))
    count_vectorizer = CountVectorizer(max_df=0.55, min_df=150, stop_words=combined_stopwords)
    dtm = count_vectorizer.fit_transform(processed_docs)

    # 设置主题数并训练LDA模型
    topic_nums = list(range(1, 11, 1))
    perplexities = []

    for n_topics in topic_nums:
        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50, learning_method='online',
                                        learning_decay=0.4, learning_offset=50., random_state=0,
                                        doc_topic_prior=0.07, topic_word_prior=0.8)
        lda.fit(dtm)
        # 计算困惑度
        perplexity = lda.perplexity(dtm)
        perplexities.append(perplexity)
        print(f"主题数: {n_topics}, 困惑度: {perplexity}")
        print(f"\n主题数为{n_topics}的各主题对应特征词")
        display_topics(lda, count_vectorizer.get_feature_names_out(), no_top_words=20)

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
    lda_best = LatentDirichletAllocation(n_components=best_n_topics, max_iter=50, learning_method='online',
                                         learning_decay=0.4, learning_offset=50., random_state=0,
                                         doc_topic_prior=0.07, topic_word_prior=0.8)
    lda_best.fit(dtm)
    display_topics(lda_best, count_vectorizer.get_feature_names_out(), no_top_words=20)

    # 使用pyLDAvis生成LDA可视化并保存为HTML
    data = pyLDAvis.lda_model.prepare(lda_best, dtm, count_vectorizer)
    # 指定输出文件路径
    output_file_path = '/Users/colddog/Desktop/pythonprojects/lda-test/lda_vis_new.html'
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

    # 指定保存模型和向量器的文件名
    lda_model_filename = '/Users/colddog/Desktop/pythonprojects/lda-test/lda_best_model.joblib'
    vectorizer_filename = '/Users/colddog/Desktop/pythonprojects/lda-test/count_vectorizer.joblib'

    # 保存 LDA 模型
    dump(lda_best, lda_model_filename)
    print(f"LDA 模型已保存至: {lda_model_filename}")

    # 保存 CountVectorizer
    dump(count_vectorizer, vectorizer_filename)
    print(f"CountVectorizer 已保存至: {vectorizer_filename}")


if __name__ == '__main__':
    main()
