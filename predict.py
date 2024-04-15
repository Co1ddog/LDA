import spacy
import joblib
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# 加载英语模型
nlp = spacy.load("en_core_web_sm")
# 训练好的LDA模型和CountVectorizer实例
lda_model_path = '/Users/colddog/Desktop/pythonprojects/lda-test/lda_best_model.joblib'
vectorizer_path = '/Users/colddog/Desktop/pythonprojects/lda-test/count_vectorizer.joblib'

# 加载保存的模型和向量器
lda_model = joblib.load(lda_model_path)
vectorizer = joblib.load(vectorizer_path)


# 定义自定义预处理函数
def custom_preprocessor(text):
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


def parse_document(text):
    authors = []
    abstract = ''
    lines = text.split('\n')

    # 用于判断当前是否在读取作者列表
    in_author_list = False

    for line in lines:
        # 开始读取作者列表
        if line.startswith('AF '):
            in_author_list = True
            author = ' '.join(line.split()[1:])  # 移除 'AF ' 并合并姓名
            authors.append(author)
        # 作者列表以 'AF ' 开头，后续行以空格开头
        elif in_author_list and line.startswith(' '):
            author = ' '.join(line.strip().split())  # 移除前导空格，并合并姓名
            authors.append(author)
        # 不再是作者列表部分
        elif line and not line.startswith(' '):
            in_author_list = False

        # 摘要只在一个 'AB ' 行开始，直到遇到 'ER' 结束
        if line.startswith('AB '):
            abstract = line[3:]
        elif line.startswith('ER'):
            # 到达条目末尾，不需要再读取任何内容
            break
        elif abstract:
            abstract += line.strip() + ' '

    return authors, abstract.strip()


def read_documents(file_path):
    # 读取整个文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 用ER分割不同的文档
    raw_documents = content.split('ER\n\n')
    parsed_documents = []

    for doc in raw_documents:
        authors, abstract = parse_document(doc)
        parsed_documents.append({'authors': authors, 'abstract': abstract})

    return parsed_documents


# 预处理文档，获取词频矩阵
def preprocess_and_predict(text, vectorizer, lda_model):
    # 使用custom_preprocessor进行文本预处理
    processed_text = custom_preprocessor(text)

    # 转换文档为词频矩阵
    doc_vectorized = vectorizer.transform([processed_text])

    # 预测主题
    topic_probs = lda_model.transform(doc_vectorized)
    # 获取最可能的主题
    topic = topic_probs.argmax()
    return topic


# 解析文档并预测主题
def parse_and_predict(text, vectorizer, lda_model):
    authors, abstract = parse_document(text)
    topic = preprocess_and_predict(abstract, vectorizer, lda_model)
    return authors, abstract, topic


def main():
    # 文献数据路径
    file_path = '/Users/colddog/Desktop/pythonprojects/lda-test/predict_reference'

    # 输出文件路径
    output_path = '/Users/colddog/Desktop/pythonprojects/lda-test/predicted_topics.txt'

    # 读取文献信息
    documents = read_documents(file_path)

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for document in documents:
            abstract = document['abstract']  # 使用摘要文本进行预测
            authors = document['authors']
            topic = preprocess_and_predict(abstract, vectorizer, lda_model)
            output_file.write(f"Authors: {authors}\n")
            output_file.write(f"Abstract: {abstract}\n")
            output_file.write(f"Topic: {topic}\n")
            output_file.write("----------\n")
    print("写入完毕")


if __name__ == '__main__':
    main()
