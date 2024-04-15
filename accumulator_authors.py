import re
from collections import defaultdict


# 解析文档函数，返回作者列表和主题
def parse_document(document):
    # 使用正则表达式找到作者和主题
    authors_match = re.search(r"Authors: \[(.*?)\]", document)
    topic_match = re.search(r"Topic: (\d+)", document)

    authors = []
    if authors_match:
        # 正确分割作者全名，移除多余的空格和引号
        authors_raw = authors_match.group(1)
        authors = [author.strip(" '") for author in authors_raw.split("', '")]
    topic = topic_match.group(1) if topic_match else None

    return authors, topic


# 处理文件函数，返回作者-主题频率的字典
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 根据"----------"分割文档，忽略最后一个空分割
    documents = content.strip().split("----------\n")[:-1]

    # 字典，用于保存作者-主题频率
    author_topic_freq = defaultdict(lambda: defaultdict(int))

    for document in documents:
        authors, topic = parse_document(document)
        for author in authors:
            author_topic_freq[author][topic] += 1

    return author_topic_freq


def main():
    # 文件路径
    file_path = 'predicted_topics.txt'
    output_path = '/Users/colddog/Desktop/pythonprojects/lda-test/author_frequency.txt'
    author_topic_freq = process_file(file_path)

    # 打印频率示例
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for author, topics in author_topic_freq.items():
            output_file.write(f"Author: {author}; Topics: {dict(topics)}\n")

    print("文件已保存到/Users/colddog/Desktop/pythonprojects/lda-test/author_frequency.txt")


if __name__ == '__main__':
    main()
