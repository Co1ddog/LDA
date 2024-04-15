current_authors = [
    'Zhao, Xingping', 'Liu, Minqiang', 'Wu, Susu', 'Zhang, Baiyun',
    'Burjoo, Arvind', 'Yang, Yimin', 'Xu, Dabao'
]

# 使用列表推导式移除每个作者名字中的逗号
formatted_authors = [name.replace(',', '') for name in current_authors]

print(formatted_authors)
