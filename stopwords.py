import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

english_stopwords = stopwords.words('english')

print("原来的：")
print(english_stopwords)

new_stopwords_verb = ['could', 'going', 'went', 'gone', 'seems', 'gives', 'takes', 'asks', 'sees', 'thinks', 'looks',
                      'gets', 'says']
new_stopwords_social = ['like', 'okay', 'oh', 'yes', 'hey', 'hmm', 'ah', 'um']
new_stopwords_indicator = ['thing', 'something', 'anything', 'someone', 'somewhere', 'one']
new_stopwords_adverb = ['really', 'quite', 'basically', 'actually', 'probably', 'literally', 'completely', 'generally',
                        'mostly']
new_stopwords_time = ['day', 'week', 'month', 'year', 'time', 'minute', 'hour']

english_stopwords.extend(new_stopwords_time)
english_stopwords.extend(new_stopwords_adverb)
english_stopwords.extend(new_stopwords_indicator)
english_stopwords.extend(new_stopwords_social)
english_stopwords.extend(new_stopwords_verb)

english_stopwords_set = set(english_stopwords)

print("现在的：")
print(english_stopwords_set)
