import pandas as pd
import numpy as np
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

DATASET = '/Users/abdurrehman/Desktop/Oulu Courses /NLP/kaggle_text_to_emotion/Kaggle_Text_to_Emotion/Dataset/tweet_emotions.csv'

#reading the dataset
df = pd.read_csv(DATASET)

#creating a dictionary -> sentiments : [list of tweets of that particular sentiment]
sentiments_dict = {}

#!!! This section performs the list concatenation (was unsure whether with concatenation they meant string-wise or list-wise, so implemented both) 
# #iterating over all the dataframe rows and ordering the tweets based on the sentiment/emotion
# for index, row in df.iterrows():
#     emotion = row['sentiment']
#     if emotion not in sentiments_dict.keys():
#         sentiments_dict[emotion] = []
#     else:
#         sentiments_dict[emotion].append(row['content'])

# #was having an error while converting the resulting dictionary of sentiments due to unbalance frequencies. Therefore, balancing them.
# max_size = max(len(value) for value in sentiments_dict.values())

# #now filling NA values for all other sentiments
# for i in sentiments_dict.values():
#         empty_string = [''] * (max_size - len(i))
#         i.extend(empty_string)

# #all the sentiments are converted to dataframe with equal frequency 
# df_list = pd.DataFrame.from_dict(sentiments_dict)
# # print(df_list.head())

# #10 most frequent keywords in each category
# columns = df_list.columns
# for column in columns:
#      for index, rows in df.iterrows():
#           print(column)
# # print(columns[0])
#!!!

#iterating over all the dataframe rows and ordering the tweets based on the sentiment/emotion
for index, row in df.iterrows():
    emotion = row['sentiment']
    if emotion not in sentiments_dict.keys():
        sentiments_dict[emotion] = ['']
    else:
        sentiments_dict[emotion][0] += (row['content'])

# #all the sentiments are converted to dataframe with equal frequency 
df_list = pd.DataFrame.from_dict(sentiments_dict)

# #10 most frequent keywords in each category
most_frequent_words_before = {}
for col in df_list.columns:
    text = word_tokenize(df_list.loc[0][col])
    most_frequent_words_before[col] = FreqDist(text).most_common(10)
print()
print('[MOST FREQUENT WORDS BEFORE PREPROCESSING]: \n', most_frequent_words_before)
print()

stopwords_list = stopwords.words('english')
most_frequent_words_after = {}
for col in df_list.columns:
    text = df_list.loc[0][col]
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = word_tokenize(text)
    text = [words.lower() for words in text if not words.lower() in stopwords_list]
    most_frequent_words_after[col] = FreqDist(text).most_common(10)
print('[MOST FREQUENT WORDS AFTER PREPROCESSING]: \n', most_frequent_words_after)

def output_as_percentage(most_frequent_words_before, most_frequent_words_after):
    total_words_before = 0
    total_words_after = 0

#counting total words to calculate the percentage
    for sentiment, occurences in most_frequent_words_before.items():
        for word, frequency in occurences:
            total_words_before += frequency
    for sentiment, occurences in most_frequent_words_after.items():
        for word, frequency in occurences:
            total_words_after += frequency
    for sentiment, occurences in most_frequent_words_before.items():
        for index in range(len(occurences)):
            most_frequent_words_before[sentiment][index] = list(most_frequent_words_before[sentiment][index])
            most_frequent_words_before[sentiment][index][1] = str(round((most_frequent_words_before[sentiment][index][1] * 100)/ total_words_before, 3)) + '%'
            most_frequent_words_before[sentiment][index] = tuple(most_frequent_words_before[sentiment][index])
    for sentiment, occurences in most_frequent_words_after.items():
        for index in range(len(occurences)):
            most_frequent_words_after[sentiment][index] = list(most_frequent_words_after[sentiment][index])
            most_frequent_words_after[sentiment][index][1] = str(round((most_frequent_words_after[sentiment][index][1] * 100)/ total_words_after, 3)) + '%'
            most_frequent_words_after[sentiment][index] = tuple(most_frequent_words_after[sentiment][index])

    
    print()
    print('[MOST FREQUENT WORDS BEFORE PREPROCESSING AS %]: \n', most_frequent_words_before)
    print()
    print('[MOST FREQUENT WORDS AFTER PREPROCESSING AS %]: \n', most_frequent_words_after)


output_as_percentage(most_frequent_words_before=most_frequent_words_before, most_frequent_words_after=most_frequent_words_after)