import pandas as pd 
import nltk
from nltk.corpus import wordnet as wn, stopwords
from nltk.tokenize import word_tokenize

DATASET = '/Users/abdurrehman/Desktop/Oulu Courses /NLP/kaggle_text_to_emotion/Kaggle_Text_to_Emotion/Dataset/tweet_emotions.csv'

#reading the dataset
df = pd.read_csv(DATASET)

#Task 2 
dict = {}
for index, row in df.iterrows():
    emotion = row['sentiment']
    if emotion not in dict.keys():
        dict[emotion] = ['']
    else:
        dict[emotion][0] += (row['content']) 
df_list = pd.DataFrame.from_dict(dict)

# print(df_list.head())

#checking for entry in wordnet 
stopwords_list = stopwords.words('english')
table = {}
for col in df_list.columns:
    entry = 0
    no_entry = 0 
    text = df_list.loc[0][col]
    text = word_tokenize(text)
    text = [word.lower() for word in text if word.lower() in stopwords_list]
    text = list(set(text))
    for word in text:
        if len(wn.synsets(word)) > 0:
            entry += 1 
        else:
            no_entry += 1
    table[col] = (entry, no_entry, str(round((no_entry * 100)/(entry + no_entry), 3)) + '%')
# print(table)

#outputting as a table
df_table = pd.DataFrame(table, ['Identified Entries', 'Unidentified Entries', 'Percentage'])
print(df_table)