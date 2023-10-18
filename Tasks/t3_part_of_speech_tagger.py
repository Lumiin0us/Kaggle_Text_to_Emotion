import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.corpus import treebank

DATASET = '/Users/abdurrehman/Desktop/Oulu Courses /NLP/kaggle_text_to_emotion/Kaggle_Text_to_Emotion/Dataset/tweet_emotions.csv'

#reading the dataset
df = pd.read_csv(DATASET)

#From Task 2 - sentiment dataframes with col as sentiments and rows as concatenated content
dict = {}
for index, row in df.iterrows():
    emotion = row['sentiment']
    if emotion not in dict.keys():
        dict[emotion] = ['']
    else:
        dict[emotion][0] += (row['content']) 
df_list = pd.DataFrame.from_dict(dict)

#Task 3 POS Tagger
pos_tagger_dict_unigram = {}
pos_tagger_dict_penn_treebank = {}

#example sentence
sentence = 'The lazy brown fox jumped over the fence'
sentence = word_tokenize(sentence)

#golden tagset
tagged_sents = treebank.tagged_sents()

#Unigram Tagger (Our choice)
unigram_tagger = nltk.UnigramTagger(tagged_sents)

#45 Penn Tree Bank Tagger (In question), perceptronTagger uses 45 Penn Tree Bank for its POS tagging
perceptron_tagger = nltk.PerceptronTagger()

top_five_most_occuring_pos_tags_unigram = defaultdict(lambda: 0)
top_five_most_occuring_pos_tags_perceptron = defaultdict(lambda: 0)

def total_tags_sum(l):
    sum = 0 
    for index, items in enumerate(l):
        sum += int(items[1])

    new_list = l[0: 5].copy()
    for index, items in enumerate(new_list):
        new_list[index] = (items[0], str(round((int(items[1]) /sum) * 100, 3)) + '%')
    return new_list

#iterating over all the columns and picking the contcatenated tweet content and applying the POS tagging, the results are expressed as percentage
for col in df_list.columns:
    text = word_tokenize(df_list.loc[0][col])
    pos_tagger_dict_unigram[col] = unigram_tagger.tag(text)
    pos_tagger_dict_penn_treebank[col] = perceptron_tagger.tag(text)

    for tags in pos_tagger_dict_unigram[col]:
        if tags[1] not in top_five_most_occuring_pos_tags_unigram:
            top_five_most_occuring_pos_tags_unigram[tags[1]] += 1
        else:
            top_five_most_occuring_pos_tags_unigram[tags[1]] += 1
    for tags in pos_tagger_dict_penn_treebank[col]:
        if tags[1] not in top_five_most_occuring_pos_tags_perceptron:
            top_five_most_occuring_pos_tags_perceptron[tags[1]] += 1
        else:
            top_five_most_occuring_pos_tags_perceptron[tags[1]] += 1

    top_five_occurences_unigram =  sorted(top_five_most_occuring_pos_tags_unigram.items(), key=lambda x: x[1], reverse=True)
    top_five_occurences_perceptron =  sorted(top_five_most_occuring_pos_tags_perceptron.items(), key=lambda x: x[1], reverse=True)
    print(f'Most dominant tags in [{col}] USING [UNIGRAM_TAGGER]: ', total_tags_sum(top_five_occurences_unigram))
    print(f'Most dominant tags in [{col}] USING [PERCEPTRON_TAGGER]: ', total_tags_sum(top_five_occurences_perceptron))


