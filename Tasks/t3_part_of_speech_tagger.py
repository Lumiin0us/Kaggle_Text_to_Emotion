import pandas as pd 
import nltk
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

print(df_list.head())

#Task 3 
#POS Tagger
pos_tagger_dict_unigram = {}
pos_tagger_dict_penn_treebank = {}


# for col in df_list.columns:
#     text = word_tokenize(df_list.loc[0][col])
#     pos_tagger_dict[col] = nltk.pos_tag(text)
# print(pos_tagger_dict['empty'])




# sentence = "Although the weather forecast predicted rain for the weekend, we decided to go camping, and we brought plenty of waterproof gear just in case the skies opened up."
# sentence = word_tokenize(sentence)

# # Load the test set
# test_set = nltk.corpus.brown.tagged_sents()

# #Perceptron Tagger(Default)
# # pos_tag = nltk.pos_tag(test_set)
# # print(pos_tag)
# # nltk.help.upenn_tagset()
# tagger = nltk.PerceptronTagger()
# tagg = tagger.tag(test_set)
# print(tagg)
# print('Accuracy', nltk.accuracy(tagg, test_set))

# default_tagger = nltk.DefaultTagger('NN')
# default_pred = default_tagger.tag(test_set)
# print(default_pred)
# print('Accuracy', default_tagger.accuracy(test_set))
# # print(len(tagger.classes))


from nltk.corpus import treebank

sentence = 'The lazy brown fox jumped over the fence'
sentence = word_tokenize(sentence)
tagged_sents = treebank.tagged_sents()

#!Unigram Tagger (Our choice)
unigram_tagger = nltk.UnigramTagger(tagged_sents)

# unigram_tagger_pred = unigram_tagger.tag(sentence)
# print(unigram_tagger_pred)
# print('Accuracy', unigram_tagger.accuracy(tagged_sents))

#!45 Penn Tree Bank Tagger (In question)
perceptron_tagger = nltk.PerceptronTagger()

# perceptron_tagger_pred = perceptron_tagger.tag(sentence)
# print(perceptron_tagger_pred)
# print('Accuracy', perceptron_tagger.accuracy(tagged_sents))
from collections import defaultdict
top_five_most_occuring_pos_tags_unigram = defaultdict(lambda: 0)
top_five_most_occuring_pos_tags_perceptron = defaultdict(lambda: 0)

def total_tags_sum(l):
    sum = 0 
    # new_list = []
    for index, items in enumerate(l):
        # new_list.append((key, items))
        sum += int(items[1])
    # print(l)
    new_list = l[0: 5].copy()
    for index, items in enumerate(new_list):
        new_list[index] = (items[0], str(round((int(items[1]) /sum) * 100, 3)) + '%')
    return new_list

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
# print(top_five_occurences_unigram[: 5])
# print(top_five_occurences_perceptron[: 5])



# print('Most dominant tags [UNIGRAM_TAGGER]: ', total_tags_sum(top_five_occurences_unigram))
# print('Most dominant tags [PERCEPTRON_TAGGER]: ', total_tags_sum(top_five_occurences_perceptron))