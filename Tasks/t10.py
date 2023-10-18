import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from itertools import combinations
# File path to the dataset
FILEPATH_TWEETS = '/Users/abdurrehman/Desktop/Oulu Courses /NLP/kaggle_text_to_emotion/Kaggle_Text_to_Emotion/Dataset/tweet_emotions.csv'
twitter_df = pd.read_csv(FILEPATH_TWEETS)

all_sentiments = twitter_df['sentiment'].unique()

def calculate_word_embeddings_for_each_record(cat_1, cat_2):
    all_sentiments = []
    all_sentiments.append(cat_1)
    all_sentiments.append(cat_2)
    sentiment_vector_map = defaultdict(lambda:[])
    stopwords_list = stopwords.words('english')
    for sentiment in all_sentiments:
        tweet_list = list(twitter_df.loc[twitter_df['sentiment'] == sentiment]['content'])
        for index, tweet in enumerate(tweet_list):
            tweet = word_tokenize(tweet)
            tweet = [words.lower() for words in tweet if words.lower() not in stopwords_list]
            tweet_list[index] = tweet
            if tweet:
                vector_sum = 0.0
                wordlist_length = 0
                word2vec_model = Word2Vec([tweet], min_count=1)
                for word in tweet:
                    vector = word2vec_model.wv[word]
                    wordlist_length += 1
                    vector_sum += vector
                sentiment_vector_map[sentiment].append((vector_sum)/wordlist_length)

    return sentiment_vector_map


def calculate_word_embeddings_for_overall_list(cat_1, cat_2):
    all_sentiments = []
    all_sentiments.append(cat_1)
    all_sentiments.append(cat_2)
    sentiment_map = {}
    sentiment_vector_map = defaultdict(list)
    stopwords_list = stopwords.words('english')
    for sentiment in all_sentiments: 
        tweet_list = list(twitter_df.loc[twitter_df['sentiment'] == sentiment]['content'])
        for index, tweet in enumerate(tweet_list):
            tweet = word_tokenize(tweet)
            tweet = [words.lower() for words in tweet if words.lower() not in stopwords_list]
            tweet_list[index] = tweet
            
        vector_sum = 0.0
        tweetlist_length = len(tweet_list)
        word2vec_model = Word2Vec(tweet_list, min_count=1)
        for tweet in tweet_list:
            for word in tweet: 
                vector_sum += word2vec_model.wv[word]
        sentiment_vector_map[sentiment].append((vector_sum)/tweetlist_length)
        sentiment_map[sentiment] = tweet_list

    return sentiment_vector_map, cat_1, cat_2

def calculate_cosine_similarities(word_embeddings_dict, list_embeddings_dict, cat_1, cat_2):
    all_sentiments = []
    all_sentiments.append(cat_1)
    all_sentiments.append(cat_2)
    similarities = []
    vector_sum = np.zeros(100)
    vector_sum_list = []
    for key, vectors in word_embeddings_dict.items():
        for vector in vectors:
            vector_sum += vector
        vector_sum /= len(vectors)
        vector_sum_list.append(vector_sum)

    for index_of_sentiment in range(len(vector_sum_list)):
        cosine_similarity = np.dot(vector_sum_list[index_of_sentiment], list(list_embeddings_dict.values())[index_of_sentiment][0])/(norm(vector_sum_list[index_of_sentiment]) * norm(list(list_embeddings_dict.values())[index_of_sentiment][0]))
        similarities.append((all_sentiments[index_of_sentiment], cosine_similarity))

    return similarities
# list_embeddings_dict, cat_1, cat_2 = calculate_word_embeddings_for_overall_list('empty', 'sadness')
# word_embeddings_dict = calculate_word_embeddings_for_each_record('empty', 'sadness')

# similarities = calculate_cosine_similarities(list_embeddings_dict=list_embeddings_dict, word_embeddings_dict=word_embeddings_dict, cat_1=cat_1, cat_2=cat_2)
# print(similarities)

def record_based_similarity_between_each_pair(all_sentiments):
    for pairs in combinations(all_sentiments, 2):
        list_embeddings_dict, cat_1, cat_2 = calculate_word_embeddings_for_overall_list(pairs[0], pairs[1])
        word_embeddings_dict = calculate_word_embeddings_for_each_record(pairs[0], pairs[1])
        similarities = calculate_cosine_similarities(list_embeddings_dict=list_embeddings_dict, word_embeddings_dict=word_embeddings_dict, cat_1=cat_1, cat_2=cat_2)
        print(similarities)

record_based_similarity_between_each_pair(all_sentiments=all_sentiments)
