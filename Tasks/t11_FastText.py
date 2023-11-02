import pandas as pd
from gensim.models import FastText
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from itertools import combinations

#Loading dataset
FILEPATH_TWEETS = "Dataset/tweet_emotions.csv"
twitter_df = pd.read_csv(FILEPATH_TWEETS)

#unique sentiments in our dataset
all_sentiments = twitter_df['sentiment'].unique()

#calculating word2vec embeddings for each record or tweet of our sentiment(cat_1, cat_2) and storing them in "sentiment_vector_map"
def calculate_word_embeddings_for_each_record(cat_1, cat_2):
    all_sentiments = []
    all_sentiments.append(cat_1)
    all_sentiments.append(cat_2)
    sentiment_vector_map = defaultdict(lambda: [])
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
                fasttext_model = FastText(sentences=[tweet], min_count=1)
                for word in tweet:
                    vector = fasttext_model.wv[word]
                    wordlist_length += 1
                    vector_sum += vector
                sentiment_vector_map[sentiment].append((vector_sum) / wordlist_length)

    return sentiment_vector_map

#calculating word2vec embeddings for the overall tweet_list of our sentiment(cat_1, cat_2) and storing them in "sentiment_vector_map"
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
        fasttext_model = FastText(sentences=tweet_list, min_count=1)
        for tweet in tweet_list:
            for word in tweet:
                vector_sum += fasttext_model.wv[word]
        sentiment_vector_map[sentiment].append((vector_sum) / tweetlist_length)
        sentiment_map[sentiment] = tweet_list

    return sentiment_vector_map, cat_1, cat_2


def calculate_cosine_similarities(word_embeddings_dict, list_embeddings_dict, cat_1, cat_2):
    similarities = []  # Initialize an empty list to store all similarities
    all_sentiments = [cat_1, cat_2]

    vector_sum_list = []
    
    for key, vectors in word_embeddings_dict.items():
        vector_sum = np.zeros(100)
        for vector in vectors:
            vector_sum += vector
        vector_sum /= len(vectors)
        vector_sum_list.append(vector_sum)

    for index_of_sentiment in range(len(vector_sum_list)):
        for j in range(len(vector_sum_list)):
            cosine_similarity = np.dot(vector_sum_list[index_of_sentiment], list(list_embeddings_dict.values())[j][0]) / (norm(vector_sum_list[index_of_sentiment]) * norm(list(list_embeddings_dict.values())[j][0]))
        # similarities.append((all_sentiments[index_of_sentiment], cosine_similarity))
            # print((all_sentiments[index_of_sentiment], all_sentiments[j], cosine_similarity))
            similarities.append((all_sentiments[index_of_sentiment], all_sentiments[j], cosine_similarity))
    return similarities


#displaying the cosine-similarties between each category
def record_based_similarity_between_each_pair(all_sentiments):
    similarity_dict = {}
    for pairs in combinations(all_sentiments, 2):
        list_embeddings_dict, cat_1, cat_2 = calculate_word_embeddings_for_overall_list(pairs[0], pairs[1])
        word_embeddings_dict = calculate_word_embeddings_for_each_record(pairs[0], pairs[1])
        similarities = calculate_cosine_similarities(list_embeddings_dict=list_embeddings_dict, word_embeddings_dict=word_embeddings_dict, cat_1=cat_1, cat_2=cat_2)
        # print(similarities)

        for label_1, label_2, similarity in similarities:
            if label_1 not in similarity_dict:
                similarity_dict[label_1] = {}
            similarity_dict[label_1][label_2] = similarity
    similarity_df = pd.DataFrame(similarity_dict)
    print(similarity_df)
record_based_similarity_between_each_pair(all_sentiments=all_sentiments)

