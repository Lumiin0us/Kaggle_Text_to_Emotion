import pandas as pd
from collections import Counter
from gensim.models import Word2Vec
import numpy as np
from numpy.linalg import norm

# File path to the dataset
FILEPATH_TWEETS = '/Users/abdurrehman/Desktop/Oulu Courses /NLP/kaggle_text_to_emotion/Kaggle_Text_to_Emotion/Dataset/tweet_emotions.csv'
twitter_df = pd.read_csv(FILEPATH_TWEETS)

def get_sentiments_and_calculate_similarities(twitter_df):
    # Get unique sentiments and filter those with fewer than 1500 records - setting 1500 samples as the max thresold for being considered as a small category
    all_sentiments = list(twitter_df['sentiment'].unique())
    small_categories = [sentiment for sentiment in all_sentiments if len(twitter_df[twitter_df['sentiment'] == sentiment]) < 1500]

    cosine_similarities = calculate_similarities(all_sentiments)
    print(cosine_similarities)
    display_similarity_table(cosine_similarities)

    # Merge small categories based on calculated cosine_similarities
    merge_small_categories(twitter_df, all_sentiments, small_categories, cosine_similarities)

def calculate_similarities(all_sentiments):
    cosine_similarities = {}
    sentiment_list = all_sentiments.copy()
    all_sentiments = [[sentiment] for sentiment in all_sentiments]

    word2vec_model = Word2Vec(all_sentiments, min_count=1)
    word2vec_model.train(all_sentiments, total_examples=len(all_sentiments), epochs=10)

    for sentiment_index in range(len(sentiment_list)):
        for iter in range(sentiment_index + 1, len(sentiment_list)):
            cosine_similarities[(sentiment_list[sentiment_index], sentiment_list[iter])] = np.dot(word2vec_model.wv[sentiment_list[sentiment_index]], word2vec_model.wv[sentiment_list[iter]])/(norm(word2vec_model.wv[sentiment_list[sentiment_index]]) * norm(word2vec_model.wv[sentiment_list[iter]]))
    # print(word2vec_model.wv['hate'])
    return cosine_similarities

def display_similarity_table(cosine_similarities):
    df_table = pd.DataFrame(cosine_similarities, index=['Cosine Similarities'])
    print(df_table)

def merge_small_categories(twitter_df, all_sentiments, small_categories, cosine_similarities):
    merging_threshold = 0.01

    merge_categories_matched = []
    output_dict = {}

    for key, value in cosine_similarities.items():
        if key[0] in small_categories and key[1] in small_categories and value >= merging_threshold:
            merge_categories_matched.extend(key)
            output_dict[key] = value
    counter = Counter(merge_categories_matched)
    max_count = max(counter.values())

    keys_with_max_count = [key for key, count in counter.items() if count == max_count]
    checker = {}
    other_categories = {}
    for key in keys_with_max_count:
        for k, v in output_dict.items():
            if key in k[0] or key in k[1]:
                if key in checker.keys():
                    checker[key] += v
                else:
                    checker[key] = v
                if key in k[0] and key in other_categories.keys():
                    other_categories[key].append(k[1])
                elif key in k[0] and key not in other_categories.keys():
                    other_categories[key] = [k[1]]
                elif key in k[1] and key not in other_categories.keys():
                    other_categories[key] = [k[0]]
                elif key in k[1] and key in other_categories.keys():
                    other_categories[key].append(k[0])

    if len(keys_with_max_count) > 1:
        checker = sorted(checker.items(), key=lambda x:x[1], reverse=True)
        checker = {checker.keys[0] : checker.values[0]}    

    new_sentiment_df = pd.DataFrame(columns=[sentiment for sentiment in all_sentiments])
    new_sentiment_df = pd.concat([new_sentiment_df, twitter_df], ignore_index=True)

    for sentiment in all_sentiments:
        tweets = twitter_df.loc[twitter_df['sentiment'] == sentiment]['content']
        new_sentiment_df[sentiment] = pd.Series(list(tweets.values) + ([''] * (len(new_sentiment_df) - len(tweets))))
        new_sentiment_df.drop(['sentiment', 'content', 'tweet_id'], axis=1, errors='ignore', inplace=True)

    for col_index in range(len(list(other_categories.values())[0])):
        pd.concat([new_sentiment_df[list(other_categories.keys())[0]], new_sentiment_df[list(other_categories.values())[0][col_index]]], axis=1, ignore_index=True)
        new_sentiment_df.drop([list(other_categories.values())[0][col_index]], axis=1, inplace=True)

    col_name = list(other_categories.keys())[0]
    
    for labels in list(other_categories.values())[0]:
        col_name += '_' + labels
    new_sentiment_df = new_sentiment_df.rename(columns={list(other_categories.keys())[0]: col_name})
    
    print(new_sentiment_df.head())
    
get_sentiments_and_calculate_similarities(twitter_df)
