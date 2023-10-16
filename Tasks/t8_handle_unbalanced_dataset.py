import pandas as pd
from nltk.corpus import wordnet as wn
from itertools import combinations
from collections import Counter

# File path to the dataset
FILEPATH_TWEETS = '/Users/abdurrehman/Desktop/Oulu Courses /NLP/kaggle_text_to_emotion/Kaggle_Text_to_Emotion/Dataset/tweet_emotions.csv'
twitter_df = pd.read_csv(FILEPATH_TWEETS)

def get_sentiments_and_calculate_similarities(twitter_df):
    # Get unique sentiments and filter those with fewer than 1500 records - setting 1500 samples as the max thresold for being considered as a small category
    all_sentiments = list(twitter_df['sentiment'].unique())
    small_categories = [sentiment for sentiment in all_sentiments if len(twitter_df[twitter_df['sentiment'] == sentiment]) < 1500]

    similarities = calculate_similarities(all_sentiments)
    display_similarity_table(similarities)

    # Merge small categories based on calculated similarities
    merge_small_categories(twitter_df, all_sentiments, small_categories, similarities)

def calculate_similarities(all_sentiments):
    similarities = {}
    for pair in combinations(all_sentiments, 2):
        synset_1 = wn.synsets(pair[0])
        synset_2 = wn.synsets(pair[1])
        for lemma_1 in synset_1:
            max_similarity_score = 0
            mean_score = 0
            for lemma_2 in synset_2:
                similarity_score = lemma_1.wup_similarity(lemma_2)
                mean_score += similarity_score
                if max_similarity_score < similarity_score:
                    max_similarity_score = similarity_score
                    similarities[pair] = [lemma_1, lemma_2, round(max_similarity_score, 3), round((mean_score)/(len(synset_1) * len(synset_2)), 3)]
    return similarities

def display_similarity_table(similarities):
    df_table = pd.DataFrame(similarities, index=['Best Matching Synsets', 'Best Matching Synsets', 'Maximum Similarity', 'Mean Similarity'])
    print(df_table)

def merge_small_categories(twitter_df, all_sentiments, small_categories, similarities):
    merging_threshold = 0.4

    merge_categories_matched = []
    merge_mean = []

    for key, value in similarities.items():
        if key[0] in small_categories and key[1] in small_categories and value[-2] >= merging_threshold:
            merge_categories_matched.extend(key)
            merge_mean.append(value[-1])

    counter = Counter(merge_categories_matched)
    max_count = max(counter.values())
    labels_with_sorted_average_similarities = {}
    keys_with_max_count = [key for key, count in counter.items() if count == max_count]

    for values in keys_with_max_count:
        total_count = 0
        for key, val in similarities.items():
            if key[0] in small_categories and key[1] in small_categories and val[-2] >= merging_threshold and values in key:
                total_count += val[-1]
        labels_with_sorted_average_similarities[values] = total_count

    merging_cat = max(labels_with_sorted_average_similarities, key=labels_with_sorted_average_similarities.get)

    labels_to_merged_with_merging_cat = []

    for key, val in similarities.items():
        if key[0] in small_categories and key[1] in small_categories and val[-2] >= merging_threshold and merging_cat in key:
            labels_to_merged_with_merging_cat.extend(key)

    labels_to_merged_with_merging_cat = list(set(labels_to_merged_with_merging_cat))
    labels_to_merged_with_merging_cat.remove(merging_cat)

    new_sentiment_df = pd.DataFrame(columns=[sentiment for sentiment in all_sentiments])
    new_sentiment_df = pd.concat([new_sentiment_df, twitter_df], ignore_index=True)

    for sentiment in all_sentiments:
        tweets = twitter_df.loc[twitter_df['sentiment'] == sentiment]['content']
        new_sentiment_df[sentiment] = pd.Series(list(tweets.values) + ([''] * (len(new_sentiment_df) - len(tweets))))
        new_sentiment_df.drop(['sentiment', 'content', 'tweet_id'], axis=1, errors='ignore', inplace=True)

    for col_index in range(len(labels_to_merged_with_merging_cat)):
        pd.concat([new_sentiment_df[merging_cat], new_sentiment_df[labels_to_merged_with_merging_cat[col_index]]], axis=1, ignore_index=True)
        new_sentiment_df.drop([labels_to_merged_with_merging_cat[col_index]], axis=1, inplace=True)

    col_name = merging_cat
    
    for labels in labels_to_merged_with_merging_cat:
        col_name += '_' + labels
    new_sentiment_df = new_sentiment_df.rename(columns={merging_cat: col_name})
    
    print(new_sentiment_df.head())
    
get_sentiments_and_calculate_similarities(twitter_df)
