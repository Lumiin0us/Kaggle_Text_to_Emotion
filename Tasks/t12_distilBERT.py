from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
import torch
import numpy as np
from numpy.linalg import norm
from itertools import combinations

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Loading dataset
FILEPATH_TWEETS = "Dataset/tweet_emotions.csv"
twitter_df = pd.read_csv(FILEPATH_TWEETS)

# Unique sentiments in our dataset
all_sentiments = twitter_df['sentiment'].unique()

# Calculate distilBERT embeddings for each record or tweet of our sentiment (cat_1, cat_2) and store them in "sentiment_vector_map"
def calculate_distilbert_embeddings_for_each_record(cat_1, cat_2):
    sentiment_vector_map = {}

    for sentiment in [cat_1, cat_2]:
        tweet_list = list(twitter_df.loc[twitter_df['sentiment'] == sentiment]['content'])
        embeddings = []
        if len(tweet_list) > 500:
            tweet_list = tweet_list[:100]
        for tweet in tweet_list:
            inputs = tokenizer(tweet, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = distilbert_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy())
        sentiment_vector_map[sentiment] = embeddings
    return sentiment_vector_map

def calculate_distilbert_embeddings_for_overall_list(cat_1, cat_2):
    sentiment_vector_map = {}
    all_sentiments = [cat_1, cat_2]

    for sentiment in all_sentiments:
        tweet_list = list(twitter_df.loc[twitter_df['sentiment'] == sentiment]['content'])
        
        if len(tweet_list) > 100:
            tweet_list = tweet_list[:100] 

        # Tokenize the entire list of tweets
        inputs = tokenizer(tweet_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        sentiment_vector_map[sentiment] = embeddings
    return sentiment_vector_map, cat_1, cat_2

# Calculate cosine similarities between BERT embeddings
def calculate_cosine_similarities(bert_embeddings_words, bert_embeddings_list, cat_1, cat_2):
    similarities = []
    all_sentiments = [cat_1, cat_2]

    for sentiment in all_sentiments:
        embeddings = bert_embeddings_words[sentiment]
        mean_embedding = np.mean(embeddings, axis=0)

        for other_sentiment in all_sentiments:
            other_embeddings = bert_embeddings_list[other_sentiment]
            other_mean_embedding = np.mean(other_embeddings, axis=0)

            cosine_similarity = np.dot(mean_embedding, other_mean_embedding) / (norm(mean_embedding) * norm(other_mean_embedding))
            similarities.append((sentiment, other_sentiment, cosine_similarity))

    return similarities

# Display the cosine similarities between each category
def record_based_similarity_between_each_pair(all_sentiments):
    similarity_dict = {}
    for pairs in combinations(all_sentiments, 2):
        bert_embeddings_words = calculate_distilbert_embeddings_for_each_record(pairs[0], pairs[1])
        bert_embeddings_list, cat_1, cat_2 = calculate_distilbert_embeddings_for_overall_list(pairs[0], pairs[1])

        similarities = calculate_cosine_similarities(bert_embeddings_words, bert_embeddings_list, cat_1, cat_2)

        for label_1, label_2, similarity in similarities:
            if label_1 not in similarity_dict:
                similarity_dict[label_1] = {}
            similarity_dict[label_1][label_2] = similarity
        print(similarity_dict)
    similarity_df = pd.DataFrame(similarity_dict)
    print(similarity_df)

record_based_similarity_between_each_pair(all_sentiments=all_sentiments)
