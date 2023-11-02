import numpy as np 
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from numpy.linalg import norm
from itertools import combinations

glove_vectors = {}

glove_file = open("/Users/abdurrehman/Desktop/Oulu Courses /NLP/kaggle_text_to_emotion/Kaggle_Text_to_Emotion/util/glove.twitter.27B/glove.twitter.27B.100d.txt", encoding='utf-8')
for line in glove_file:
    values = line.split()
    word = values[0]
    coeff = np.asarray(values[1:], dtype = 'float32')
    glove_vectors[word] = coeff


#Loading dataset
FILEPATH_TWEETS = "Dataset/tweet_emotions.csv"
twitter_df = pd.read_csv(FILEPATH_TWEETS)

#unique sentiments in our dataset
all_sentiments = twitter_df['sentiment'].unique()

#calculating word2vec embeddings for each record or tweet of our sentiment(cat_1, cat_2) and storing them in "sentiment_vector_map"
def calculate_word_embeddings_for_each_record(cat_1, cat_2, glove_vectors):
    all_sentiments = [cat_1, cat_2]
    sentiment_vector_map = defaultdict(list)
    stopwords_list = stopwords.words('english')
    
    for sentiment in all_sentiments:
        tweet_list = list(twitter_df.loc[twitter_df['sentiment'] == sentiment]['content'])
        for index, tweet in enumerate(tweet_list):
            tweet = word_tokenize(tweet)
            tweet = [word.lower() for word in tweet if word.lower() not in stopwords_list]
            tweet_list[index] = tweet
            
        vector_sum = np.zeros(100)  # Initialize a vector of zeros
        word_count = 0

        for tweet in tweet_list:
            for word in tweet:
                if word in glove_vectors:
                    # vector_sum += glove_vectors[word].reshape(1, -1)
                    vector_sum += glove_vectors[word]
                    word_count += 1

        if word_count > 0:
            sentiment_vector_map[sentiment].append(vector_sum / word_count)
        else:
            sentiment_vector_map[sentiment].append(np.zeros(100))  # Use zeros if no valid word vectors are found
            # sentiment_vector_map[sentiment].append(np.zeros((1, 100)))

    return sentiment_vector_map


def calculate_cosine_similarities(word_embeddings_dict, cat_1, cat_2):
    similarities = []  # Initialize an empty list to store all similarities
    all_sentiments = [cat_1, cat_2]
    # print(word_embeddings_dict[cat_1])
    # cosine_similarity = np.dot(word_embeddings_dict[cat_1], word_embeddings_dict[cat_2]) / (norm(word_embeddings_dict[cat_1]) * norm(word_embeddings_dict[cat_2]))
    # cosine_similarity = np.dot(word_embeddings_dict[cat_1], word_embeddings_dict[cat_2].T) / (norm(word_embeddings_dict[cat_1]) * norm(word_embeddings_dict[cat_2]))
    

    # return (cat_1, cat_2, cosine_similarity)
    # return []
    vector_sum_list = []
    
    for key, vectors in word_embeddings_dict.items():
        vector_sum = np.zeros(100)
        for vector in vectors:
            vector_sum += vector
        vector_sum /= len(vectors)
        vector_sum_list.append(vector_sum)
    cosine_similarity = np.dot(vector_sum_list[0], vector_sum_list[1]) / (norm(vector_sum_list[0]) * norm(vector_sum_list[1]))
    return [(cat_1, cat_2, cosine_similarity)]
    

    # for index_of_sentiment in range(len(vector_sum_list)):
    #     for j in range(len(vector_sum_list)):
    #         cosine_similarity = np.dot(vector_sum_list[index_of_sentiment], list(list_embeddings_dict.values())[j][0]) / (norm(vector_sum_list[index_of_sentiment]) * norm(list(list_embeddings_dict.values())[j][0]))
    #     # similarities.append((all_sentiments[index_of_sentiment], cosine_similarity))
    #         # print((all_sentiments[index_of_sentiment], all_sentiments[j], cosine_similarity))
    #         similarities.append((all_sentiments[index_of_sentiment], all_sentiments[j], cosine_similarity))
    # return similarities


#displaying the cosine-similarties between each category
def record_based_similarity_between_each_pair(all_sentiments, glove_vectors):
    similarity_dict = {}
    for pairs in combinations(all_sentiments, 2):
        word_embeddings_dict = calculate_word_embeddings_for_each_record(pairs[0], pairs[1], glove_vectors= glove_vectors)
        similarities = calculate_cosine_similarities(word_embeddings_dict=word_embeddings_dict, cat_1=pairs[0], cat_2=pairs[1])
        print(similarities)

        for label_1, label_2, similarity in similarities:
            if label_1 not in similarity_dict:
                similarity_dict[label_1] = {}
            similarity_dict[label_1][label_2] = similarity
            if label_2 not in similarity_dict:
                similarity_dict[label_2] = {}
            similarity_dict[label_2][label_1] = similarity
    similarity_df = pd.DataFrame(similarity_dict)
    similarity_df.fillna(1.0, inplace=True)

    print(similarity_df)
record_based_similarity_between_each_pair(all_sentiments=all_sentiments, glove_vectors=glove_vectors)

