import pandas as pd
from nltk.tokenize import word_tokenize
from scipy.stats import pearsonr

#loading both datasets
FILEPATH_NRC = "/Users/abdurrehman/Desktop/Oulu Courses /NLP/kaggle_text_to_emotion/Kaggle_Text_to_Emotion/Dataset/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
nrc_df = pd.read_csv(FILEPATH_NRC,  names=["word", "emotion", "association"], skiprows=45, sep='\t')
total_emotions = nrc_df['emotion'].unique()

FILEPATH_TWEETS = '/Users/abdurrehman/Desktop/Oulu Courses /NLP/kaggle_text_to_emotion/Kaggle_Text_to_Emotion/Dataset/tweet_emotions.csv' 
twitter_df = pd.read_csv(FILEPATH_TWEETS)

#mapping sentiments keys are twitter sentiments and values are nrc sentiments - (categories they may belong to)
sentiment_mapping = {
    "happiness": ['joy', 'positive'],  
    "anger": ['anger', 'disgust',  'negative', 'sadness'],      
    "empty": ['negative'],
    "sadness": ['sadness', 'negative', 'fear'],
    "enthusiasm": ['joy', 'positive', 'anticipation'],
    "neutral": ['positive', 'trust'],
    "worry": ['fear', 'negative'],
    "surprise": ['surprise', 'positive'],
    "love": ['joy', 'trust', 'positive'],
    "fun": ['joy', 'positive'],
    "hate": ['anger', 'disgust', 'negative'],
    "boredom": ['sadness', 'negative'],
    "relief": ['trust', 'positive'],
}
#initializing V1 and V2
vector_1 = {key: [] for key, value in sentiment_mapping.items()}
vector_2 = {key: [] for key, value in sentiment_mapping.items()}

#matching datasets' records and finding correlation/p-value between matched words
def compare_sentiment(sentiment_mapping):
    correlation_table = {}
    for key, value in sentiment_mapping.items():
        m = 0 #matches
        tweets = list(twitter_df.loc[twitter_df['sentiment'] == key]['content'])
        sentiment_list = []
        for i in value:
            sentiment_list.extend(list(nrc_df.loc[nrc_df['emotion'] == i]['word']))

        #lower-casing sentiments list (with lower casing the matching ratio went from 84% to 89.9%)
        sentiment_list = [sentiment.lower() for sentiment in sentiment_list if type(sentiment) == str]

        for tweet in tweets:
            tweet = word_tokenize(tweet)

            #lower casing tweets
            tweet = [words.lower() for words in tweet]
            match = set(set(sentiment_list)).intersection(set(tweet))
            if len(match) > 0:
                vector_1[key].append(1)
                vector_2[key].append(len(match))
                m += 1
            else:
                vector_1[key].append(0)
                vector_2[key].append(len(match))

        print('Sentiment', key,'Total matches:', m, 'Out of:', len(tweets))
        correlation_coefficient, p_value = pearsonr(vector_1[key], vector_2[key])
        correlation_table[key] = [correlation_coefficient, p_value]
    return correlation_table

pearson_correlation_table = compare_sentiment(sentiment_mapping=sentiment_mapping)

df_table = pd.DataFrame(pearson_correlation_table, index=['Pearson Correlation', 'P-Value'])
print(df_table)

# Analysis 
# Very low P-values for all the sentiments indicate that none of the correlation is random instead all are statistically significant. Additionally, positive
# pearson correlation values show that the vectors have a positive linear relationship.

# notes
# correlation ranges from [-1, 1] => 1 represents a perfect positive linear correlation, -1 represents a perfect negative linear correlation and 0 represents
# no correlation 
# low P-value (< 0.05) represents that the relation between the vectors is statistically signifcant and not through random chances 
# high P-valu(> 0.05) there is a chance that the relationship could be random
