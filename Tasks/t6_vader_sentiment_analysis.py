import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json 

#reading dataset
FILEPATH_TWEETS = '/Users/abdurrehman/Desktop/Oulu Courses /NLP/kaggle_text_to_emotion/Kaggle_Text_to_Emotion/Dataset/tweet_emotions.csv' 
twitter_df = pd.read_csv(FILEPATH_TWEETS)

sentiment_label_list = list(twitter_df['sentiment'].unique())
sentiment_label_dict = {}

#initializing vader analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

#checking the sentiment of each label/category and assigning them a score
for sentiment in sentiment_label_list:
    sentiment_label_dict[sentiment] = sentiment_analyzer.polarity_scores(sentiment)['compound']
    # print('Polarity of', sentiment, ' :', sentiment_analyzer.polarity_scores(sentiment))
print(sentiment_label_dict)

#creating json_file and storing tweet-vader-analysis
def file_storage(vader_sentiments_list):
    with open('Tasks/t6_vader.json', 'w') as json_file:
        json_object = json.dumps(vader_sentiments_list)
        json_file.write(json_object)

#checking sentiments for each tweet and assigning them a score using vader
def sentiment_for_each_tweet(sentiment_label_list):
    vader_sentiments = {key : [] for key in sentiment_label_list}
    print(vader_sentiments)
    for sentiment in sentiment_label_list:
        tweets = twitter_df.loc[twitter_df['sentiment'] == sentiment]['content']
        for tweet in tweets:
            vader_sentiments[sentiment].append(sentiment_analyzer.polarity_scores(tweet))
    file_storage(vader_sentiments)

### commented this function cause it analyzes sentiment for each tweet, since we already have the analysis stored in json file so dont need this. ###
# sentiment_for_each_tweet(sentiment_label_list=sentiment_label_list)


FILE_PATH = './Tasks/t6_vader.json'

#json file loading function
def loading_json(FILE_PATH):
    with open(FILE_PATH, 'r') as json_file:
        json_data = json.load(json_file)
        return json_data

#loading the json file
vader_sentiments = loading_json(FILE_PATH=FILE_PATH)

#matching sentiments between tweets and label categories that they belong to
def sentiment_matching(vader_sentiments, sentiment_label_dict): 
    vader_score_table = {key : 0 for key in list(twitter_df['sentiment'])}
    for label, sentiment_probability in sentiment_label_dict.items():
        correctly_labeled = 0 
        for lists in vader_sentiments[label]:
            if lists['compound'] < 0 and sentiment_probability < 0: 
                correctly_labeled += 1
            elif lists['compound'] == 0 and sentiment_probability == 0: 
                correctly_labeled += 1
            elif lists['compound'] > 0 and sentiment_probability > 0: 
                correctly_labeled += 1
        vader_score_table[label] = [correctly_labeled, len(vader_sentiments[label]) - correctly_labeled,  len(vader_sentiments[label])]
    return vader_score_table

vader_score_table = sentiment_matching(vader_sentiments, sentiment_label_dict)

#displaying the results in a table
df_vader_score_table = pd.DataFrame(vader_score_table, index=['Correctly Labeled', 'Incorrectly Labeled', 'Total Samples'])
print(df_vader_score_table)