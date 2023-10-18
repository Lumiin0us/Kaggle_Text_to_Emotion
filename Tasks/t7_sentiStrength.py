import pandas as pd
from sentistrength import PySentiStr
import json 

FILEPATH_TWEETS = '/Users/abdurrehman/Desktop/Oulu Courses /NLP/kaggle_text_to_emotion/Kaggle_Text_to_Emotion/Dataset/tweet_emotions.csv' 
twitter_df = pd.read_csv(FILEPATH_TWEETS)
sentiment_label_list = list(twitter_df['sentiment'].unique())

sentiment_label_dict = {}

senti = PySentiStr()
senti.setSentiStrengthPath('/Users/abdurrehman/Desktop/Oulu Courses /NLP/kaggle_text_to_emotion/Kaggle_Text_to_Emotion/util/SentiStrength.jar')
senti.setSentiStrengthLanguageFolderPath('/Users/abdurrehman/Desktop/Oulu Courses /NLP/kaggle_text_to_emotion/Kaggle_Text_to_Emotion/util/SentiStrength_Data')

for sentiment in sentiment_label_list:
    sentiment_label_dict[sentiment] = senti.getSentiment(sentiment, score='trinary')
    print('Polarity of', sentiment, ' :', senti.getSentiment(sentiment, score='trinary'))
print(sentiment_label_dict)
def file_storage(sentiStrength_sentiments_list):
    with open('Tasks/t7_sentiStrength.json', 'w') as json_file:
        json_object = json.dumps(sentiStrength_sentiments_list)
        json_file.write(json_object)

def sentiment_for_each_tweet(sentiment_label_list):
    sentiStrength_sentiments = {key : [] for key in sentiment_label_list}
    for sentiment in sentiment_label_list:
        tweets = twitter_df.loc[twitter_df['sentiment'] == sentiment]['content']
        for tweet in tweets:
            sentiStrength_sentiments[sentiment].append(senti.getSentiment(tweet, score='trinary'))
            print(tweet)
        print(sentiStrength_sentiments)
    file_storage(sentiStrength_sentiments)

# sentiment_for_each_tweet(sentiment_label_list=sentiment_label_list)

FILE_PATH = './Tasks/t7_sentiStrength.json'

def loading_json(FILE_PATH):
    with open(FILE_PATH, 'r') as json_file:
        json_data = json.load(json_file)
        return json_data

sentiStrength_sentiments = loading_json(FILE_PATH=FILE_PATH)

def sentiment_matching(sentiStrength_sentiments, sentiment_label_dict): 
    sentiStrength_score_table = {key : 0 for key in list(twitter_df['sentiment'])}
    for label, sentiment_probability in sentiment_label_dict.items():
        print(label, sentiment_probability)
        correctly_labeled = 0 
        for lists in sentiStrength_sentiments[label]:
            if lists[0][2] < 0 and sentiment_probability[0][2] < 0: 
                correctly_labeled += 1
            elif lists[0][2] == 0 and sentiment_probability[0][2] == 0: 
                correctly_labeled += 1
            elif lists[0][2] > 0 and sentiment_probability[0][2] > 0: 
                correctly_labeled += 1
        sentiStrength_score_table[label] = [correctly_labeled, len(sentiStrength_sentiments[label]) - correctly_labeled,  len(sentiStrength_sentiments[label])]
    return sentiStrength_score_table
        
sentiStrength_score_table = sentiment_matching(sentiStrength_sentiments, sentiment_label_dict)

df_sentiStrength_score_table = pd.DataFrame(sentiStrength_score_table, index=['Correctly Labeled', 'Incorrectly Labeled', 'Total Samples'])
print(df_sentiStrength_score_table)