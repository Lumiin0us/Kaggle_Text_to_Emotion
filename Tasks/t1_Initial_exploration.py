import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import Counter
import pandas as pd 
import numpy as np

DATASET = '/Users/abdurrehman/Desktop/Oulu Courses /NLP/kaggle_text_to_emotion/Kaggle_Text_to_Emotion/Dataset/tweet_emotions.csv'

#reading the dataset
df = pd.read_csv(DATASET)

#extracting only the sentiments and converting the dataframe to an array
df_sentiments = df.iloc[:, 1]
sentiments_array = np.array(df_sentiments)
print(sentiments_array)

#creating a dict with keys as emotions and values as their total occurrences
sentiments_dict = Counter(sentiments_array)
sentiments = dict(sentiments_dict)
print(sentiments)

#plotting the histogram
# y_ticks = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
# plt.figure(figsize=(13, 8))
# labels = [f'{key} : {value}' for key, value in sentiments.items()]
# plt.bar(sentiments.keys(), sentiments.values(), color='blue', alpha=0.5, width= 0.4, label=labels)
# plt.title('Frequency distribution of the dataset')
# plt.xlabel('Sentiments')
# plt.ylabel('Frequencies')
# plt.yticks(y_ticks)
# plt.legend()
# plt.show()

plt.figure(figsize=(8, 8))
bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
y_ticks = [1, 2, 3, 4, 5]
plt.hist(sentiments.values(), bins=bins, ec='black', color='blue', alpha=0.5, rwidth=0.4)
plt.title('Frequency distribution of the dataset')
plt.xlabel('Sentiments')
plt.xticks(bins)
plt.ylabel('Frequencies')
plt.yticks(y_ticks)
labels = {k: v for k, v in sorted(sentiments.items(), key=lambda item: item[1])}
labels = [f'{key} : {value}' for key, value in labels.items()]
handles = [Rectangle((0, 0), 1, 1, color='blue', alpha=0.5, ec='blue') for label in labels]
plt.legend(handles, labels)
plt.show()

#Comment on the distribution of the training samples across various categories.
#! The histogram shows that most of the sentiment categories lie between the frequency range of 0-1000 with very few data samples, followed by the next 
#! highest occurences in the range 1000-2000 while the smallest occurences between the ranges 2000-3000 and 3000-4000 (only single categories) 
#! From the histogram we can infer that the data distribution is uneven, most of the labels have very few training samples while some have average number 
#! of samples and some with too many samples. Training ML models on this dataset without addressing the data imbalance will most likely result in a biased model.
