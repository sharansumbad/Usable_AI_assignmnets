from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import string
import random
import nltk
import pandas as pd
import numpy as np

# Opening the file
f = open("amazon_cells_labelled.txt", "r")

data =[]
# Converting it to pandas dataframe
for line in f:
    review = line[:len(line) - 2]
    sentiment = "neg" if line[len(line)-2] == "0" else "pos"
    row = [review, sentiment]
    data.append(row)

df = pd.DataFrame(data, columns = ['reviews', 'sentiment'])
# The below pandas dataframe has reviews and sentiment
print(df)




## Todo: LDA ##
# 3.a.i Clean the data by removing Stop words, punctuations, emoticons, etc
## slides: W12_1_nlp.pptx(19, 20, 23) and W12_2_nlp.pptx(9, 10, 11)



# 3.a.ii Apply LDA with a range of 10 topics
## slides: W13_1_emotion_detection.pptx(25,26,27)





## Todo: Chatbot ##
# Preparing Chatbot data
## Combining all the reviews into a single string.
reviews = ""
for review in df["reviews"]:
    reviews += review
print(reviews)


## Todo: Chatbot ##
# 3.b.i Clean the data by removing Stop words, punctuations, emoticons, etc
## slides: W12_1_nlp.pptx(19, 20, 23) and W12_2_nlp.pptx(9, 10, 11)
# It's completely upto you whether you wanty to re-do the cleaning or use the cleaned data from the 3.a.i



# 3.b.ii
## slides: W13_2_artificial_intelligence.pptx(5,6,7,8)
