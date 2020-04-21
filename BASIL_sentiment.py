import re
import sklearn
import nltk
import numpy as np
import pandas as pd
import tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from pandas import DataFrame

def get_error_type(pred, label):
    # return the type of error: tp,fp,tn,fn
    return "fp"

#need to do some check of row review vs inqtabs_dict and swn_dict
def classify(text, inqtabs_dict, swn_dict):
      text = [text]
      #print(text)
      training_words = []
      training_sentiment = []
      df = pd.DataFrame(inqtabs_dict.items(), columns = ['Words', 'Rating'])
      X_train, X_test, y_train, y_test = train_test_split(df['Words'].values, df['Rating'].values, test_size=0.2)
      vect = CountVectorizer(stop_words='english')
      tf_train = vect.fit_transform(X_train)
      #print(tf_train)
      #tf_test = vect.transform(X_test)
      #print(tf_test)
      #tf_textset = vect.transform(r for r in text)
      #print(tf_textset)
      nb = MultinomialNB()
      nb.fit(tf_train, y_train)
      predictions = nb.predict(r for r in text)
      #print(type(predictions))
      #model = LogisticRegression()
      #model.fit(tf_train, y_train)
      #predictions = model.predict(tf_textset)
      return predictions
