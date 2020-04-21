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
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from pandas import DataFrame

def get_error_type(pred, label):
    # return the type of error: tp,fp,tn,fn
    return "fp"

#need to do some check of row review vs inqtabs_dict and swn_dict
def classify(text, inqtabs_dict, swn_dict):
      print(type(text))
      text = [text]
      print(type(text))
      df_inq = pd.DataFrame(inqtabs_dict.items(), columns = ['Words', 'Rating'])
      X_train, X_test, y_train, y_test = train_test_split(df_inq['Words'].values, df_inq['Rating'].values, test_size=0.2)
      #vect = CountVectorizer(stop_words='english')
      #tf_train = vect.fit_transform(X_train)
      #tf_test = vect.transform(text)
      #print(tf_test)
      #nb = MultinomialNB()
      #nb.fit(tf_train, y_train)
      #predictions = nb.predict(tf_test).astype(int)
      #print('Fitted Naive Bayes')
##      model = LogisticRegression()
##      model.fit(tf_train, y_train)
##      predictions = model.predict(tf_test).astype(int)
      #print('Fitted Logistic Regression')
      vect = TfidfVectorizer()
      tfidf_train = vect.fit_transform(X_train)
      tfidf_test = vect.transform(text)
      model = LogisticRegression()
      model.fit(tfidf_train, y_train)
      predictions = model.predict(tfidf_test).astype(int)
      return predictions
      
