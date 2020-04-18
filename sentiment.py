import re
import sklearn
import nltk
import numpy as np
import tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize

def get_error_type(pred, label):
    # return the type of error: tp,fp,tn,fn
    return "fp"

#need to do some check of row review vs inqtabs_dict and swn_dict
def classify(text, inqtabs_dict, swn_dict):
      #print(text)
      #print(inqtabs_dict)
      #text = [text]
      training_words = []
      training_sentiment = []
      for i in inqtabs_dict:
        #print(i, inqtabs_dict[i])
        training_words.append(i)
        training_sentiment.append(inqtabs_dict[i])
      #print(training_sentiment)
      X_train, X_test, y_train, y_test = train_test_split(training_words, training_sentiment, test_size=0.2)
      #re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
      #def tokenize(s): return re_tok.sub(r' \1 ', s).split()
      vect = CountVectorizer(tokenizer=tokenize)
      tf_train = vect.fit_transform(X_train)
      tf_test = vect.transform(X_test)
      #tf_train
      model = LogisticRegression()
      model.fit(tf_train, y_train)
      preds = model.predict(text)
      #print(preds)
      return preds
