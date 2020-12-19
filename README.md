# Sentiment Analysis using TF-IDF Vectorization

This repository hosts my work on the classic work of Sentiment Analysis using **Term Frequency-Inverse Document Frequency** (TF-IDF) method.

TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. 

This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents.

TF-IDF was invented for document search and information retrieval. It works by increasing proportionally to the number of times a word appears in a document, but is offset by the number of documents that contain the word. So, words that are common in every document, such as this, what, and if, rank low even though they may appear many times, since they don’t mean much to that document in particular.


![TF-IDF](__pycache__\tf-idf.png)
