import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()


def split_tokenizer(s):
    return s.split(" ")


def stem(word, except_words=None):
    if except_words and (word in except_words):
        return word
    else:
        return(stemmer.stem(word))


def stem_tokenizer(s):
    return [stem(word) for word in s.split(" ")]


def lemmatize(word, except_words=set()):
    if word in except_words:
        return word
    else:
        return(lemmatizer.lemmatize(word))


def lemmatize_tokenizer(s):
    return [lemmatize(word) for word in s.split(" ")]


class Vectorizer:
    def __init__(self, recipients):
        self.input_bow = CountVectorizer(tokenizer=stem_tokenizer, min_df=5)
        self.output_bow = CountVectorizer(tokenizer=split_tokenizer,
                                          vocabulary=recipients)

    def fit_input(self, s_clean_body):
        print("Fitting input ...")
        self.input_bow.fit(s_clean_body)
        self.n_features = len(self.input_bow.get_feature_names())

    def fit_output(self, s_recipients):
        print("Fitting  output ...")
        self.output_bow.fit(s_recipients)
        self.n_outputs = len(self.output_bow.get_feature_names())

    def vectorize_input(self, s_clean_body):
        X = self.input_bow.transform(s_clean_body).toarray()
        return X

    def vectorize_output(self, s_recipients):
        Y = self.output_bow.transform(s_recipients).toarray()
        return Y
