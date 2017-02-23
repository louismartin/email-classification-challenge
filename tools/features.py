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
