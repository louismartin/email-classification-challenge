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


def graph_of_words(text, window=5):
    """ Implement the graph of word (GoW) method on the input string.
    Args:
        text (str): input text to be represented using GoW
        window (int): Size of the sliding window
    Returns:
        words (list): list of words
        count (list): Number of inbound edges for each word
    """
    graph = {}
    # Represent a graph as a dictionary
    # The usual implementation is:
    #     - Each key is a node and its value is a list of its children
    # However we will implement it in a reversed way because the final
    # information we need is the number of inbound nodes, i.e. number
    # of parents and not number of children. The implementation is:
    #     - Each key is a node and its value is a list of its parents

    text = clean(text)
    words = text.split()
    for i, parent in enumerate(words):
        children = words[i+1:i+window]
        for child in children:
            if child not in graph:
                graph[child] = set()
            graph[child].add(parent)

    inbound_count = {k: len(v) for k, v in graph.items()}
    words = list(graph.keys())
    count = [len(v) for v in graph.values()]
    return words, count
