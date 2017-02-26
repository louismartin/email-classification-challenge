from collections import Counter

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
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
    def __init__(self, input_bow, output_bow):
        self.input_bow = input_bow
        self.output_bow = output_bow

    def fit_input(self, s_clean_body):
        print("Fitting input ...")
        self.input_bow.fit(s_clean_body)
        self.n_features = len(self.input_bow.get_feature_names())

    def fit_output(self, s_recipients):
        print("Fitting output ...")
        self.output_bow.fit(s_recipients)
        self.n_outputs = len(self.output_bow.get_feature_names())

    def vectorize_input(self, s_clean_body):
        X = self.input_bow.transform(s_clean_body)
        if sp.issparse(X):
            X = X.toarray()
        return X

    def vectorize_output(self, s_recipients):
        Y = self.output_bow.transform(s_recipients)
        if sp.issparse(Y):
            Y = Y.toarray()
        return Y


# Code inspired from sklearn.feature_extraction.text.CountVectorizer
class AbstractFastVectorizer(BaseEstimator):
    def __init__(self, min_df=1, max_features=None):
        self.min_df = min_df
        self.max_features = max_features

    def _make_vocabulary(self, raw_documents):
        """Create a vocabulary from the documents.
        The vocabulary is a dictionary with the words as keys and their
        associated index as value, e.g. {"foo": 0, "bar": 1, "baz": 2, ...}.
        """
        counter = Counter()
        counter.update((" ".join(raw_documents)).split())
        # List of tuples (word, count) from most frequent to less frequent
        word_count = counter.most_common(self.max_features)
        words_kept = [word for (word, count)
                      in word_count if count >= self.min_df]
        self.vocabulary_ = {word: i for i, word in enumerate(words_kept)}
        self.n_features = len(words_kept)
        self.feature_names = words_kept

    def get_feature_names(self):
        return self.feature_names

    def fit(self, raw_documents, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        self
        """
        self._make_vocabulary(raw_documents)
        return self

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.
        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        X : matrix, [n_samples, n_features]
            Document-term matrix.
        """
        n_samples = len(raw_documents)
        X = np.zeros((n_samples, self.n_features))
        for i, doc in enumerate(raw_documents):
            X[i] = self._vectorize(doc)
        return X

    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return term-document matrix.
        This is equivalent to fit followed by transform.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        X : array, [n_samples, n_features]
            Document-term matrix.
        """
        self.fit(raw_documents)
        X = self.transform(raw_documents)
        return X

    def most_important_words(self, text):
        """ Returns a comprehensive list of tuples (word, count) sorted
        in decreasing order of importance.
        The count is the graph of words measure of number of inbound edges."""
        counts = self._vectorize(text)
        words = self.get_feature_names()
        results = [(x, y) for (y, x) in sorted(zip(counts, words))][::-1]
        return results


class FastCountVectorizer(AbstractFastVectorizer):
    def __init__(self, min_df=1, max_features=None):
        # Set method used vectorize a single string
        self._vectorize = self._bow_vectorize
        super(FastCountVectorizer, self).__init__(min_df=min_df,
                                                  max_features=max_features)

    def _bow_vectorize(self, text):
        # Vector representation of input text
        x = np.zeros(self.n_features)
        vocabulary = self.vocabulary_
        words = text.split()
        for word in words:
            if word in vocabulary:
                index = vocabulary[word]
                x[index] += 1
        return x


class GoWVectorizer(AbstractFastVectorizer):
    def __init__(self, window=5, min_df=1, max_features=None):
        self.window = window
        # Set method used vectorize a single string
        self._vectorize = self._gow_vectorize
        super(GoWVectorizer, self).__init__(min_df=min_df,
                                            max_features=max_features)

    def _gow_vectorize(self, text):
        """ Implement the graph of word (GoW) method on the input string.
        Args:
            text (str): input text to be represented using GoW
        Returns:
            x (ndarray): vector representation of input text
        """
        graph = {}
        # Represent a graph as a dictionary
        # The usual implementation is:
        #     - Each key is a node and its value is a list of its children
        # However we will implement it in a reversed way because the final
        # information we need is the number of inbound nodes, i.e. number
        # of parents and not number of children. The implementation is:
        #     - Each key is a node and its value is a list of its parents
        words = text.split()
        for i, parent in enumerate(words):
            children = words[i+1:i+self.window]
            for child in children:
                if child not in graph:
                    graph[child] = set()
                graph[child].add(parent)

        # Vector representation of input text
        x = np.zeros(self.n_features)
        vocabulary = self.vocabulary_
        for word, parents in graph.items():
            if word in vocabulary:
                index = vocabulary[word]
                inbound_edges = len(parents)
                # We normalize with window to have lower values
                # I don't know if it improves performance but it seemed like a
                # good idea.
                x[index] = inbound_edges/self.window
        return x
