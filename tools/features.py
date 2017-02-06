import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(text_series):
    '''From a pd series containing cleansed text data, computes the bag of words
    (BoW) scikit learn object and returns the BoW vector.
        Arguments:
            - text_series (pd series): the series containg the text data you
            want to bag.
        Output:
            - np array: BoW vectors.
            - CountVectorizer: the scikit learn BoW object.
    '''
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(text_series)
    return X, vectorizer


def bag_of_emails(recipient_series, unique_recipients):
    '''From a pd series containing email recipients, computes a np array that is
    the equivalent of a bag of words for email addresses.
        Arguments:
            - recipient_series (pd series): the series containing the
            recipients of each email.
            - unique_recipients (sorted list): pre-computed sorted list of all
            unique recipients you want to consider.
        Output:
            - np array: BoW vectors.
    '''
    def split_tokenizer(s):
        return s.split(" ")
    vectorizer = CountVectorizer(min_df=1,
                                 tokenizer=split_tokenizer,
                                 vocabulary=unique_recipients)
    X = vectorizer.fit_transform(recipient_series)
    return X
