import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(s_text, min_df=1, max_features=None):
    '''From a pd series containing cleansed text data, computes the bag of words
    (BoW) scikit learn object and returns the BoW vector.
        Arguments:
            - s_text (pd series): the series containg the text data you
            want to bag.
            - max_features (int): maximum features in the bow (most frequent)
        Output:
            - np array: BoW vectors.
            - CountVectorizer: the scikit learn BoW object.
    '''
    vectorizer = CountVectorizer(min_df=min_df, max_features=max_features)
    X = vectorizer.fit_transform(s_text)
    return X, vectorizer


def bag_of_emails(s_recipient, unique_recipients, min_df=1, max_features=None):
    '''From a pd series containing email recipients, computes a np array that is
    the equivalent of a bag of words for email addresses.
        Arguments:
            - s_recipient (pd series): the series containing the
            recipients of each email.
            - unique_recipients (sorted list): pre-computed sorted list of all
            unique recipients you want to consider.
            - max_features (int): maximum features in the bow (most frequent)
        Output:
            - np array: BoW vectors.
            - CountVectorizer: the scikit learn BoW object.
    '''
    def split_tokenizer(s):
        return s.split(" ")
    if max_features:
        # max_features is not taken into account if there is a vocabulary
        unique_recipients = None
    vectorizer = CountVectorizer(min_df=min_df,
                                 tokenizer=split_tokenizer,
                                 vocabulary=unique_recipients,
                                 max_features=max_features)

    X = vectorizer.fit_transform(s_recipient)
    return X, vectorizer
