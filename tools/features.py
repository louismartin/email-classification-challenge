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
    recipients_list = recipient_series.str.split()
    unique_rec_set = set(unique_recipients)

    def filter_non_email(rec_list):
        return [rec for rec in rec_list if rec in unique_rec_set]

    recipients_list = recipients_list.apply(filter_non_email)

    def single_boe_string(rec_list):
        boe_vector = np.zeros(len(unique_recipients))
        for rec in rec_list:
            idx = unique_recipients.index(rec)
            boe_vector[idx] = 1
        return " ".join((str(int(el)) for el in boe_vector))

    recipients_boe = recipients_list.apply(single_boe_string)
    df_boe = recipients_boe.str.split(pat=" ", expand=True)
    df_boe = df_boe.apply(pd.to_numeric, downcast="integer")
    return df_boe.as_matrix()
