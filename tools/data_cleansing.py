import re

from nltk.corpus import stopwords, words

from tools.utils import save_and_reload_df
from tools.data_handling import enrich_emails, unique_recipients, address_book

stopwords = set(stopwords.words("english"))
english_words = set(words.words())


def remove_after_indicator(text, indicator):
    '''Removes everything in text after indicator if found. If not found, leaves
    text as is.
        Arguments:
            - text (str): the text you want to shorten.
            - indicator (str): the indicator after which you want to cut the
            text.
        Output:
            - str: the shortened text.
    '''
    indic_match = re.search(indicator, text)
    if indic_match:
        simple_text = text[:indic_match.span(0)[0]]
    else:
        simple_text = text
    return simple_text


def remove_numbers_and_ponctuation(s_text):
    '''Removes numbers and ponctuation from all text elements of the pd series
    and returns the result as a new pd series.
        Arguments:
            - s_text (pd series): the series containing the text data you
            want to cleanse.
        Output:
            - pd series: the cleansed series.
    '''
    only_letters_pattern = "[^a-zA-Z]"
    return s_text.str.lower().str.replace(only_letters_pattern, " ")


def remove_stopwords(text):
    '''Remove all stopwords from nltk from a string.
        Arguments:
            - text (str): the string from which to remove stopwords
        Output:
            - clean_text (str): the string without the stopwords
    '''
    words = text.split(" ")
    clean_words = [word for word in words
                   if word not in stopwords and len(word) > 0]

    clean_text = " ".join(clean_words)
    return clean_text


def remove_non_english_words(text, except_words=None):
    '''Remove all non english words as defined by nltk from a string.
        Arguments:
            - text (str): string to with words to be removed
            - except_words (iterable of strings): contains all strings you want
            to keep even though not an english word as defined by nltk.
        Output:
            - clean_text (str): string where non-english words are removed.
    '''
    words = text.split()
    if except_words:
        words_to_remove = english_words.union(set(except_words))
    else:
        words_to_remove = english_words

    clean_words = [word for word in words if (word not in words_to_remove)]
    clean_text = " ".join(clean_words)
    return clean_text


def clean(s, except_words, only_english=False):
    s = s.apply(lambda x: remove_after_indicator(x, "Original Message"))
    s = s.apply(lambda x: remove_after_indicator(x, "Forwarded by"))
    s = remove_numbers_and_ponctuation(s)
    s = s.apply(lambda x: remove_stopwords(x))
    if only_english:
        s = s.apply(lambda x: remove_non_english_words(x, except_words))
    s = s.fillna("")
    return s


@save_and_reload_df
def get_clean_df_train(ratio=0.9):
    """Quick'n'Dirty method"""
    df_emails = enrich_emails()

    n_train = int(ratio * df_emails.shape[0])
    df_train = df_emails.sample(n=n_train, random_state=0)

    recipients = unique_recipients(df_train)
    names = address_book(recipients)
    df_train["clean body"] = clean(df_train["body"], except_words=names)
    return df_train


@save_and_reload_df
def get_clean_df_test():
    """Quick'n'Dirty method"""
    df_emails = enrich_emails()

    df_train = get_clean_df_train()
    df_test = df_emails.drop(df_train.index)

    recipients = unique_recipients(df_train)
    names = address_book(recipients)
    df_test["clean body"] = clean(df_test["body"], except_words=names)
    return df_test
