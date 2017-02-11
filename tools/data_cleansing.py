import re

from nltk.corpus import stopwords, words

from tools.utils import save_and_reload_df
from tools.data_handling import enrich_emails, unique_recipients, address_book


def remove_after_indicator_single(text, indicator):
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


def remove_after_indicator(s_text, indicator):
    '''Applies remove_after_indicator_single to a pd series.
    '''
    return s_text.apply(remove_after_indicator_single,
                        indicator=indicator)


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


def remove_stopwords(s_text):
    '''Remove all stopwords from nltk from a pd series.
        Arguments:
            - s_text (pd series): the series containing the text data
            whose stopwords you want to remove.
        Output:
            - pd series: a pd series containing the text data where stopwords
            are removed.
    '''
    s_word_list = s_text.str.split(" ")
    stops = set(stopwords.words("english"))

    def filter_stopwords(word_list):
        return [word for word in word_list
                if word not in stops and len(word) > 0]

    return s_word_list.apply(lambda text: " ".join(filter_stopwords(
        text)))


def remove_non_english_words(s_text, address_book=None):
    '''Remove all non english words as defined by nltk from a pd series.
        Arguments:
            - s_text (pd series): the series containing the text data
            whose non-english words you want to remove.
            - address_book (iterable of strings): contains all strings you want
            to keep even though not an english word as defined by
            nltk.
        Output:
            - pd series: a pd series containing the text data where non-english
            words are removed.
    '''
    s_word_list = s_text.str.split(" ")
    english_words = set(words.words())
    if address_book:
        english_words = english_words.union(set(address_book))

    def filter_non_english_words(word_list):
        return [word for word in word_list
                if word in english_words]

    return s_word_list.apply(lambda text: " ".join(
        filter_non_english_words(text)))


def clean(s, except_words):
    print("Removing original message")
    new_s = remove_after_indicator(s, "Original Message")
    print("Removing forwarded message")
    new_s = remove_after_indicator(new_s, "Forwarded by")
    print("Removing numbers and punctuation")
    new_s = remove_numbers_and_ponctuation(new_s)
    print("Removing stopwords")
    new_s = remove_stopwords(new_s)
    print("Removing non english words")
    new_s = remove_non_english_words(new_s,
                                     address_book=except_words)
    new_s = new_s.fillna("")
    return new_s


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
