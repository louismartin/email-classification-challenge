import re

from nltk.corpus import stopwords, words

from tools.utils import save_and_reload_df
from tools.data_handling import enrich_emails, unique_recipients, address_book
from tools.features import stem

stopwords = set(stopwords.words("english"))
english_words = set(words.words())


def remove_after_indicator(text, indicator, min_words=10):
    '''Removes everything in text after indicator if found. If not found, leaves
    text as is.
        Arguments:
            - text (str): the text you want to shorten.
            - indicator (str): the indicator after which you want to cut the
            text.
        Output:
            - str: the shortened text.
    '''
    indic_matches = re.finditer(indicator, text)
    if indic_matches:
        simple_text = False
        for match in indic_matches:
            position = match.span(0)[0]
            if len(text[:position].split()) > min_words:
                simple_text = text[:position]
        if not simple_text:
            simple_text = text
    else:
        simple_text = text
    return simple_text


def remove_punctuation(text):
    '''Removes punctuation from a string
        Arguments:
            - text (str): string to cleanse.
        Output:
            - str: the cleansed string.
    '''
    return re.sub("[\W_]+", " ", text)


def remove_numbers(text):
    '''Removes numbers from a string
        Arguments:
            - text (str): string to cleanse.
        Output:
            - str: the cleansed string.
    '''
    return re.sub("\d+", " ", text)


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


def remove_non_emails(text):
    '''Remove all emails from text which don't contain @'''
    emails = text.split()
    kept_emails = [email for email in emails if ('@' in email)]
    return " ".join(kept_emails)


def stem_words(text, except_words):
    words = text.split()
    clean_text = " ".join([stem(word, except_words) for word in words])
    return clean_text


def clean(text, except_words=None, only_english=False):
    ''' Clean a string using several methods '''
    text = remove_after_indicator(text, "Original Message")
    text = remove_after_indicator(text, "Forwarded by")
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    if only_english:
        text = remove_non_english_words(text, except_words)
    text = stem_words(text, except_words)
    return text
