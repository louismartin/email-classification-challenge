from nltk.corpus import stopwords, words


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
