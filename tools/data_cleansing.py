from nltk.corpus import stopwords


def remove_numbers_and_ponctuation(text_series):
    '''Removes numbers and ponctuation from all text elements of the pd series
    and returns the result as a new pd series.
        Arguments:
            - text_series (pd series): the series containing the text data you
            want to cleanse.
        Output:
            - pd series: the cleansed series.
    '''
    only_letters_pattern = "[^a-zA-Z]"
    return text_series.str.lower().str.replace(only_letters_pattern, " ")


def remove_stopwords(text_series):
    '''Remove all stopwords from nltk from a pd series.
        Arguments:
            - text_series (pd series): the series containing the text data
            whose stopwords you want to remove.
        Output:
            - pd series: a pd series containing list objects of the words of
            the text data that were not stopwords.
    '''
    word_list_series = text_series.str.split(" ")
    stops = stopwords.words("english")
    return word_list_series.apply(lambda text:
                                  [word for word in text
                                   if word not in stops and len(word) > 0])
