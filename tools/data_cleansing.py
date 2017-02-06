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
            - pd series: a pd series containing the text data where stopwords
            are removed.
    '''
    word_list_series = text_series.str.split(" ")
    stops = set(stopwords.words("english"))

    def filter_stopwords(word_list):
        return [word for word in word_list
                if word not in stops and len(word) > 0]

    return word_list_series.apply(lambda text: " ".join(filter_stopwords(
        text)))
