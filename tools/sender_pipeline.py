from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from data_cleansing import clean
from data_handling import address_book, unique_recipients
from evaluation import precision, top_emails
from features import split_tokenizer


class SenderModel():
    def __init__(self, mids, df_emails, rf_args, except_words={},
                 only_english=False):
        # Row unpacking
        self.n_mails = len(mids)
        self.df_interest = df_emails.ix[mids]
        self.except_words = except_words
        self.only_english = only_english
        self.rf_args = rf_args
        self.input_bow = TfidfVectorizer(norm="l2")

    def fit(self, train_prop=1):
        # data loading and separation
        df_train = self.df_interest.sample(frac=train_prop)
        self.train_ids = list(df_train.index.values)
        # data cleansing
        self.unique_rec_train = unique_recipients(df_train)
        add_book = address_book(self.unique_rec_train)
        self.except_words = self.except_words.union(add_book)
        df_train["clean body"] = df_train["body"].apply(
            lambda x: clean(x, self.except_words,
                            only_english=self.only_english))
        df_train["clean body"] = df_train["clean body"].fillna("")
        # feature engineering
        X_train = self.input_bow.fit_transform(df_train["clean body"])
        self.output_bow = CountVectorizer(tokenizer=split_tokenizer,
                                          vocabulary=self.unique_rec_train)
        Y_train = self.output_bow.fit_transform(df_train["recipients"])
        # model fitting
        self.rf = RandomForestRegressor(
            min_samples_leaf=max(1, int(0.0002 * self.n_mails)),
            **self.rf_args)
        self.rf.fit(X_train, Y_train.toarray())
        return self

    def evaluate(self, train_prop=0.7):
        self.fit(train_prop=train_prop)
        # data loading
        train_mask = self.df_interest.index.isin(self.train_ids)
        df_test = self.df_interest[~train_mask]
        n_test = df_test.shape[0]
        # data cleansing
        df_test["clean body"] = clean(df_test["body"], self.except_words)
        # feature engineering
        X_test = self.input_bow.transform(df_test["clean body"])
        # Prediction
        Y_test = self.rf.predict(X_test)
        # Decoding
        recipients_map = self.output_bow.get_feature_names()
        predicted_recipients = top_emails(Y_test, recipients_map)
        preci = 0
        for index_test, row_test in df_test.iterrows():
            i = df_test.index.get_loc(index_test)
            rec_pred = " ".join(predicted_recipients[i, :])
            preci += precision(rec_pred, row_test["recipients"])

        preci /= n_test
        return preci

    def predict(self, mids, df_submission):
        df_eval = df_submission.ix[mids]
        # data cleansing
        df_eval["clean body"] = clean(df_eval["body"], self.except_words)
        # feature engineering
        X_eval = self.input_bow.transform(df_eval["clean body"])
        # Prediction
        Y_eval = self.rf.predict(X_eval)
        # Decoding
        recipients_map = self.output_bow.get_feature_names()
        predicted_recipients = top_emails(Y_eval, recipients_map)
        for index_eval, row_eval in df_submission.iterrows():
            i = df_submission.index.get_loc(index_eval)
            rec_pred = " ".join(predicted_recipients[i, :])
            df_submission.set_value(index_eval, "recipients", rec_pred)
