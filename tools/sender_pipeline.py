from tools.evaluation import precision, top_emails


class SenderModel():
    def __init__(self, df_emails, classifier, input_vectorizer,
                 output_vectorizer):
        self.df_emails = df_emails
        self.classifier = classifier
        self.input_vectorizer = input_vectorizer
        self.output_vectorizer = output_vectorizer

    def train(self, train_prop=1):
        # data loading and separation
        df_train = self.df_emails.sample(frac=train_prop)
        self.train_ids = list(df_train.index.values)
        # feature engineering
        X_train = self.input_vectorizer.fit_transform(df_train["clean body"])
        Y_train = self.output_vectorizer.fit_transform(df_train["recipients"])
        # model fitting
        self.classifier.fit(X_train, Y_train.toarray())
        return self

    def evaluate(self, train_prop=0.7):
        self.train(train_prop=train_prop)
        # data loading
        train_mask = self.df_emails.index.isin(self.train_ids)
        df_test = self.df_emails[~train_mask]
        n_test = df_test.shape[0]
        # feature engineering
        X_test = self.input_vectorizer.transform(df_test["clean body"])
        # Prediction
        Y_test = self.classifier.predict(X_test)
        # Decoding
        recipients_map = self.output_vectorizer.get_feature_names()
        predicted_recipients = top_emails(Y_test, recipients_map)
        preci = 0
        for index_test, row_test in df_test.iterrows():
            i = df_test.index.get_loc(index_test)
            rec_pred = " ".join(predicted_recipients[i, :])
            preci += precision(rec_pred, row_test["recipients"])

        preci /= n_test
        return preci

    def predict(self, mids, df_submission):
        # data loading
        df_eval = df_submission.ix[mids]
        # feature engineering
        X_eval = self.input_vectorizer.transform(df_eval["clean body"])
        # Prediction
        Y_eval = self.classifier.predict(X_eval)
        # Decoding
        recipients_map = self.output_vectorizer.get_feature_names()
        predicted_recipients = top_emails(Y_eval, recipients_map)
        for index_eval, row_eval in df_submission.iterrows():
            i = df_submission.index.get_loc(index_eval)
            rec_pred = " ".join(predicted_recipients[i, :])
            df_submission.set_value(index_eval, "recipients", rec_pred)
