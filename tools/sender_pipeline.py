from tools.evaluation import top_emails, evaluate
from tools.staff_graph import add_team_info


class SenderModel():
    def __init__(self, df_emails, classifier, input_vectorizer,
                 output_vectorizer):
        '''
            Args:
                - df_emails (pd dataframe): the dataframe to use for this
                model. It needs "clean_body" and "recipients" columns.
                - classifier (object): the classifier (or regressor) you want
                to use. It needs to have the following functions: fit, predict.
                - input_vectorizer (object): an object that will vectorize the
                input data given as the "clean_body" column of the dataframe.
                It needs to have the following functions: fit_transform,
                transform.
                - output_vectorizer (object): an object that will vectorize
                the recipients given as the "recipients" column of the
                dataframe. It needs to have the following functions:
                fit_transform, get_feature_names.
        '''
        self.df_emails = df_emails
        self.classifier = classifier
        self.input_vectorizer = input_vectorizer
        self.output_vectorizer = output_vectorizer

    def train(self, train_prop=1, random_state=None):
        '''Trains the classifier after having vectorized input and output data.
            Args:
                - train_prop (float): the porportion of emails you want to keep
                for this training.
            Returns:
                - self
        '''
        # data loading and separation
        df_train = self.df_emails.sample(
            frac=train_prop, random_state=random_state)
        self.train_ids = list(df_train.index.values)
        # feature engineering
        X_train = self.input_vectorizer.fit_transform(df_train["clean_body"])
        Y_train = self.output_vectorizer.fit_transform(df_train["recipients"])
        # model fitting
        self.classifier.fit(X_train, Y_train.toarray())
        return self

    def evaluate(self, team_vectorizer, train_prop=0.7, random_state=None):
        '''Computes the precision at 10 for this model on a test set extracted
        from the dataframe.
            Args:
                - train_prop (float): the proportion of emails used for the
                training.
                - team_vectorizer: a instance of the dedicated class
            Returns:
                - float: the precision at 10.
        '''
        self.train(train_prop=train_prop, random_state=random_state)
        # data loading
        train_mask = self.df_emails.index.isin(self.train_ids)
        df_test = self.df_emails[~train_mask]
        # feature engineering
        X_test = self.input_vectorizer.transform(df_test["clean_body"])
        # Prediction
        Y_test = self.classifier.predict(X_test)
        # Decoding
        recipients_map = self.output_vectorizer.get_feature_names()
        predicted_recipients = top_emails(Y_test, recipients_map)
        predicted_recipients =\
            add_team_info(df_test["sender"].iloc[0],
                          predicted_recipients=predicted_recipients,
                          teams=team_vectorizer.teams,
                          n_clusters=team_vectorizer.n_features
                          )
        ground_truth = df_test["recipients"].str.split(expand=True).as_matrix()
        prec = evaluate(predicted_recipients, ground_truth)
        return prec

    def predict(self, mids, df_submission):
        '''Assigns predicted recipients to a submission dataframe.
            Args:
                - mids (list of int): the indices in the dataframe of the
                emails whose recipients you want to predict.
                - df_submission (pd dataframe): the dataframe containing the
                emails.
        '''
        # data loading
        df_eval = df_submission.ix[mids]
        # feature engineering
        X_eval = self.input_vectorizer.transform(df_eval["clean_body"])
        # Prediction
        Y_eval = self.classifier.predict(X_eval)
        # Decoding
        recipients_map = self.output_vectorizer.get_feature_names()
        predicted_recipients = top_emails(Y_eval, recipients_map)
        for index_eval, row_eval in df_eval.iterrows():
            i = df_eval.index.get_loc(index_eval)
            rec_pred = " ".join(predicted_recipients[i, :])
            df_submission.set_value(index_eval, "recipients", rec_pred)
