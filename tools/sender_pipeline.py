import numpy as np

from tools.evaluation import top_emails, evaluate
from tools.features import normalized_sent_frequency


class SenderModel():
    def __init__(self, df_emails, classifier, reranking_classifier,
                 input_vectorizer, output_vectorizer, n_received):
        '''
            Args:
                - df_emails (pd dataframe): the dataframe to use for this
                model. It needs "clean_body" and "recipients" columns.
                - classifier (object): the classifier (or regressor) you want
                to use. It needs to have the following functions: fit, predict.
                - reranking_classifier (object): the classifier (or regressor)
                you want to use to rerank the recipients. It needs to have the
                following functions: fit, predict.
                - input_vectorizer (object): an object that will vectorize the
                input data given as the "clean_body" column of the dataframe.
                It needs to have the following functions: fit_transform,
                transform.
                - output_vectorizer (object): an object that will vectorize
                the recipients given as the "recipients" column of the
                dataframe. It needs to have the following functions:
                fit_transform, get_feature_names.
                - n_received (dict): a dictionary giving for each
                recipient the number of emails received.
        '''
        self.df_emails = df_emails
        self.classifier = classifier
        self.reranking_classifier = reranking_classifier
        self.input_vectorizer = input_vectorizer
        self.output_vectorizer = output_vectorizer
        self.n_received = n_received

    def train(self, train_prop=1, reranking=False):
        '''Trains the classifier after having vectorized input and output data.
            Args:
                - train_prop (float): the porportion of emails you want to keep
                for this training.
            Returns:
                - self
        '''
        # data loading and separation
        n_train = self.df_emails.shape[0]
        df_train = self.df_emails[:int(train_prop * n_train)]
        self.train_ids = list(df_train.index.values)
        # feature engineering
        X_train = self.input_vectorizer.fit_transform(df_train["clean_body"])
        Y_train = self.output_vectorizer.fit_transform(df_train["recipients"])
        # primary model fitting
        self.classifier.fit(X_train, Y_train.toarray())
        if reranking:
            # score generation
            scores = self.classifier.predict(X_train)
            # frequency features
            self.n_recipients = scores.shape[1]
            self.n_sent_freq = normalized_sent_frequency(Y_train.toarray())
            self.n_received_freq = np.zeros(self.n_recipients)
            for r, recipient in enumerate(
                self.output_vectorizer.get_feature_names()
            ):
                user_received = sum(Y_train.toarray()[:, r])
                if user_received == 0:
                    self.n_received_freq[r] = 0
                else:
                    self.n_received_freq[r] = user_received / self.n_received[
                        recipient]
            self.n_sum = self.n_sent_freq + self.n_received_freq
            self.n_sum /= np.linalg.norm(self.n_sum)
            # reranking fitting
            reranking_matrix = np.zeros((scores.size, 4))
            reranking_output = np.zeros(scores.size)
            for m in range(scores.shape[0]):
                for r in range(self.n_recipients):
                    reranking_matrix[m * self.n_recipients + r, 0] = scores[m, r]
                    reranking_matrix[
                        m * self.n_recipients + r, 1] = self.n_sent_freq[r]
                    reranking_matrix[
                        m * self.n_recipients + r, 2] = self.n_received_freq[r]
                    reranking_matrix[m * self.n_recipients + r, 3] = self.n_sum[r]
                    reranking_output[m * self.n_recipients + r] = Y_train[m, r]
            self.reranking_classifier.fit(reranking_matrix, reranking_output)
        return self

    def evaluate(self, train_prop=0.7, reranking=False):
        '''Computes the precision at 10 for this model on a test set extracted
        from the dataframe.
            Args:
                - train_prop (float): the proportion of emails used for the
                training.
            Returns:
                - float: the precision at 10.
        '''
        self.train(train_prop=train_prop, reranking=reranking)
        # data loading
        train_mask = self.df_emails.index.isin(self.train_ids)
        df_test = self.df_emails[~train_mask]
        # feature engineering
        X_test = self.input_vectorizer.transform(df_test["clean_body"])
        # Prediction
        Y_test = self.classifier.predict(X_test)
        if reranking:
            # Reranking
            reranking_matrix = np.zeros((Y_test.size, 4))
            for m in range(Y_test.shape[0]):
                for r in range(self.n_recipients):
                    reranking_matrix[
                        m * self.n_recipients + r, 0] = Y_test[m, r]
                    reranking_matrix[
                        m * self.n_recipients + r, 1] = self.n_sent_freq[r]
                    reranking_matrix[
                        m * self.n_recipients + r, 2] = self.n_received_freq[r]
                    reranking_matrix[
                        m * self.n_recipients + r, 3] = self.n_sum[r]

            reranking_output = self.reranking_classifier.predict(
                reranking_matrix)
            for m in range(Y_test.shape[0]):
                for r in range(self.n_recipients):
                    Y_test[m, r] = reranking_output[m * self.n_recipients + r]
        # Decoding
        recipients_map = self.output_vectorizer.get_feature_names()
        predicted_recipients = top_emails(Y_test, recipients_map)
        ground_truth = df_test["recipients"].str.split(expand=True).as_matrix()
        prec = evaluate(predicted_recipients, ground_truth)
        return prec

    def predict(self, mids, df_submission, reranking=False):
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
        if reranking:
            # Reranking
            reranking_matrix = np.zeros((Y_eval.size, 4))
            for m in range(Y_eval.shape[0]):
                for r in range(self.n_recipients):
                    reranking_matrix[
                        m * self.n_recipients + r, 0] = Y_eval[m, r]
                    reranking_matrix[
                        m * self.n_recipients + r, 1] = self.n_sent_freq[r]
                    reranking_matrix[
                        m * self.n_recipients + r, 2] = self.n_received_freq[r]
                    reranking_matrix[
                        m * self.n_recipients + r, 3] = self.n_sum[r]

            reranking_output = self.reranking_classifier.predict(
                reranking_matrix)
            for m in range(Y_eval.shape[0]):
                for r in range(self.n_recipients):
                    Y_eval[m, r] = reranking_output[m * self.n_recipients + r]
        # Decoding
        recipients_map = self.output_vectorizer.get_feature_names()
        predicted_recipients = top_emails(Y_eval, recipients_map)
        for index_eval, row_eval in df_eval.iterrows():
            i = df_eval.index.get_loc(index_eval)
            rec_pred = " ".join(predicted_recipients[i, :])
            df_submission.set_value(index_eval, "recipients", rec_pred)
