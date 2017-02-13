from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tools.data_handling import enrich_emails, load_email_senders,\
    address_book, unique_recipients
from tools.data_cleansing import clean
from tools.evaluation import precision


def expected_precision(sender_prop=0.3,
                       train_prop=0.7,
                       min_df=1,
                       min_rec=1,
                       n_estimators=10,
                       max_depth=30,
                       min_sample_prop=0.04,
                       top=10):
    df_emails = enrich_emails()
    df_email_senders = load_email_senders()
    unique_rec_train = unique_recipients(df_emails)
    add_book = address_book(unique_rec_train)
    add_book.add("fyi")
    st = LancasterStemmer()

    def stem(word):
        if word in add_book:
            return word
        else:
            return(st.stem(word))

    def stem_tokenizer(s):
        return [stem(word) for word in s.split(" ")]

    def split_tokenizer(s):
        return s.split(" ")

    df_small_senders = df_email_senders.sample(frac=sender_prop)
    df_precision = pd.DataFrame(columns=["sender", "n_mails", "precision"])
    for index, row in df_small_senders.iterrows():
        sender = row["sender"]
        mids = list(map(int, row["mids"].split()))
        n_mails = len(mids)
        # data loading and separation
        df_interest = df_emails.ix[mids]
        df_train = df_interest.sample(frac=train_prop)
        train_ids = list(df_train.index.values)
        train_mask = df_interest.index.isin(train_ids)
        df_test = df_interest[~train_mask]
        n_mails_test = df_test.shape[0]
        # data cleansing
        unique_rec_train = unique_recipients(df_train)
        add_book = address_book(unique_rec_train)
        df_train["clean body"] = clean(df_train["body"], add_book)
        df_test["clean body"] = clean(df_test["body"], add_book)
        # feature engineering
        input_bow = TfidfVectorizer(norm="l2",
                                    min_df=min_df,
                                    tokenizer=stem_tokenizer)
        X_train = input_bow.fit_transform(df_train["clean body"])
        hour_train = sparse.csr_matrix(
            df_train["timestamp"].dt.hour.as_matrix()).transpose()
        day_train = sparse.csr_matrix(
            df_train["timestamp"].dt.dayofweek.as_matrix()).transpose()
        X_train = sparse.hstack((X_train, hour_train, day_train))
        X_test = input_bow.transform(df_test["clean body"])
        hour_test = sparse.csr_matrix(
            df_test["timestamp"].dt.hour.as_matrix()).transpose()
        day_test = sparse.csr_matrix(
            df_test["timestamp"].dt.dayofweek.as_matrix()).transpose()
        X_test = sparse.hstack((X_test, hour_test, day_test))
        output_bow = CountVectorizer(tokenizer=split_tokenizer,
                                     min_df=min_rec,
                                     vocabulary=unique_rec_train)
        Y_train = output_bow.fit_transform(df_train["recipients"])
        # model fitting
        min_samples_leaf = max(1, int(min_sample_prop*n_mails))
        rf = RandomForestRegressor(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   n_jobs=-1,
                                   min_samples_leaf=min_samples_leaf)
        rf.fit(X_train, Y_train.toarray())
        # predictions
        Y_test = rf.predict(X_test)
        # decoding
        recipients_map = output_bow.get_feature_names()
        if len(Y_test.shape) > 1 and top < Y_test.shape[1]:
            best_pred_idx = np.argpartition(-Y_test, top, axis=1)[:, :top]
            sorted_ids = np.argsort(
                Y_test[np.arange(
                    Y_test.shape[0])[:, None], best_pred_idx])[:, ::-1]
            sorted_idx = best_pred_idx[np.arange(
                best_pred_idx.shape[0])[:, None], sorted_ids]
        else:
            sorted_idx = np.argsort(-Y_test)
        preci = 0
        for index_test, row_test in df_test.iterrows():
            i = df_test.index.get_loc(index_test)
            if len(recipients_map) > 1:
                rec_ids = sorted_idx[i, :]
                rec_pred = " ".join(
                    [recipients_map[rec_id] for rec_id in rec_ids])
            else:
                rec_pred = recipients_map[0]
            preci += precision(rec_pred, row_test["recipients"])
        preci /= n_mails_test
        df_precision.loc[index] = [sender, n_mails_test, preci]
    return (df_precision["n_mails"]*df_precision["precision"]/df_precision["n_mails"].sum()).sum()
