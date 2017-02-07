import os.path as op

import pandas as pd

from tools.utils import save_and_reload_df

DATA_FOLDER = "data"


def load_emails(set_type="training"):
    '''Returns a pd dataframe containing the content of "training_info.csv" or
    "test_info.csv".
        Arguments:
            - set_type (str): either "training" or "test".
        Output:
            - pd dataframe: the emails. The columns are [date, body,
            recipients] and the id is mid.
    '''
    training_info_path = op.join(DATA_FOLDER, "{}_info.csv".format(set_type))
    df_emails = pd.read_csv(training_info_path, index_col=0)
    return df_emails


def load_email_senders(set_type="training"):
    '''Returns a pd dataframe containing the content of "training_set.csv"
    or "test_set.csv".
        Arguments:
            - set_type (str): either "training" or "test".
        Output:
            - pd dataframe: the email senders. The columns are [sender, mids]
    '''
    training_set_path = op.join(DATA_FOLDER, "{}_set.csv".format(set_type))
    return pd.read_csv(training_set_path)


@save_and_reload_df
def enrich_emails():
    '''Adds the sender column to the emails dataframe and returns it.
        Output:
            - pd dataframe: the emails enriched. The columns are [mid, date,
            body, recipients, sender]
    '''
    df_emails = load_emails()
    df_email_senders = load_email_senders()
    df_emails["sender"] = ""
    for index, row in df_email_senders.iterrows():
        for mid in row["mids"].split():
            df_emails.set_value(int(mid), "sender", row["sender"])
    return df_emails


def unique_recipients(df_emails):
    '''Returns a sorted list of all unique recipients
        Arguments:
            - df_emails (pd dataframe): the emails dataframe.
        Output:
            - list: all unique recipients.
    '''
    all_recipients = df_emails["recipients"].str.cat(sep=" ").split()
    unique_recipients = set(all_recipients)
    # we need to get rid of recipients that are not emails
    return sorted([rec for rec in list(unique_recipients) if "@" in rec])


def address_book(recipients):
    '''Returns a set containing all names that can be found in the recipients'
    email addresses.
        Arguments:
            - recipients (iterable): contains all the unique recipients
            "verfied" email addresses.
    '''
    address_book = set()
    for rec in recipients:
        # At first we split the email address into two to get the separate the
        # alias from the domain name.
        split_address = rec.split("@")
        if len(split_address) > 1:
                # Then we split the alias into chunks separated by blocks in
                # order to identify if these chunks could be names.
                potential_names = split_address[0].split(".")
                for name in potential_names:
                    # We say that a chunk is a name if it's not only digits and
                    # has at least 2 characters in it.
                    if not name.isdigit() and len(name) > 1:
                        address_book.add(name)
    return address_book
