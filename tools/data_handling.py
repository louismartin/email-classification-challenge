import os.path as op

import pandas as pd

DATA_FOLDER = "data"


def load_emails():
    '''Returns a pd dataframe containing the content of "training_info.csv".
        Output:
            - pd dataframe: the emails. The columns are [date, body,
            recipients] and the id is mid.
    '''
    training_info_path = op.join(DATA_FOLDER, "training_info.csv")
    emails = pd.read_csv(training_info_path)
    emails = emails.set_index("mid")
    return emails


def load_email_senders():
    '''Returns a pd dataframe containing the content of "training_set.csv".
        Output:
            - pd dataframe: the email senders. The columns are [sender, mids]
    '''
    training_set_path = op.join(DATA_FOLDER, "training_set.csv")
    return pd.read_csv(training_set_path)


def enrich_emails(overwrite=False):
    '''Adds the sender column to the email dataframe. By default takes it from
    "training_info_enriched.csv".
        Arguments:
            - overwrite (bool): whether you want to recompute the enrichment.
            If True, the enrichment will be recomputed and the file
            overwritten.
        Output:
            - pd dataframe: the emails enriched. The columns are [mid, date,
            body, recipients, sender]
    '''
    enriched_emails_path = op.join(DATA_FOLDER, "training_info_enriched.csv")
    if overwrite or not op.exists(enriched_emails_path):
        emails = load_emails()
        email_senders = load_email_senders()
        emails["sender"] = ""
        for index, row in email_senders.iterrows():
            for mid in row["mids"].split():
                emails.set_value(int(mid), "sender", row["sender"])
        emails.to_csv(enriched_emails_path)
    else:
        emails = pd.read_csv(enriched_emails_path)
        emails = emails.set_index("mid")
    return emails
