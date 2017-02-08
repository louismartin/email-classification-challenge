from collections import Counter
import os.path as op
import string

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


def unique_recipients(df_emails, min_rec=1):
    '''Returns a sorted list of all unique recipients
        Arguments:
            - df_emails (pd dataframe): the emails dataframe.
            - min_rec (int): minimum number of times an email has been received
            by a recipient.
        Output:
            - list: all unique recipients.
    '''
    all_recipients = df_emails["recipients"].str.cat(sep=" ").split()
    recipients_count = Counter(all_recipients)
    unique_recipients = set(
        (rec for rec in recipients_count if recipients_count[rec] >= min_rec))
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


def get_domain_name(email_address):
    '''Simple heuristic to get domain name out of email address.
        Arguments:
            - email_address (str): the email address whose domain you want to
            get.
        Output:
            - str: the domain name as guessed by the heuristic.
    '''
    return email_address.split("@")[1]


def unique_domain_names(unique_rec_set):
    '''Out of the set of unique recipients given by their email addresses,
    computes all the unique domain names, sorted in descending order by number
    of times it appears in the set of unique recipients.
        Arguments:
            - unique_rec_set (set): set of all unique recipients given by email
            addresses.
        Output:
            - list: all domain names given by the simple heuristic, sorted,
            starting from the most frequent.
    '''
    domain_names_gen = map(get_domain_name, unique_rec_set)
    domain_names_count = Counter(domain_names_gen)
    return sorted(domain_names_count, key=domain_names_count.get, reverse=True)


def potential_aliases(name):
    '''Very simple heuristic to give potential aliases for a name in an email
    address.
        Arguments:
            - name (str): the name whose potential aliases list you want.
        Output:
            - list: the list of potential aliases.
    '''
    only_letters_pattern = "[^a-zA-Z]"
    name = name.lower()
    name = name = "".join((ch for ch in name if ch not in string.punctuation))
    name_elements = name.split()
    # Suppose the surname comes second (with middle name last), we can take the
    # initial of the surname.
    initial_surname = [name_elements[0], name_elements[1][0]]
    return [
        ".".join(name_elements[:2][::-1]),
        ".".join(name_elements[:2]),
        ".".join(name_elements[::-1]),
        ".".join(name_elements),
        "..".join(name_elements[::-1]),
        ".".join(initial_surname[::-1]),
        "..".join(initial_surname[::-1]),
        "_".join(name_elements[::-1]),
        "_".join(name_elements),
        "-".join(name_elements[::-1]),
        "-".join(name_elements),
        name_elements[0],
        "".join(name_elements[::-1]),
        "".join(initial_surname[::-1]),
        name_elements[1]
    ]


def name_to_address(unique_rec_set, domain_names, name):
    '''From a simple heuristic to find potential aliases for a given name
    (potential_aliases), a list of domain names returns a possible
    email address in the unique_rec_set for that name ("most probable" if you
    sort the domain names).
        Arguments:
            - unique_rec_set (iterable): all unique recipients.
            - domain_names (iterable): all possible domain names (sorted is
            better for efficiency and likeliness).
            - name (str): the name whose email address you want to find.
        Output:
            - str: the email address or None if none was find in the
            unique_rec_set.
    '''
    for domain_name in domain_names:
        for alias in potential_aliases(name):
            email = "@".join([alias, domain_name])
            if email in unique_rec_set:
                return email


def test_name_to_address():
    names = [
        "Miller, Mary Kay",
        "Nelson, Michel",
        "Neubauer, Dave",
        "McGowan, Mike W.",
        "team.waterloo plant@enron.com",
        "Melville, Keith"
    ]
    df_emails = enrich_emails()
    unique_rec = set(unique_recipients(df_emails))
    domain_names = unique_domain_names(unique_rec)
    for name in names:
        print(name_to_address(unique_rec_set=unique_rec,
                              domain_names=domain_names,
                              name=name))
