from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def print_email_by_id(df_emails, mid):
    '''Uses the dataframe emails to print the email given by its mid in a
    classic format.
        Arguments:
            - df_emails (pd dataframe): the pd dataframe containing the emails
            - mid (int)
    '''
    email_dict = df_emails.loc[mid].to_dict()
    email_content = "From: {sender} \nTo: {recipients}\nOn: {date} \n\
Body:\n {body}".format(**email_dict)
    print(email_content)


def emails_sent_distribution(df_email_senders, bins=20, fig_number=None):
    '''Plots the distribution of number of emails sent by sender.
        Arguments:
            - df_email_senders (pd dataframe): the pd dataframe containing the
            emails sent for every sender.
            - bins (int): the number of bins in the histogram plot.
            - fig_number (int): the index of the plot figure.
    '''
    n_emails_sent = df_email_senders["mids"].str.split().str.len()
    if fig_number:
        plt.figure(fig_number)
    else:
        plt.figure()
    plt.hist(n_emails_sent.as_matrix(), bins=bins, normed=True)
    plt.title("Number of emails sent distribution")
    plt.xlabel("Number of emails sent")
    plt.ylabel("Frequency")
    plt.show()


def emails_received_distribution(df_emails, bins=20, fig_number=None):
    '''Plots the distribution of number of emails received by recipient.
        Arguments:
            - df_emails (pd dataframe): the pd dataframe containing the
            emails sent for every sender.
            - bins (int): the number of bins in the histogram plot.
            - fig_number (int): the index of the plot figure.
    '''
    all_recipients = df_emails["recipients"].str.cat(sep=" ").split()
    recipients_count = Counter(all_recipients)
    n_emails_received = np.array(list(recipients_count.values()))
    if fig_number:
        plt.figure(fig_number)
    else:
        plt.figure()
    plt.hist(n_emails_received, bins=bins, normed=True)
    plt.title("Number of emails received distribution")
    plt.xlabel("Number of emails received")
    plt.ylabel("Frequency")
    plt.show()
