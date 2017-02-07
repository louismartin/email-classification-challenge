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
    if "sender" not in email_dict:
        email_dict["sender"] = "Unknown sender"
    email_content = "From: {sender}\nTo: {recipients}\nOn: {date}\n\
Body:\n    {body}".format(**email_dict)
    print(email_content)


def plot_histogram(values, title, xlabel, ylabel, bins=20, fig_number=None):
    '''Plots a histogram for the values given.
        Arguments:
            - values (np array or pd series): the values whose distribution you
            want to plot.
            - title (str): the title of the plot.
            - xlabel (str): the x axis caption.
            - ylabel (str): the y axis caption.
            - bins (int): the number of bins in the histogram plot.
            - fig_number (int): the index of the plot figure.
    '''
    if fig_number:
        plt.figure(fig_number)
    else:
        plt.figure()
    plt.hist(values, bins=bins, normed=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def emails_sent_distribution(df_email_senders, bins=20, fig_number=None,
                             max_value=None):
    '''Plots the distribution of number of emails sent by sender.
        Arguments:
            - df_email_senders (pd dataframe): the pd dataframe containing the
            emails sent for every sender.
            - bins (int): the number of bins in the histogram plot.
            - fig_number (int): the index of the plot figure.
            - max_value (float or int): the maximum value to take into account.
    '''
    n_emails_sent = df_email_senders["mids"].str.split().str.len()
    if max_value is None:
        max_value = n_emails_sent.max()
    n_emails_sent = n_emails_sent[n_emails_sent <= max_value]
    plot_histogram(n_emails_sent,
                   title="Number of emails sent distribution",
                   xlabel="Number of emails sent",
                   ylabel="Frequency",
                   bins=bins,
                   fig_number=fig_number)


def emails_received_distribution(df_emails, bins=20, fig_number=None,
                                 max_value=None):
    '''Plots the distribution of number of emails received by recipient.
        Arguments:
            - df_emails (pd dataframe): the pd dataframe containing the
            emails sent for every sender.
            - bins (int): the number of bins in the histogram plot.
            - fig_number (int): the index of the plot figure.
            - max_value (float or int): the maximum value to take into account.
    '''
    all_recipients = df_emails["recipients"].str.cat(sep=" ").split()
    recipients_count = Counter(all_recipients)
    n_emails_received = np.array(list(recipients_count.values()))
    if max_value is None:
        max_value = np.max(n_emails_received)
    n_emails_received = n_emails_received[n_emails_received <= max_value]
    plot_histogram(n_emails_received,
                   title="Number of emails received distribution",
                   xlabel="Number of emails received",
                   ylabel="Frequency",
                   bins=bins,
                   fig_number=fig_number)


def body_length_distribution(df_emails, bins=20, fig_number=None,
                             max_value=None):
    '''Plots the distribution of the body length for each email.
        Arguments:
            - df_emails (pd dataframe): the pd dataframe containing the
            emails sent for every sender.
            - bins (int): the number of bins in the histogram plot.
            - fig_number (int): the index of the plot figure.
            - max_value (float or int): the maximum value to take into account.
    '''
    body_length = df_emails["body"].str.len()
    if max_value is None:
        max_value = body_length.max()
    body_length = body_length[body_length <= max_value]
    plot_histogram(body_length,
                   title="Body length distribution",
                   xlabel="Body length",
                   ylabel="Frequency",
                   bins=bins,
                   fig_number=fig_number)


def number_of_recipients_distribution(df_emails, bins=20, fig_number=None,
                                      max_value=None):
    '''Plots the distribution of the number of recipients for each email.
        Arguments:
            - df_emails (pd dataframe): the pd dataframe containing the
            emails sent for every sender.
            - bins (int): the number of bins in the histogram plot.
            - fig_number (int): the index of the plot figure.
            - max_value (float or int): the maximum value to take into account.
    '''
    n_recipients = df_emails["recipients"].str.split().str.len()
    if fig_number:
        plt.figure(fig_number)
    else:
        plt.figure()
    if max_value is None:
        max_value = n_recipients.max()
    n_recipients = n_recipients[n_recipients <= max_value]
    plot_histogram(n_recipients,
                   title="Number of recipients distribution",
                   xlabel="Number of recipients",
                   ylabel="Frequency",
                   bins=bins,
                   fig_number=fig_number)
