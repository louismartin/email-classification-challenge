import matplotlib.pyplot as plt


def print_email_by_id(emails, mid):
    '''Uses the dataframe emails to print the email given by its mid in a
    classic format.
        Arguments:
            - emails (pd dataframe): the pd dataframe containing the emails
            - mid (int)
    '''
    email_row = emails.loc[mid]
    email_dict = email_row.to_dict()
    email_content = "From: {sender} \nTo: {recipients}\nOn: {date} \n\
Body:\n {body}".format(**email_dict)
    print(email_content)


def emails_sent_distribution(email_senders, fig_number=None):
    '''Plots the distribution of number of emails sent by sender.
        Arguments:
            - email_senders (pd dataframe): the pd dataframe containing the
            emails sent for every sender.
            - fig_number (int): the index of the plot figure.
    '''
    n_emails_sent = email_senders["mids"].str.split().str.len()
    if fig_number:
        plt.figure(fig_number)
    else:
        plt.figure()
    plt.hist(n_emails_sent.as_matrix(), bins=20, normed=True)
    plt.title("Number of email sent distribution")
    plt.xlabel("Number of email sent")
    plt.ylabel("Frequency")
    plt.show()
