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
