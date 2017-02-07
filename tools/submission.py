import time
import os.path as op


def save_submission(df_submission, algo="Random Forest", member="Zac"):
    '''Saves the submission dataframe in the correct format.
        Arguments:
            - df_submission (pd dataframe): the submission dataframe.
            - algo (str): the algo you used.
            - member (str): Who are you ?
        Output:
            - str: the name of the file in which the dataframe is saved.
    '''
    date = str(time.time())
    submission_name = "_".join([algo, member, date])
    submission_name = "{}.csv".format(submission_name)
    df_submission["recipients"].to_csv(op.join("submissions", submission_name),
                                       header=True)
    return submission_name
