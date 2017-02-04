import os

import pandas as pd


def save_and_reload_df(func):
    """
    Decorator that saves the dataframe computed by the function
    and loads it if it was already saved.
    """
    def func_wrapper(*args, overwrite=False, **kwargs):
        # Create all the paths and filenames necessary
        data_dir = "data"
        filename = "{}.csv".format(func.__name__)
        csv_path = os.path.join("data", filename)
        # The file already exists so we just read it from disk
        if os.path.exists(csv_path) and not overwrite:
            print("Reading dataframe from {}".format(csv_path))
            df = pd.read_csv(csv_path, index_col=0)
        # Either the file does not exist or we want to compute it again
        else:
            # Make sure the data directory already exists
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            # Compute the new file
            df = func(*args, **kwargs)

            # Back up the old file if it exists
            if os.path.exists(csv_path) and overwrite:
                timestamp = int(time.time())
                backup_filename = "{}_{}.csv".format(func.__name__, timestamp)
                backup_dir = os.path.join(data_dir, "backup")
                backup_path = os.path.join(backup_dir, backup_filename)
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir)
                os.rename(csv_path, backup_path)

            # Save the new file
            df.to_csv(csv_path)
        return df
    return func_wrapper
