import glob
import os
import os.path as op
import time

from keras.callbacks import Callback
import matplotlib.pyplot as plt

from tools.evaluation import top_emails, evaluate


def data_generator(df_train, vectorizer, batch_size=32):
    """Online training data generator"""
    while True:
        df_batch = df_train.sample(n=batch_size)
        X_batch = vectorizer.vectorize_input(df_batch["clean_body"])
        Y_batch = vectorizer.vectorize_output(df_batch["recipients"])
        yield (X_batch, Y_batch)


class EvaluateAndSave(Callback):
    """
    Keras callback for evaluating the model and saving the weights at every
    epoch end.
    """
    def __init__(self, X_test, recipients_map, ground_truth, batch_size=8):
        self.X_test = X_test
        self.recipients_map = recipients_map
        self.ground_truth = ground_truth
        self.batch_size = batch_size
        self.start_time = int(time.time())
        self.save_folder = op.join("models", "nnet_{}".format(self.start_time))
        os.makedirs(self.save_folder)
        self.logs = {"precision": [],
                     "path": []}

    def on_epoch_end(self, epoch, logs=None):
        Y_pred = self.model.predict(self.X_test, batch_size=self.batch_size)
        predictions = top_emails(Y_pred, self.recipients_map)
        precision = evaluate(predictions, self.ground_truth)
        print("*** Precision: {prec:.3f} ***\n".format(prec=precision))
        # Save everything
        self.logs["precision"].append(precision)
        if precision == max(self.logs["precision"]):
            # Remove other saved models
            paths = glob.glob(op.join(self.save_folder, "*.hdf5"))
            for path in paths:
                os.remove(path)

            # Save new model
            path = op.join(self.save_folder,
                           "nnet_{prec:.2f}.hdf5".format(prec=precision))
            self.model.save_weights(path)

        # Save and plot precisions
        with open(op.join(self.save_folder, "logs.csv"), "w") as fp:
            for prec in self.logs["precision"]:
                fp.write("{prec:.4f}, ".format(prec=prec))
        plt.plot(self.logs["precision"])
        plt.savefig(op.join(self.save_folder, "precision.png"))
