import glob
import os
import os.path as op
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint

from tools.data_cleansing import get_clean_df_train, get_clean_df_test
from tools.data_handling import enrich_emails, unique_recipients, address_book
from tools.features import bag_of_words, bag_of_emails
from tools.evaluation import top_emails, evaluate

overwrite = False
df_emails = enrich_emails(overwrite=overwrite)
df_train = get_clean_df_train(ratio=0.9, overwrite=overwrite)
df_test = get_clean_df_test(overwrite=overwrite).head(100)
df_train = df_train.fillna("")
df_test = df_test.fillna("")

bow_vectorizer = bag_of_words(df_train["body"], min_df=5, max_features=None)
boe_vectorizer = bag_of_emails(
    df_train["recipients"], unique_recipients(df_train),
    min_df=5, max_features=None
    )

X_test = bow_vectorizer.transform(df_test["body"]).toarray()

n_features = len(bow_vectorizer.vocabulary_)
n_outputs = len(boe_vectorizer.vocabulary_)
print("Features: {} - Outputs: {}".format(n_features, n_outputs))

batch_size = 32
samples_per_epoch = 1000
nb_epoch = 20

inputs = Input(shape=(n_features,))

# x = Dense(1024, activation='relu')(inputs)
predictions = Dense(n_outputs, activation='softmax')(inputs)

model = Model(input=inputs, output=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


def data_generator(df_train, bow_vectorizer, boe_vectorizer, batch_size=32):
    """Online training data generator"""
    while True:
        df_batch = df_train.sample(n=batch_size)
        X_batch = bow_vectorizer.transform(df_batch["body"]).toarray()
        Y_batch = boe_vectorizer.transform(df_batch["recipients"]).toarray()
        yield (X_batch, Y_batch)


class EvaluateAndSave(Callback):
    def __init__(self, X_test, recipients_map, ground_truth):
        self.X_test = X_test
        self.recipients_map = recipients_map
        self.ground_truth = ground_truth
        self.start_time = int(time.time())
        self.save_folder = op.join("models", "nnet_{}".format(self.start_time))
        os.makedirs(self.save_folder)
        self.logs = {"precision": [],
                     "path": []}

    def on_epoch_end(self, epoch, logs=None):
        # TODO: fix global variables used inside local method
        Y_pred = self.model.predict(self.X_test, batch_size=batch_size)
        predictions = top_emails(Y_pred, self.recipients_map)
        precision = evaluate(predictions, ground_truth)
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


# Create callbacks
callbacks = []
filepath = "models/weights_{epoch:02d}.hdf5"
model_checkpoint = ModelCheckpoint(filepath,
                                   save_best_only=False,
                                   save_weights_only=True)
recipients_map = np.array(boe_vectorizer.get_feature_names())
ground_truth = df_test["recipients"].str.split(expand=True).as_matrix()
evaluate_and_save = EvaluateAndSave(X_test, recipients_map, ground_truth)
callbacks.append(evaluate_and_save)

generator = data_generator(
    df_train, bow_vectorizer, boe_vectorizer, batch_size=batch_size
    )

model.fit_generator(
    generator,
    samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
    callbacks=callbacks, nb_worker=1)
