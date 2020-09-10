import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, classification_report
from time import time

from mixup_generator import MixupGenerator

import sys

import keras_metrics

from IPython.display import SVG

from tensorflow.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

#Limit number of cores on Keras
parallelization_factor = 8

sess = tf.Session(config=
    tf.ConfigProto(
        inter_op_parallelism_threads=parallelization_factor,
               intra_op_parallelism_threads=parallelization_factor,
#                    device_count = {'CPU': parallelization_factor},
))

dropout_input = 0.2
dropout_hidden = 0
hidden_dim_1 = 100
hidden_dim_2 = 20
epochs = 100
batch_size = 64
learning_rate = 0.01

flag_mixup=False

###############
## Load Data ##
###############

#X_brca_train = pd.read_pickle("../data/hybrids/tcga_brca_cna_rna_meta_train.pkl")

#X_brca_train = pd.read_pickle("../data/tcga_brca_raw_19036_row_log_norm_train.pkl")
#X_brca_train = pd.read_csv("../data/miRNA_filtered_norm_scaled_train.csv")
#X_brca_train = pd.read_pickle("../data/cna_brca_train_0.8_threshold_0.6_chrX.pkl")
#X_brca_train = pd.read_pickle("../data/hybrids/tcga_brca_mirna_rna_meta_train.pkl")
X_brca_train = pd.read_pickle("../data/hybrids/tcga_brca_cna_mirna_meta_train.pkl")
#X_brca_train = pd.read_pickle("../data/hybrids/tcga_brca_cna_rna_meta_train.pkl")
#X_brca_train = pd.read_pickle("../data/hybrids/tcga_brca_cna_mirna_rna_meta_train.pkl")

y_brca_train = X_brca_train["Ciriello_subtype"]

#X_brca_train.drop(['tcga_id', 'Ciriello_subtype', 'sample_id', 'cancer_type'], axis="columns", inplace=True)
X_brca_train.drop(['tcga_id', 'Ciriello_subtype'], axis="columns", inplace=True)
#X_brca_train.drop(['Ciriello_subtype'], axis="columns", inplace=True)




subtypes = ["Basal", "Her2", "LumA", "LumB", "Normal"]

d_rates1 = [0.6]
d_rates2 = [0.8]

for dropout_input in d_rates1:
    for dropout_hidden in d_rates2:

        skf = StratifiedKFold(n_splits=10)
        scores = []
        i=1
        classify_df = pd.DataFrame(columns=["Fold", "accuracy"])
        conf_matrix = np.zeros([5,5])

        for train_index, test_index in skf.split(X_brca_train, y_brca_train):
            print('Fold {} of {}'.format(i, skf.n_splits))

            X_train, X_val = X_brca_train.iloc[train_index], X_brca_train.iloc[test_index]
            y_train, y_val = y_brca_train.iloc[train_index], y_brca_train.iloc[test_index]

            scaler = MinMaxScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

            enc = OneHotEncoder(sparse=False)
            y_labels_train = enc.fit_transform(y_train.values.reshape(-1, 1))
            y_labels_val = pd.DataFrame(enc.fit_transform(y_val.values.reshape(-1, 1)))
            y_labels_val_conf = enc.fit_transform(y_val.values.reshape(-1, 1))

            X_train_train, X_train_val, y_labels_train_train, y_labels_train_val = train_test_split(X_train, y_labels_train, test_size=0.2, stratify=y_train, random_state=42)


        # Only 1 hidden layer

            inputs = Input(shape=(X_train.shape[1], ), name="encoder_input")
            dropout_in = Dropout(rate=dropout_input)(inputs)
            hidden1_dense = Dense(hidden_dim_1)(dropout_in)
            hidden1_batchnorm = BatchNormalization()(hidden1_dense)
            hidden1_encoded = Activation("relu")(hidden1_batchnorm)
            dropout_hidden1 = Dropout(rate=dropout_hidden)(hidden1_encoded)
            hidden2_dense = Dense(hidden_dim_2)(dropout_hidden1)
            hidden2_batchnorm = BatchNormalization()(hidden2_dense)
            hidden2_encoded = Activation("relu")(hidden2_batchnorm)
            out = Dense(5, activation="softmax")(hidden2_encoded)
            #out = Dense(5, activation="softmax")(hidden1_encoded)


            model = Model(inputs, out, name="fully_con_nn")

            adam = optimizers.Adam(lr=learning_rate)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

            if(flag_mixup):
                training_generator = MixupGenerator(X_train_train, y_labels_train_train, batch_size=batch_size, alpha=0.2)()
                validation_generator = MixupGenerator(X_train_val, y_labels_train_val, batch_size=batch_size, alpha=0.2)()
                print("HEREEEEE")
                print(X_train_train.shape)
                print(X_train_val.shape)
                #sys.exit()
                fit_hist = model.fit(x=training_generator, 
                                        epochs=epochs,
                                        steps_per_epoch=X_train_train.shape[0]//64,
                                        callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)],
                                        validation_data=validation_generator,
                                        validation_steps=X_train_val.shape[0])
            else:
                model.fit(x=X_train_train, 
                            y=y_labels_train_train,
                            shuffle=True,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)],
                            validation_data=(X_train_val, y_labels_train_val))
            score = model.evaluate(X_val, y_labels_val)

            report = classification_report(y_labels_val_conf.argmax(axis=1), model.predict(X_val).argmax(axis=1), target_names=subtypes, output_dict=True)    
            classify_df = classify_df.append({"Fold":str(i), "accuracy":score[1], "other_metrics":report}, ignore_index=True)
            print(score)
            scores.append(score[1])

            conf = confusion_matrix(y_labels_val_conf.argmax(axis=1), model.predict(X_val).argmax(axis=1))
            conf_matrix = np.add(conf_matrix, conf)
            i+=1

        print('10-Fold results: {}'.format(scores))
        print('Average accuracy: {}'.format(np.mean(scores)))

        classify_df = classify_df.assign(mean_accuracy=np.mean(scores))
        classify_df = classify_df.assign(hidden_1=hidden_dim_1)
        classify_df = classify_df.assign(hidden_2=hidden_dim_2)
        classify_df = classify_df.assign(batch_size=batch_size)
        classify_df = classify_df.assign(epochs_vae=epochs)
        classify_df = classify_df.assign(learning_rate=learning_rate)
        classify_df = classify_df.assign(dropout_input=dropout_input)
        classify_df = classify_df.assign(dropout_hidden=dropout_hidden)

        output_filename="../results/10-fold/miRNA+cna/FFNN/100_hidden_20_emb/{}_hidden_{}_emb_tcga_classifier_dropout_{}_in_{}_hidden_10fold.csv".format(hidden_dim_1, hidden_dim_2, dropout_input, dropout_hidden)

        conf_filename="../results/10-fold/miRNA+cna/FFNN/100_hidden_20_emb/confusion_matrix/{}_hidden_{}_emb_tcga_classifier_dropout_{}_in_{}_hidden_confusion_matrix_10fold.csv".format(hidden_dim_1, hidden_dim_1, dropout_input, dropout_hidden)

        #output_filename="../results/10-fold/cna+RNA/FFNN/50_hidden_0_emb/{}_hidden_0_emb_tcga_classifier_dropout_{}_10fold_cv.csv".format(hidden_dim_1, dropout_input)

        #conf_filename="../results/10-fold/cna+RNA/FFNN/50_hidden_0_emb/confusion_matrix/{}_hidden_0_emb_tcga_classifier_dropout_{}_cv_confusion_matrix_10fold.csv".format(hidden_dim_1, dropout_input)

        confs_matrix = pd.DataFrame(conf_matrix)
        confs_matrix.to_csv(conf_filename, sep=',')
        classify_df.to_csv(output_filename, sep=',')
        # Put to 0 for next cv iteration
        conf_matrix = np.zeros([5,5])




'''

dropout_input = 0.8
dropout_hidden = 0.4
hidden_dim_1 = 300
hidden_dim_2 = 100
epochs = 100
batch_size = 64
learning_rate = 0.001

###############
## Load Data ##
###############


X_brca_train = pd.read_pickle("../data/hybrids/tcga_brca_cna_mirna_rna_meta_train.pkl")
y_brca_train = X_brca_train["Ciriello_subtype"]
X_brca_train.drop(['Ciriello_subtype', 'tcga_id'], axis="columns", inplace=True)

X_brca_test = pd.read_pickle("../data/hybrids/tcga_brca_cna_mirna_rna_meta_test.pkl")
y_brca_test = X_brca_test["subtype"]
X_brca_test.drop(['subtype', 'tcga_id'], axis="columns", inplace=True)


scores = []
classify_df = pd.DataFrame(columns=["accuracy"])


scaler = MinMaxScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_brca_train), columns=X_brca_train.columns)
X_test = pd.DataFrame(scaler.transform(X_brca_test), columns=X_brca_test.columns)

enc = OneHotEncoder(sparse=False)
y_labels_train = enc.fit_transform(y_brca_train.values.reshape(-1, 1))
y_labels_test = pd.DataFrame(enc.fit_transform(y_brca_test.values.reshape(-1, 1)))

X_train_train, X_train_val, y_labels_train_train, y_labels_train_val = train_test_split(X_train, y_labels_train, test_size=0.1, stratify=y_brca_train, random_state=42)



inputs = Input(shape=(X_train.shape[1], ), name="encoder_input")
dropout_in = Dropout(rate=dropout_input)(inputs)
hidden1_dense = Dense(hidden_dim_1)(dropout_in)
hidden1_batchnorm = BatchNormalization()(hidden1_dense)
hidden1_encoded = Activation("relu")(hidden1_batchnorm)
dropout_hidden1 = Dropout(rate=dropout_hidden)(hidden1_encoded)
hidden2_dense = Dense(hidden_dim_2)(dropout_hidden1)
hidden2_batchnorm = BatchNormalization()(hidden2_dense)
hidden2_encoded = Activation("relu")(hidden2_batchnorm)
out = Dense(5, activation="softmax")(hidden1_encoded)

model = Model(inputs, out, name="fully_con_nn")

adam = optimizers.Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(x=X_train_train, 
			y=y_labels_train_train,
			shuffle=True,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)],
			validation_data=(X_train_val, y_labels_train_val))

score = model.evaluate(X_test, y_labels_test)
conf_matrix = pd.DataFrame(confusion_matrix(y_labels_test.values.argmax(axis=1), model.predict(X_test).argmax(axis=1)))

y_brca_test.to_csv('../ffnn_test_labels.csv')

predictions_ffnn = pd.DataFrame(model.predict(X_test), columns = ['Basal', 'Her2', 'LumA', 'LumB', 'Normal'])
predictions_ffnn.to_csv('../ffnn_all_data_test_predictions.csv')

#conf_matrix.to_csv("../results/fully_con_ieee/rna+miRNA+cna/confusion_matrix/confusion_matrix_text.csv")

classify_df = classify_df.append({"accuracy":score[1]}, ignore_index=True)
print(score)
scores.append(score[1])
	

print('Result: {}'.format(scores))
print('Average accuracy: {}'.format(np.mean(scores)))

classify_df = classify_df.assign(hidden_1=hidden_dim_1)
classify_df = classify_df.assign(batch_size=batch_size)
classify_df = classify_df.assign(epochs_vae=epochs)
classify_df = classify_df.assign(learning_rate=learning_rate)
classify_df = classify_df.assign(dropout_input=dropout_input)



#output_filename="../results/fully_con_ieee/rna+miRNA+cna/{}_hidden_tcga_classifier_dropout_{}_test_set.csv".format(hidden_dim_1, dropout_input)


#classify_df.to_csv(output_filename, sep=',')
'''
