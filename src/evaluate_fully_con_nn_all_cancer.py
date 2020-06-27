import pandas as pd
import numpy as np

import sys

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
from time import time

from IPython.display import SVG

from tensorflow.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

#Limit number of cores on Keras
parallelization_factor = 5

sess = tf.Session(config=
    tf.ConfigProto(
        inter_op_parallelism_threads=parallelization_factor,
               intra_op_parallelism_threads=parallelization_factor,
#                    device_count = {'CPU': parallelization_factor},
))


dropout_input = 0.4
dropout_hidden = 0.8
hidden_dim_1 = 300
hidden_dim_2 = 100
epochs = 100
batch_size = 50
learning_rate = 0.001

###############
## Load Data ##
###############

X_all_train = pd.read_pickle("../data/tcga_raw_no_brca_log_row_normalized_ff_all_cancer.pkl")


y_all_train = X_all_train["tumor_type"]

X_all_train.drop(['tumor_type'], axis="columns", inplace=True)

X_brca_test= pd.read_pickle("../data/tcga_brca_raw_19036_row_log_norm_train.pkl")
y_brca_test = X_brca_test["Ciriello_subtype"]
X_brca_test.drop(['tcga_id', 'Ciriello_subtype', 'sample_id', 'cancer_type'], axis="columns", inplace=True)

#X_brca_test= pd.read_pickle("../data/tcga_brca_raw_19036_row_log_norm_test.pkl")
#y_brca_test = X_brca_test["subtype"]
#X_brca_test.drop(['tcga_id', 'subtype', 'sample_id', 'cancer_type'], axis="columns", inplace=True)


scores = []
classify_df = pd.DataFrame(columns=["accuracy"])


scaler = MinMaxScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_all_train), columns=X_all_train.columns)
X_test = pd.DataFrame(scaler.transform(X_brca_test), columns=X_brca_test.columns)

enc = OneHotEncoder(sparse=False)
y_labels_train = pd.DataFrame(enc.fit_transform(y_all_train.values.reshape(-1, 1)))
y_labels_test = pd.DataFrame(enc.fit_transform(y_brca_test.values.reshape(-1, 1)))

y_labels_train["32"]=0.0
y_labels_train["33"]=0.0
y_labels_train["34"]=0.0
y_labels_train["35"]=0.0
y_labels_train["36"]=0.0
print(y_labels_train.head())


# Insert 0 to the places where the output for subtypes is
for j in range(len(np.unique(y_all_train.values))):
        if j not in (32,33,34,35,36): #subtypes
            y_labels_test.insert(j, "Dummy"+str(j), 0.0)
            

print(y_labels_test.head())
#sys.exit()

X_train_train, X_train_val, y_labels_train_train, y_labels_train_val = train_test_split(X_train, y_labels_train, test_size=0.1, stratify=y_all_train, random_state=42)

print(y_labels_train.shape)
print(y_labels_train_train.shape)
print(y_labels_train_val.shape)
#sys.exit()

#subtypes on the one-hot encoding are 2, 11, 16, 17, 22

inputs = Input(shape=(X_train.shape[1], ), name="encoder_input")
dropout_in = Dropout(rate=dropout_input)(inputs)
hidden1_dense = Dense(hidden_dim_1)(dropout_in)
hidden1_batchnorm = BatchNormalization()(hidden1_dense)
hidden1_encoded = Activation("relu")(hidden1_batchnorm)
dropout_hidden1 = Dropout(rate=dropout_hidden)(hidden1_encoded)
hidden2_dense = Dense(hidden_dim_2)(dropout_hidden1)
hidden2_batchnorm = BatchNormalization()(hidden2_dense)
hidden2_encoded = Activation("relu")(hidden2_batchnorm)
outLayer = Dense(37, activation="softmax")
out = outLayer(hidden2_encoded)
model = Model(inputs, out, name="fully_con_nn")

adam = optimizers.Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

print(y_labels_test)
#sys.exit()

model.fit(x=X_train_train, 
			y=y_labels_train_train,
			shuffle=True,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)],
			validation_data=(X_train_val, y_labels_train_val))


new_weights = np.empty([100,37])
biases = outLayer.get_weights()[1]
for i in range(len(outLayer.get_weights()[0])):
	weights = outLayer.get_weights()[0][i]
	for j in range(len(weights)):
		if j not in (32,33,34,35,36): #subtypes
			weights[j]=0
	new_weights[i] = weights


outLayer.set_weights([new_weights,biases])

score = model.evaluate(X_test, y_labels_test)
print(model.predict(X_test).argmax(axis=1))
conf_matrix = pd.DataFrame(confusion_matrix(y_labels_test.values.argmax(axis=1), model.predict(X_test).argmax(axis=1)))

conf_matrix.to_csv("../results/fully_con/all_cancer/feed_forward_300_100_conf_matrix_test.csv")

classify_df = classify_df.append({"accuracy":score[1]}, ignore_index=True)
print(score)
scores.append(score[1])
	

print('Result: {}'.format(scores))
print('Average accuracy: {}'.format(np.mean(scores)))

classify_df = classify_df.assign(hidden_1=hidden_dim_1)
classify_df = classify_df.assign(hidden_2=hidden_dim_2)
classify_df = classify_df.assign(batch_size=batch_size)
classify_df = classify_df.assign(epochs_vae=epochs)
classify_df = classify_df.assign(learning_rate=learning_rate)
classify_df = classify_df.assign(dropout_input=dropout_input)
classify_df = classify_df.assign(dropout_hidden=dropout_hidden)

output_filename="../results/fully_con/all_cancer/{}_hidden_{}_emb_tcga_classifier_dropout_{}_in_{}_hidden_test_set_final.csv".format(hidden_dim_1, hidden_dim_2, dropout_input, dropout_hidden)


classify_df.to_csv(output_filename, sep=',')

