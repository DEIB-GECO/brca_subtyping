import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Input, Lambda, BatchNormalization, concatenate
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from time import time
from tensorflow.python.keras.callbacks import TensorBoard

from IPython.display import SVG

from tensorflow.keras import backend as K


class BaseVAE():
	"""docstring for BaseVAE"""
	def __init__(self):
		pass

	def initialize_model(self):
		"""
		Helper function to initialize the models
		"""
		self._build_encoder_layers()
		self._build_decoder_layers()
		self._compile_vae()
		self._compile_encoder_decoder()

	def sampling(self, args):

		"""Reparameterization trick by sampling from an isotropic unit Gaussian.
		# Arguments
		args (tensor): mean and log of variance of Q(z|X)
		# Returns
		z (tensor): sampled latent vector
		"""

		z_mean, z_log_var = args
		batch = K.shape(z_mean)[0]
		dim = K.int_shape(z_mean)[1]
		# by default, random_normal has mean = 0 and std = 1.0
		epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
		
		return z_mean + K.exp(0.5 * z_log_var) * epsilon

	def vae_loss(self, y_true, y_pred):

		# E[log P(X|z)]
		if self.rec_loss == "binary_crossentropy":
			reconstruction_loss = self.original_dim * binary_crossentropy(y_true, y_pred) # because it returns the mean cross-entropy
		elif self.rec_loss == "mse":
			reconstruction_loss = self.original_dim * mse(y_true, y_pred) # because it returns the mean cross-entropy

		# D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
		kl_loss = -0.5 * K.sum(1. + self.z_log_var_encoded - K.exp(self.z_log_var_encoded) - K.square(self.z_mean_encoded), axis=1)

		return K.mean(reconstruction_loss + kl_loss)


	def save_model(self, stacked=False):
		if(not stacked):
			model_name = "../models/rna_vae_{}_hidden_{}_emb_{}_drop_in_{}_drop_hidden_noise_input_{}.h5".format(self.intermediate_dim, self.latent_dim, self.dropout_rate_input, self.dropout_rate_hidden, self.noise_input)
		else:
			model_name = "../models/rna_vae_{}_hidden_{}_emb_{}_drop_in_{}_drop_hidden_noise_input_{}_stacked_classifier.h5".format(self.intermediate_dim, self.latent_dim, self.dropout_rate_input, self.dropout_rate_hidden, self.noise_input)
		self.vae.save_weights(model_name)

	def load_model(self, filename):
		self.vae.load_weights(filename)
