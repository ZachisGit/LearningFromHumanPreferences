import numpy as np

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from datetime import datetime
import os

class RLModel(object):
	def __init__(self,obs_size,action_size,datetime_str,layer_sizes=[12,6,8]):
		self.obs_size = obs_size
		self.action_size = action_size
		self.layer_sizes = layer_sizes
		self.lr = 0.00025
		self.lr_decay = 0.000

		self.create_model()

		self.datetime_str = datetime_str
		self.checkpoint_path = "./human_critic/rl_model/"+self.datetime_str+"/rl_keras_"+self.datetime_str+".h5"


	# Helper Functions
	def lrelu(self,x, leak=0.2, name="lrelu"):
		with tf.variable_scope(name):
			f1 = 0.5 * (1+ leak)
			f2 = 0.5 * (1-leak)
			return f1*x+f2*abs(x)

	def linear(self,x,name="linear"):
		with tf.variable_scope(name):
			return x

	def create_network(self):
		model = Sequential()
		for i in range(len(self.layer_sizes)):
			if i == 0:
				model.add(Dense(self.layer_sizes[i],input_dim=self.obs_size,init="uniform",activation="relu"))
			elif i +1 == len(self.layer_sizes):
				model.add(Dense(self.action_size,init="uniform",activation="linear"))
			else:
				model.add(Dense(self.layer_sizes[i],init="uniform",activation="relu"))

		model.compile(loss="mean_squared_error",optimizer="adam")

		return model

	def loss(self,std_loss=True):
		if std_loss:
			return tf.reduce_mean(tf.square(self.target-self.prediction))
		else:
			return 1.0

	def create_model(self):
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.model = self.create_network();

	def run(self,obs,sess):
		with self.graph.as_default():
			pred = self.model.predict(obs)

		return pred

	def train(self,batch,sess):
		with self.graph.as_default():
			self.model.fit(batch[0],batch[1],nb_epoch=1,batch_size=len(batch[0]),verbose=0)

	def save(self):		
		if not os.path.isdir(os.path.dirname(self.checkpoint_path)):
			os.mkdir(os.path.dirname(self.checkpoint_path))
		with self.graph.as_default():
			self.model.save_weights(self.checkpoint_path)

	def load(self,datetime_str):
		with self.graph.as_default():
			self.model.load_weights("./human_critic/rl_model/"+datetime_str+"/rl_keras_"+datetime_str+".h5")
