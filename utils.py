import numpy as np
import tensorflow as tf

class MiniBatch(object):
	def __init__(self,obs_size,actions,batch_size=10):
		self.sample_obs = []
		self.sample_ys = []
		self.gamma = 0.99
		self.obs_size = obs_size
		self.actions = actions
		self.batch_size = batch_size
		self.capacity = 100000
		self.samples = []

	def _add_sample(self,obs,q,max_q,reward,action,done=False):
		self.sample_obs.append(obs)
		
		y = np.copy(q).reshape([self.actions])
		update = reward if done else reward + (max_q*self.gamma)
		y[action] = update
		self.sample_ys.append(y)

		if self.capacity < len(self.sample_obs):
			del self.sample_obs[0]
			del self.sample_ys[0]

	def add_sample(self,obs,future_obs,reward,action,done=False):
		self.samples.append([obs,future_obs,reward,action,done])
		if len(self.samples) > self.capacity:
			del self.samples[0]

	def _get_batch(self):
		r = np.arange(0,len(self.sample_obs))
		np.random.shuffle(r)
		obs = np.asarray(self.sample_obs)[r[:min(self.batch_size,len(self.sample_obs))]]
		ys = np.asarray(self.sample_ys)[r[:min(self.batch_size,len(self.sample_ys))]]
		return obs,ys

	def get_batch(self,model,sess):
		r = np.arange(0,len(self.samples))
		np.random.shuffle(r)
		batch_size = int(min(self.batch_size,len(self.samples)))
		batch = np.asarray(self.samples)[r[:batch_size]]

		states = np.array([sample[0] for sample in batch],dtype=np.float32)
		future_states = np.array([(np.zeros(self.obs_size) if sample[4] else sample[1]) for sample in batch],dtype=np.float32)

		q_value_batch = model.run(states,sess)
		future_q_value_batch = model.run(future_states,sess)

		x = np.zeros([batch_size,self.obs_size],dtype=np.float32)
		y = np.zeros([batch_size,self.actions],dtype=np.float32)

		for i in range(batch_size):
			state,future_state,reward,action,done = batch[i]

			q_values = q_value_batch[i]
			
			if done:
				q_values[action] = reward
			else:
				#print future_q_value_batch[i],reward,reward + self.gamma*np.amax(future_q_value_batch[i])
				q_values[action] = reward + self.gamma*np.amax(future_q_value_batch[i])

			x[i] = state
			y[i] = q_values

		return x,y

	def get_batch_hc(self,model,sess,hc_model):
		r = np.arange(0,len(self.samples))
		np.random.shuffle(r)
		batch_size = int(min(self.batch_size,len(self.samples)))
		batch = np.asarray(self.samples)[r[:batch_size]]

		states = np.array([sample[0] for sample in batch],dtype=np.float32)
		future_states = np.array([(np.zeros(self.obs_size) if sample[4] else sample[1]) for sample in batch],dtype=np.float32)

		q_value_batch = model.run(states,sess)
		future_q_value_batch = model.run(future_states,sess)

		x = np.zeros([batch_size,self.obs_size],dtype=np.float32)
		y = np.zeros([batch_size,self.actions],dtype=np.float32)

		for i in range(batch_size):
			state,future_state,_,action,done = batch[i]
			reward = hc_model.predict(np.reshape(future_state,[1,-1]))[0]

			q_values = q_value_batch[i]
			
			if done:
				q_values[action] = reward
			else:
				#print future_q_value_batch[i],reward,reward + self.gamma*np.amax(future_q_value_batch[i])
				q_values[action] = reward + self.gamma*np.amax(future_q_value_batch[i])

			x[i] = state
			y[i] = q_values

		return x,y

	def trim(self):
		self.sample_obs = []
		self.sample_ys = []