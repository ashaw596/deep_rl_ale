'''
ExperienceMemory is a class for experience replay.  
It stores experience samples and samples minibatches for training.
'''

import numpy as np
import random
import math


class ExperienceMemory:

	def __init__(self, args, num_actions):
		''' Initialize emtpy experience dataset. '''

		# params
		self.capacity = args.memory_capacity
		self.history_length = args.history_length
		self.batch_size = args.batch_size
		self.num_actions = num_actions
		self.screen_dims = args.screen_dims
		self.priority_replay = args.priority_replay

		# initialize dataset
		self.observations = np.zeros((self.capacity, self.screen_dims[0], self.screen_dims[1]), dtype=np.uint8)
		self.actions = np.zeros(self.capacity, dtype=np.uint8)
		self.rewards = np.zeros(self.capacity, dtype=np.integer)
		self.terminals = np.zeros(self.capacity, dtype=np.bool)
		self.real_discounted_reward = np.zeros(self.capacity, dtype=np.float32)
		self.is_processed = np.zeros(self.capacity, dtype=np.bool)

		#self.min_real_discounted_reward = np.empty(self.capacity, dtype=np.float32)
		#self.max_real_discounted_reward = np.empty(self.capacity, dtype=np.float32)

		self.size = 0
		self.current = 0
		self.episode_length = 0

		self.discount_factor = 0.99

		self.td_error = np.zeros(self.capacity, dtype=np.float32)


	def add(self, obs, act, reward, terminal):
		''' Add experience to dataset.

		Args:
			obs: single observation frame
			act: action taken
			reward: reward
			terminal: is this a terminal state?
		'''

		self.observations[self.current] = obs
		self.actions[self.current] = act
		self.rewards[self.current] = reward
		self.terminals[self.current] = terminal

		self.current = (self.current + 1) % self.capacity

		self.size = min(self.size + 1, self.capacity)


		self.episode_length += 1
		if terminal:
			self.calc_real_discounted_reward(0)

	def calc_real_discounted_reward(self, end_reward = 0):
		#TODO
		last = end_reward
		#maxReward = end_reward
		#minReward = end_reward
		for i in range(self.episode_length):
			current_index = (self.current - 1 - i) % self.capacity
			self.real_discounted_reward[current_index] = self.rewards[current_index] + self.discount_factor*last
			last = self.real_discounted_reward[current_index]
			self.is_processed[current_index] = True
			#maxReward = max(maxReward, last)
			#minReward = min(minReward, last)

		#for i in range(self.episode_length):
		#	current_index = (self.current - 1 - i) % self.capacity
			#self.min_real_discounted_reward[current_index] = minReward
			#self.max_real_discounted_reward[current_index] = maxReward

		self.episode_length = 0

	def reset_episode_length(self):
		self.episode_length = 0


	def get_state(self, indices):
		''' Return the observation sequence that ends at index 

		Args:
			indices: list of last observations in sequences
		'''
		state = np.empty((len(indices), self.screen_dims[0], self.screen_dims[1], self.history_length))
		count = 0

		for index in indices:
			frame_slice = np.arange(index - self.history_length + 1, (index + 1)) % self.capacity
			state[count] = np.transpose(np.take(self.observations, frame_slice, axis=0), [1,2,0])
			count += 1
		return state


	def get_current_state(self):
		'''  Return most recent observation sequence '''

		return self.get_state([(self.current-1)%self.capacity])


	def get_batch(self, inference_function = None):
		''' Sample minibatch of experiences for training '''
		K = 4
		alpha = 0.6
		samples = [] # indices of the end of each sample
		if self.priority_replay:
			probs = np.clip(self.td_error, 0, 1)[:self.size] + 0.001
			probs = np.power(probs, alpha)
			probs /= np.sum(probs)
			indexes = np.random.choice(self.size, size=self.batch_size*2, p=probs)
			for index in indexes:
				if len(samples) > self.batch_size:
					break 
				if self.terminals[(index - self.history_length):index].any() or \
					(index<self.current-self.size+self.history_length and index>=self.current-self.size):
					continue
				else:
					samples.append(index)
		else:
			while len(samples) < self.batch_size:

				if self.size < self.capacity:  # make this better
					index = random.randrange(self.history_length, self.current)
				else:
					# make sure state from index doesn't overlap with current's gap
					index = (self.current + random.randrange(self.history_length, self.size)) % self.capacity
				# make sure no terminal observations are in any state other than the last state
				if self.terminals[(index - self.history_length):index].any():
					continue
				else:
					samples.append(index)

		max_Ls = np.full(len(samples), float("-inf"), dtype=np.float32)

		if inference_function:
			indexes = []
			startI = []
			endI = []
			estimates = None
		
			for index in samples:

				#print "index" + str(index)

				#print"inference_function"
				#real_discounted_reward = []
				startI.append(len(indexes))
				i=0
				while i<K+1:
					current = (index + i)%self.capacity
					if not self.is_processed[current] or self.terminals[current]:
						break
					if i>=1:
						indexes.append(current)
					i+=1
				endI.append(len(indexes))

			estimates = inference_function(self.get_state(indexes))

			
			l_index = 0
			for start,end in zip(startI,endI):
				reward = self.rewards[samples[l_index]]
				max_L = float("-inf")
				assert end>=start
				for i, index in enumerate(range(start,end)):
					reward += math.pow(self.discount_factor, i + 1) * self.rewards[indexes[i]]
					L = reward + math.pow(self.discount_factor, i + 2) * np.amax(estimates[index])
					max_L = max(max_L, L)

				max_Ls[l_index] = max(max_L, self.real_discounted_reward[samples[l_index]])
				l_index += 1




		min_us = np.full(len(samples), float("inf"), dtype=np.float32)


		if inference_function:
			indexes = []
			startI = []
			endI = []
			estimates = None
		
			for index in samples:
				startI.append(len(indexes))
				i=0
				while i<K+1:
					current = (index - i)%self.capacity
					if not self.is_processed[current] or self.terminals[current]:
						break
					if i>=1:
						indexes.append(current)
					i+=1
				endI.append(len(indexes))

			estimates = inference_function(self.get_state(indexes))

			
			l_index = 0
			for start,end in zip(startI,endI):
				reward = 0
				min_u = float("inf")
				assert end>=start
				for i, index in enumerate(range(start,end)):
					reward += math.pow(self.discount_factor, -(i+1)) * self.rewards[indexes[i]]
					u = math.pow(self.discount_factor, -(i+1)) * estimates[index][self.actions[indexes[i]]] - reward
					min_u = min(min_u, u)

				min_us[l_index] = min_u
				l_index += 1


		# endwhile
		samples = np.asarray(samples)

		# create batch
		o1 = self.get_state((samples - 1) % self.capacity)
		a = np.eye(self.num_actions)[self.actions[samples]] # convert actions to one-hot matrix
		r = self.rewards[samples]
		o2 = self.get_state(samples)
		t = self.terminals[samples].astype(int)
		#min_dr = self.min_real_discounted_reward[samples]
		#max_dr = self.max_real_discounted_reward[samples]

		return [samples, o1, a, r, o2, t, max_Ls, min_us]
