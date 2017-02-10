
import random
import numpy as np
from tqdm import tqdm

class DQNAgent():

	def __init__(self, args, q_network, emulator, experience_memory, num_actions, train_stats):

		self.network = q_network
		self.emulator = emulator
		self.memory = experience_memory
		self.train_stats = train_stats

		self.num_actions = num_actions
		self.history_length = args.history_length
		self.training_frequency = args.training_frequency
		self.random_exploration_length = args.random_exploration_length
		self.initial_exploration_rate = args.initial_exploration_rate
		self.final_exploration_rate = args.final_exploration_rate
		self.final_exploration_frame = args.final_exploration_frame
		self.test_exploration_rate = args.test_exploration_rate
		self.recording_frequency = args.recording_frequency
		self.enable_constraints = args.enable_constraints

		self.exploration_rate = self.initial_exploration_rate
		self.total_steps = 0

		self.test_state = []


	def choose_action(self):

		if random.random() >= self.exploration_rate:
			state = self.memory.get_current_state()
			q_values = self.network.inference(state)
			self.train_stats.add_q_values(q_values)
			return np.argmax(q_values)
		else:
			return random.randrange(self.num_actions)


	def checkGameOver(self):
		if self.emulator.isGameOver():
			self.memory.calc_real_discounted_reward(0)
			initial_state = self.emulator.reset()
			for experience in initial_state:
				self.memory.add(experience[0], experience[1], experience[2], experience[3])
			self.train_stats.add_game()
			return True
		return False


	def run_random_exploration(self):
		print("Running random Exploration")
		with tqdm(total=self.random_exploration_length) as pbar:
			step = 0
			terminal = False
			
			while step<self.random_exploration_length or not terminal:
			
				#print("step:" + str(step))
				state, action, reward, terminal, raw_reward = self.emulator.run_step(random.randrange(self.num_actions))
				self.train_stats.add_reward(raw_reward)
				self.memory.add(state, action, reward, terminal)
				self.checkGameOver()
				self.total_steps += 1
				if (self.total_steps % self.recording_frequency == 0):
					self.train_stats.record(self.total_steps)

				step += 1
				pbar.update(1)
				#if step == self.random_exploration_length-1 and not terminal:
				#	q_values = self.network.inference(self.memory.get_current_state())
				#	self.memory.calc_real_discounted_reward(np.amax(q_values))


	def run_epoch(self, steps, epoch):

		with tqdm(total=steps) as pbar:
			step = 0
			terminal = False
			#TODO: This will infinite loop if game never ends
			while (step < steps or not terminal):
				#if step%1000==0:
				#	print step

				state, action, reward, terminal, raw_reward = self.emulator.run_step(self.choose_action())
				self.memory.add(state, action, reward, terminal)
				self.train_stats.add_reward(raw_reward)
				self.checkGameOver()

				# training
				if self.total_steps % self.training_frequency == 0:
					inference_function = None
					if self.enable_constraints:
						inference_function = lambda s: self.network.target_inference(s)
					states, actions, rewards, next_states, terminals, max_ls, min_us = self.memory.get_batch(inference_function)
					loss = self.network.train(states, actions, rewards, next_states, terminals, max_ls, min_us)
					self.train_stats.add_loss(loss)

				self.total_steps += 1

				if self.total_steps < self.final_exploration_frame:
					self.exploration_rate -= (self.exploration_rate - self.final_exploration_rate) / (self.final_exploration_frame - self.total_steps)

				if self.total_steps % self.recording_frequency == 0:
					self.train_stats.record(self.total_steps)
					self.network.record_params(self.total_steps)

				step += 1
				#print terminal
				#print (step)
				#print (steps)
				pbar.update(1)
		self.memory.reset_episode_length()

	def test_step(self, observation):

		if len(self.test_state) < self.history_length:
			self.test_state.append(observation)

		# choose action
		q_values = None
		action = None
		if random.random() >= self.test_exploration_rate:
			state = np.expand_dims(np.transpose(self.test_state, [1,2,0]), axis=0)
			q_values = self.network.inference(state)
			action = np.argmax(q_values)
		else:
			action = random.randrange(self.num_actions)

		self.test_state.pop(0)
		return [action, q_values]


	def save_model(self, epoch):
		self.network.save_model(epoch)