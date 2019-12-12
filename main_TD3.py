import numpy as np
import torch
import gym
import argparse
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bc_gym_planning_env.envs.base import spaces
from bc_gym_planning_env.envs.base.action import Action
from bc_gym_planning_env.envs.egocentric import EgocentricCostmap
from bc_gym_planning_env.envs.base.params import EnvParams
from bc_gym_planning_env.robot_models.standard_robot_names_examples import StandardRobotExamples
from bc_gym_planning_env.envs.mini_env import RandomMiniEnv
from bc_gym_planning_env.envs.synth_turn_env import RandomAisleTurnEnv
from gym_wrapper import bc_gym_wrapper
import math
from collections import OrderedDict
from torch.distributions import Normal
import matplotlib
import matplotlib.pyplot as plt
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer(object):
	def __init__(self, init_size, max_size=int(1e5)):
		self.max_size = max_size
		self.init_size = init_size
		self.ptr = 0
		self.size = 0
		self.state = []
		self.action = []
		self.next_state = []
		self.reward = []
		self.not_done = []

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done):
		if self.size < self.max_size: 
			self.state.append(state)
			self.action.append(action)
			self.next_state.append(next_state)
			self.reward.append(reward)
			self.not_done.append(1.0 - done)
		else:
			self.state[self.ptr] = state
			self.action[self.ptr] = action
			self.next_state[self.ptr] = next_state
			self.reward[self.ptr] = reward
			self.not_done[self.ptr] = 1.0 - done
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = [int(i) for i in list(np.random.randint(0, self.size, size=batch_size))]


		return (
			torch.stack([self.state[i] for i in ind]).to(self.device),
			torch.stack([self.action[i] for i in ind]).to(self.device),
			torch.stack([self.next_state[i] for i in ind]).to(self.device),
			torch.stack([torch.FloatTensor([self.reward[i]]) for i in ind]).to(self.device),
			torch.stack([torch.FloatTensor([self.not_done[i]]) for i in ind]).to(self.device)
		)





#add normalization option
def choose_env(test=False, normalization=True):
	seeds = [5, 44, 122, 134, 405, 587, 1401, 1408, 1693, 1796]    
	validation_seeds = [2262, 2302, 4151, 2480, 2628]
	if test: 
		rand_index = np.random.randint(low=0, high = len(validation_seeds), size=1).item()
		seed =  validation_seeds[rand_index]
	else:
		rand_index = np.random.randint(low=0, high = len(seeds), size=1).item()
		seed =  seeds[rand_index]
	max_steps = 300
	env = RandomMiniEnv
	env_param = EnvParams(iteration_timeout=max_steps,
							goal_ang_dist=np.pi/8,
							goal_spat_dist=1,
							robot_name=StandardRobotExamples.INDUSTRIAL_TRICYCLE_V1)

	env = EgocentricCostmap(env(params=env_param, 
    						turn_off_obstacles=False,
 							draw_new_turn_on_reset=False,
							seed=seed))	
	env = bc_gym_wrapper(env, normalize=normalization)
	return env, seed


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		self.tanh = nn.Tanh()
		self.apply(weights_init_)

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.tanh(self.l3(a))
    
class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-3)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-3)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()



	def train(self, replay_buffer, batch_size=100):
		self.total_it += 1
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			noise = torch.randn_like(action) * self.policy_noise
			noise = noise.clamp(-self.noise_clip, self.noise_clip)
			next_action = (
                self.actor_target(next_state) + noise
			).clamp(-1, 1)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		actor_loss = None
		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		return critic_loss, actor_loss


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))



# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, num_envs=1):
	episode_rewards = []
	for _ in range(num_envs):
		env, seed = choose_env(test=True)
		avg_reward = 0.
		state, done = env.reset(), False
		while not done:
			action = policy.actor(torch.FloatTensor(state).to(device)).cpu().detach().numpy()
			next_state, reward, done, _  = env.step(action)
			avg_reward += reward
			state = next_state
		episode_rewards.append(avg_reward)
	
	average_reward = sum(episode_rewards)

	print("---------------------------------------")
	print(f"Evaluation over {seed} environments: {average_reward:.3f}")
	print("---------------------------------------")
	return average_reward

def smooth_reward_curve(x, y):
	halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution	    
	k = halfwidth
	xsmoo = x
	ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='same')
	return xsmoo, ysmoo

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3)
	parser.add_argument("--env", default="RandomMiniEnv")           # BC Gym 
	parser.add_argument("--seed", default=0, type=int)              # Set BC gym seed value
	parser.add_argument("--start_timesteps", default=9e4, type=int) # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=1e-2)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", default=True)               # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()
	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, save_model: {args.save_model}")
	print("---------------------------------------")
	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")
        
	max_steps = 300
	env, seed = choose_env()
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = 1

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	# Target policy smoothing is scaled wrt the action scale
	kwargs["policy_noise"] = args.policy_noise * max_action
	kwargs["noise_clip"] = args.noise_clip * max_action
	kwargs["policy_freq"] = args.policy_freq
	policy = TD3(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = ReplayBuffer(init_size=int(args.start_timesteps))
	# Evaluate untrained policy
	state = env.reset()
	done = False
	episode_timesteps = 0
	print("Filling Replay Buffer")
	for t in range(int(args.start_timesteps)):
		episode_timesteps+=1
		action = env.action_space.sample().clip(-1,1)
		next_state, reward, done, _ = env.step(action)
		done_bool = float(done) if episode_timesteps < max_steps else 0
		replay_buffer.add(torch.FloatTensor(copy.deepcopy(state)), torch.FloatTensor(copy.deepcopy(action)), torch.FloatTensor(copy.deepcopy(next_state)), reward, done_bool)
		state = next_state
		if done:
			env, seed = choose_env()
			state = env.reset()
			done = False
			episode_timesteps = 0

	print("done filling replay buffer. Starting training")
	env, seed = choose_env()
	state = env.reset()
	done = False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	actions_taken = []
	critic_losses, actor_losses, rewards, evaluations = [], [], [], []
	for t in range(int(args.max_timesteps)):
		episode_timesteps += 1
		# Select action according to policy
		action = policy.actor(torch.FloatTensor(state).to(device))
		# Perform action
		action = (action.cpu().detach().numpy()+ np.random.normal(0, 0.2, size=action_dim)).clip(-1, 1)
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < max_steps else 0
		# Store data in replay buffer
		replay_buffer.add(torch.FloatTensor(copy.deepcopy(state)), torch.FloatTensor(copy.deepcopy(action)), torch.FloatTensor(copy.deepcopy(next_state)), reward, done_bool)
		state = next_state
		episode_reward += reward
		# Train agent after collecting sufficient data		
		critic_loss, actor_loss = policy.train(replay_buffer, 256)
		critic_losses.append(critic_loss)
		if actor_loss: 
			actor_losses.append(actor_loss)
		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Environment seed: {seed} Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			env, seed = choose_env()
			state, done = env.reset(), False
			rewards.append(episode_reward)
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
           
		# Evaluate episode
		if (t + 1) % 50000 == 0:
			evaluations.append(eval_policy(policy))
			np.save(f"./results/{file_name}_{t}_eval", evaluations)
			np.save(f"./results/{file_name}_{t}_reward", rewards)
			policy.save(f"./models/{file_name}")

		if (t) % 100000 == 0:
			policy.save(f"./models/{file_name}_{t}")
        

plt.figure()
indices_critic = [i + 1 for i in range(len(critic_losses))]
plt.plot(indices_critic, critic_losses)
plt.xlabel('number of timesteps')
plt.ylabel('critic loss')
plt.title('critic loss over training')
plt.savefig('critic_loss.png')  
plt.close()

plt.figure()
indices_actor = [i + 1 for i in range(len(actor_losses))]
plt.plot(indices_actor, actor_losses)
plt.xlabel('number of timesteps')
plt.ylabel('actor loss')
plt.title('actor loss over training')
plt.savefig('actor_loss.png')  
plt.close()

plt.figure()
indices_rewards = [i + 1 for i in range(len(rewards))]
plt.plot(indices_rewards, rewards)
plt.xlabel('number of episodes')
plt.ylabel('rewards')
plt.title('reward per episode')
plt.savefig('rewards.png')  
plt.close()


plt.figure()
indices_rewards = [i + 1 for i in range(len(rewards))]
indices_reward, smoothed_rewards = smooth_reward_curve(indices_rewards, rewards)
plt.plot(indices_rewards, smoothed_rewards)
plt.xlabel('number of episodes')
plt.ylabel('smoothed rewards')
plt.title('smoothed reward per episode')
plt.savefig('smoothed_rewards.png')  
plt.close()


plt.figure()
indices_eval = [i + 1 for i in range(len(evaluations))]
plt.plot(indices_eval, evaluations)
plt.xlabel('number of evaluation')
plt.ylabel('rewards')
plt.title('evaluation rewards per episode')
plt.savefig('eval_rewards.png')  
plt.close()

np.save(f"./results/{file_name}_final_reward_final", rewards)
np.save(f"./results/{file_name}_actor_losses_final", actor_losses)
np.save(f"./results/{file_name}_critic_losses_final", critic_losses)
np.save(f"./results/{file_name}_smoothed_reward_final", smoothed_rewards)
policy.save(f"./models/{file_name}_final")
