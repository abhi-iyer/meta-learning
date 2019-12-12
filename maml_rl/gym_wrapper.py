import numpy as np
import copy
from gym.spaces.box import Box
from bc_gym_planning_env.envs.base import spaces
from bc_gym_planning_env.envs.base.action import Action
from bc_gym_planning_env.envs.mini_env import RandomMiniEnv
from bc_gym_planning_env.envs.egocentric import EgocentricCostmap
from bc_gym_planning_env.envs.base.params import EnvParams
from bc_gym_planning_env.robot_models.standard_robot_names_examples import StandardRobotExamples


def convert_bc_space(space):
    if isinstance(space, spaces.Box):
        return Box(low=space.low, high=space.high)
    else:
        return NotImplementedError


def normalize_angle(theta):
    """
    Normalize angle between +/-pi
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


class bc_gym_wrapper():
    def __init__(self, env, normalize=False):
        self._env = env
        env = env.unwrapped()
        o = env._env._state.robot_state.to_numpy_array()
        # NOTE: These values have been only tested for RandomMiniEnv
        self.y_m = 66.0
        self.x_m = 58.0
        # Generate lines for liDaR
        self.lines = [(self.discrete_line(theta))
                      for theta in np.arange(-np.pi, np.pi, 0.05)]
        # The agent's position, goal position are stacked
        observation_dim = np.shape(o)[0] + 3 + len(self.lines)
        high = np.array([np.inf] * observation_dim)

        self._observation_space = Box(low=-high, high=high)
        if normalize:
            action_dim = self._env.action_space.high.shape[0]
            self.action_high = self._env.action_space.high
            self.action_low = self._env.action_space.low
            high = np.array([1] * action_dim)
            self._action_space = Box(low=-high, high=high)
        else:
            self._action_space = convert_bc_space(self._env.action_space)
        self.normalize = normalize

    def discrete_line(self, theta):
        """
        Generates lines on the image
        """
        if abs(np.tan(theta)) <= self.y_m / self.x_m:
            x = np.arange(self.x_m) * np.cos(theta) / np.abs(np.cos(theta))
            y = np.ceil(np.tan(theta) * x)
            return np.array(x + self.x_m, dtype=int), np.array(y + self.y_m,
                                                               dtype=int)

        y = np.arange(self.y_m) * np.sin(theta) / np.abs(np.sin(theta))
        x = np.ceil(np.tan(np.pi / 2 - theta) * y)
        return np.array(x + self.x_m, dtype=int), np.array(y + self.y_m,
                                                           dtype=int)

    def check_obstacle(self, img, line):
        """
        A function that calculates the distance of the nearest obstacle along a given line
        """
        x, y = line

        line_points = img[y, x]
        point = np.where(np.diff(line_points) > 150)
        if point[0].any():
            distance = point[0][0]
        else:
            distance = np.hypot(y[-1] - self.y_m, x[-1] - self.x_m)
        # normalize the distance
        distance = np.clip(distance, 0, 57) / 57.0
        return distance

    # Setting the observation space
    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _get_current_goal(self):
        goal = self._env.env._env._reward_provider._state.current_goal_pose(
        ).reshape((3, 1))
        return copy.copy(goal)

    def _get_world_size(self):
        world_size = self._env.env._env._state.costmap.world_size()
        return copy.copy(world_size)

    def _get_obs(self, obs):
        """
        Convert the observation from the environment to liDaR representation and normalize the observations
        """
        img = copy.copy(np.squeeze(obs['env']))
        distance = np.expand_dims(np.array(
            [self.check_obstacle(img, line) for line in self.lines]),
                                  axis=-1)
        o = copy.copy(obs['goal_n_state'][3:])
        goal = self._get_current_goal()
        world_size = self._get_world_size()
        goal[:2] = goal[:2] / np.expand_dims(world_size, -1)
        o[:2] = o[:2] / np.expand_dims(world_size, -1)
        o = np.vstack([goal, o, distance])
        return o.reshape((-1, ))

    # Wrap the step command in action
    def step(self, action):
        # If input actions are normalized then scale back to the environment
        if self.normalize:
            action = (action + 1) * (self.action_high -
                                     self.action_low) / 2 + self.action_low
        action = Action(command=np.array(action))
        obs, reward, done, info = self._env.step(action)
       
        # Extract the goal information from the observation variable
        goal_n_obs = self._get_obs(obs)
        info.update(dict(goal_n_state=obs['goal_n_state'][:3]), )

        '''
        for pt in goal_n_obs[6:][3:]:
            if pt <= 1: # critical distance to the osbtacle is 0.25
                reward -= 1.5
            elif 1 < pt <= 2: # in between critical and safe distance
                reward -= 0.2*(2 - pt)
        '''
     
        return goal_n_obs, reward, done, info

    def render(self, mode='human'):
        self._env.render(mode)

    def reset(self):
        obs = self._env.reset()
        return self._get_obs(obs)

    def close(self):
        self._env.close()

    def seed(self, seed):
        """
        Sets the seed value of the environment
        """
        self._env.seed(seed)
