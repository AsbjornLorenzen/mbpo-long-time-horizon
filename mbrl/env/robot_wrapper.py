import gymnasium as gym
import numpy as np



"""
A wrapper for dealing with the fact that observations are weird
"""
class RobotWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        
        obs_space = self.observation_space['observation']
        desired_goal = self.observation_space['desired_goal']
        self.observation_space = gym.spaces.Box(np.concatenate((obs_space.low, desired_goal.low)), 
                                                np.concatenate((obs_space.high, desired_goal.high)), 
                                                np.concatenate((obs_space.low, desired_goal.low)).shape, 
                                                obs_space.dtype, 
                                                0)

    def observation(self, observation):
        achieved_goal, desired_goal, observation_ = observation['achieved_goal'], observation['desired_goal'], observation['observation']
        new_observation = np.concatenate((observation_, desired_goal))
        return new_observation
