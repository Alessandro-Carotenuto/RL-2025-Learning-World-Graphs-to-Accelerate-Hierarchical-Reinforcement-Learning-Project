import gymnasium as gym


#----------------------------------------------------------------------------#
#                              MINIGRID WRAPPER, FAST                        #
#----------------------------------------------------------------------------#
# This need a known configuration to use the gym make built in function
# Actually fast and good for testing purposes, minimal


class FastWrapper():
    def __init__(self, env_name, max_steps=1000, render_mode=None):
        self.env = gym.make(env_name, render_mode=render_mode)
        self.max_steps = max_steps

        # the env_name is a string, that could be chosen from the documentation

        # Forward necessary attributes for ManualControl
        self.unwrapped = self.env.unwrapped

    def reset(self, seed=None, options=None):
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()
        return self._process_obs(obs), info

    def step(self, action):
        # ACTUAL STEP
        obs, reward, terminated, truncated, info = self.env.step(action)

        # LAUNCHES PROCESSING OF OBSERVATIONS AND REWARDS
        processed_obs = self._process_obs(obs)
        normalized_reward = self._normalize_reward(reward)
        return processed_obs, normalized_reward, terminated, truncated, info

    def _process_obs(self, obs):
        # EXTRACT AND PROCESS THE OBSERVATIONS
        return obs

    def _normalize_reward(self, reward):
        # SCALE AND NORMALIZE THE REWARDS
        return reward

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()