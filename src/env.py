"""Observation wrapper: extracts screen only for SB3 compatibility."""

import gymnasium as gym
import numpy as np


class ScreenOnlyWrapper(gym.ObservationWrapper):
    """
    Extracts 'screen' from VizDoom's dict observation.
    Also resizes to 120x160 for faster training.
    """

    def __init__(self, env, resize: tuple = (120, 160)):
        super().__init__(env)
        self.resize = resize
        # Update observation space to match extracted screen
        h, w, c = env.observation_space["screen"].shape
        if resize:
            h, w = resize
        self.observation_space = gym.spaces.Box(0, 255, (h, w, c), dtype=np.uint8)

    def observation(self, obs):
        screen = obs["screen"]
        if self.resize:
            import cv2
            screen = cv2.resize(screen, (self.resize[1], self.resize[0]))
        return screen
