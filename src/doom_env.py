"""
Ludicrous Speed - Doom RL Environment
VizDoom wrapped as Gymnasium environment with reward shaping.
"""

import gymnasium as gym
import numpy as np
from vizdoom import DoomGame, ScreenResolution
from gymnasium import spaces


class DoomEnv(gym.Env):
    """
    VizDoom Basic scenario as Gymnasium environment.
    Observation: Game screen (RGB or grayscale)
    Actions: Discrete (left, right, forward, shoot, etc.)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, scenario="basic", visible=False, frame_skip=4):
        super().__init__()
        self.scenario = scenario
        self.visible = visible
        self.frame_skip = frame_skip

        self.game = DoomGame()
        self.game.set_window_visible(visible)

        # Load scenario
        import os
        import vizdoom as vz
        vizdoom_dir = os.path.dirname(vz.__file__)
        scenario_dir = os.path.join(vizdoom_dir, "scenarios")
        cfg = os.path.join(scenario_dir, f"{scenario}.cfg")
        self.game.load_config(cfg)

        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_screen_format(vz.ScreenFormat.RGB24)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(False)
        self.game.set_automap_buffer_enabled(False)

        # Actions: [turn left, turn right, move forward, move backward, strafe left, strafe right, shoot, use]
        n_actions = 8
        self.action_space = spaces.Discrete(n_actions)

        # Predefined actions
        self.actions = [
            [1, 0, 0, 0, 0, 0, 0, 0],  # turn left
            [0, 1, 0, 0, 0, 0, 0, 0],  # turn right
            [0, 0, 1, 0, 0, 0, 0, 0],  # move forward
            [0, 0, 0, 1, 0, 0, 0, 0],  # move backward
            [0, 0, 0, 0, 1, 0, 0, 0],  # strafe left
            [0, 0, 0, 0, 0, 1, 0, 0],  # strafe right
            [0, 0, 0, 0, 0, 0, 1, 0],  # shoot
            [0, 0, 0, 0, 0, 0, 0, 1],  # use
        ]

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(120, 160, 3),
            dtype=np.uint8
        )

        self.episode_step = 0
        self.max_steps = 2100  # ~2 min at 35fps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.set_seed(seed)
        self.game.init()
        self.episode_step = 0

        state = self.game.get_state()
        obs = self._get_obs(state)
        info = self._get_info(state)

        return obs, info

    def step(self, action):
        self.episode_step += 1

        # Execute action
        self.game.make_action(self.actions[action], self.frame_skip)

        # Get state
        state = self.game.get_state()
        obs = self._get_obs(state)
        reward = self._compute_reward(state)
        terminated = self.game.is_episode_finished()
        truncated = self.episode_step >= self.max_steps
        info = self._get_info(state)

        return obs, reward, terminated, truncated, info

    def _get_obs(self, state):
        if state is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        return state.screen_buffer  # RGB

    def _get_info(self, state):
        if state is None:
            return {}
        return {
            "game_variables": state.game_variables,
            "label": None
        }

    def _compute_reward(self, state):
        """
        Reward shaping:
        - +1 for being alive (each step)
        - +10 for killing an enemy
        - -100 for death
        - +100 for winning (level complete)
        """
        if self.game.is_episode_finished():
            if self.game.is_player_dead():
                return -100
            else:
                return 100  # victory

        # Living reward
        reward = 1

        # Additional: check if enemy killed (binary game variable 0 = dead, 1 = alive)
        # In basic scenario: kill count is in game variable
        if state:
            kill_count = state.game_variables[0] if state.game_variables is not None else 0
            reward += kill_count * 10

        return reward

    def close(self):
        self.game.close()

    def render(self):
        return self._get_obs(self.game.get_state())

    def seed(self, seed=None):
        self.game.set_seed(seed)
        return [seed]
