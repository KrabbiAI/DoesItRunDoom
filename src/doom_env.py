"""
DoesItRunDoom? — Doom Gymnasium Environment
Wraps VizDoom as a Gymnasium environment.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import vizdoom as vz


class DoomEnv(gym.Env):
    """
    VizDoom Deadly Corridor as Gymnasium environment.

    7 Available buttons: MOVE_LEFT, MOVE_RIGHT, MOVE_FORWARD,
    MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, ATTACK

    Observation: RGB screen (160x120) + game variable (HEALTH)
    Actions: Discrete (one per button)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 35}
    max_steps = 2100  # VizDoom tics per episode

    def __init__(self, scenario="deadly_corridor", visible=False, frame_skip=4):
        super().__init__()
        self.scenario = scenario
        self.visible = visible
        self.frame_skip = frame_skip

        self.game = vz.DoomGame()
        self.game.set_window_visible(visible)

        # Set WAD paths BEFORE load_config
        import os
        vizdoom_dir = os.path.dirname(vz.__file__)
        scenario_dir = os.path.join(vizdoom_dir, "scenarios")
        scenario_wad = os.path.join(scenario_dir, f"{scenario}.wad")
        game_wad = os.path.join(vizdoom_dir, "freedoom2.wad")
        self.game.set_doom_scenario_path(scenario_wad)
        self.game.set_doom_game_path(game_wad)

        # Load config (uses our pre-set WAD paths)
        cfg_path = os.path.join(scenario_dir, f"{scenario}.cfg")
        self.game.load_config(cfg_path)

        # Override resolution to match CNN input
        self.game.set_screen_resolution(vz.ScreenResolution.RES_160X120)
        self.game.set_screen_format(vz.ScreenFormat.RGB24)

        # Observation: RGB screen buffer — use channel-last (HWC)
        # Stable-Baselines3 CnnPolicy expects HWC for image input
        # Shape: (height, width, channels) = (120, 160, 3)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(120, 160, 3),
            dtype=np.uint8
        )

        # For Deadly Corridor: 7 buttons → 7 discrete actions
        # Action i = press button i only
        n_actions = len(self.game.get_available_buttons())
        self.action_space = spaces.Discrete(n_actions)

        # Precompute action vectors
        # Each action = binary mask for one button
        self.actions = self._make_action_vectors(n_actions)

        self.episode_step = 0

    def _make_action_vectors(self, n_actions):
        """Create one-hot action vectors for each available button."""
        action_vectors = []
        available = self.game.get_available_buttons()
        for i in range(n_actions):
            vec = [0] * len(available)
            if i < len(available):
                vec[i] = 1
            action_vectors.append(vec)
        return action_vectors

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        else:
            self.game.set_seed(0)
        self.game.init()
        self.episode_step = 0

        state = self.game.get_state()
        obs = self._get_obs(state)
        info = self._get_info(state)

        return obs, info

    def step(self, action):
        self.episode_step += 1

        # Execute action (frame_skip controls steps per call)
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
        return state.screen_buffer

    def _get_info(self, state):
        if state is None:
            return {}
        health = state.game_variables[0] if state.game_variables is not None else 0
        return {"health": health, "game_variables": state.game_variables}

    def _compute_reward(self, state):
        """
        Reward: VizDoom internal reward already handles:
        - +dX for moving toward vest
        - -dX for moving away
        - -100 for death
        We add: small living reward to encourage survival.
        """
        if self.game.is_episode_finished():
            return 0  # Already handled by VizDoom
        return 1  # +1 per step survived

    def close(self):
        self.game.close()

    def render(self):
        return self._get_obs(self.game.get_state())

    def seed(self, seed=None):
        self.game.set_seed(seed)
        return [seed]
