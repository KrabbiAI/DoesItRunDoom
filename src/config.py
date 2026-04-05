"""Scenario and hyperparameter configuration for DoesItRunDoom?"""

import vizdoom
import vizdoom.gymnasium_wrapper.gymnasium_env_defns as env_defns

SCENARIOS = {
    "deadly_corridor": {
        "env_id": "VizdoomScenario-v0",
        "env_cls": env_defns.VizdoomScenarioEnv,
        "env_config": {
            "scenario_config_file": "deadly_corridor.cfg",
            "window_visible": False,
        },
        "ep_timeout": 2100,
        "ppo": {
            "learning_rate": 3e-4,
            "n_steps": 4096,
            "batch_size": 64,
            "n_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        },
    },
    "health_gathering": {
        "env_id": "VizdoomHealthGatheringSupreme-v0",
        "env_cls": env_defns.VizdoomScenarioEnv,
        "env_config": {
            "scenario_config_file": "health_gathering_supreme.cfg",
            "window_visible": False,
        },
        "ep_timeout": 2100,
        "ppo": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        },
    },
}

DEFAULT_SCENARIO = "deadly_corridor"
