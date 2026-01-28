# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an environment instance to RL-Games vectorized environment.

The following example shows how to wrap an environment for RL-Games and register the environment construction
for RL-Games :class:`Runner` class:

.. code-block:: python

    from rl_games.common import env_configurations, vecenv

    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    # configuration parameters
    rl_device = "cuda:0"
    clip_obs = 10.0
    clip_actions = 1.0

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

"""

# needed to import for allowing type-hinting:gym.spaces.Box | None
from __future__ import annotations

import gym.spaces  # needed for rl-games incompatibility: https://github.com/Denys88/rl_games/issues/261
import gymnasium
import torch
from collections.abc import Callable

from rl_games.common import env_configurations
from rl_games.common.vecenv import IVecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv, VecEnvObs

"""
Vectorized environment wrapper.
"""


class RlGamesVecEnvWrapper(IVecEnv):
    """Wraps around Isaac Lab environment for RL-Games.

    This class wraps around the Isaac Lab environment. Since RL-Games works directly on
    GPU buffers, the wrapper handles moving of buffers from the simulation environment
    to the same device as the learning agent. Additionally, it performs clipping of
    observations and actions.

    For algorithms like asymmetric actor-critic, RL-Games expects a dictionary for
    observations. This dictionary contains "obs" and "states" which typically correspond
    to the actor and critic observations respectively.

    To use asymmetric actor-critic, map privileged observation groups under ``"states"`` (e.g. ``["critic"]``).

    The wrapper supports **either** concatenated tensors (default) **or** Dict inputs:
    when wrapper is concate mode, rl-games sees {"obs": Tensor, (optional)"states": Tensor}
    when wrapper is not concate mode, rl-games sees {"obs": dict[str, Tensor], (optional)"states": dict[str, Tensor]}

    - Concatenated mode (``concate_obs_group=True``): ``observation_space``/``state_space`` are ``gym.spaces.Box``.
    - Dict mode (``concate_obs_group=False``): ``observation_space``/``state_space`` are ``gym.spaces.Dict`` keyed by
      the requested groups. When no ``"states"`` groups are provided, the states Dict is omitted at runtime.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.


    Reference:
        https://github.com/Denys88/rl_games/blob/master/rl_games/common/ivecenv.py
        https://github.com/NVIDIA-Omniverse/IsaacGymEnvs
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv | DirectRLEnv,
        rl_device: str,
        clip_obs: float,
        clip_actions: float,
        obs_groups: dict[str, list[str]] | None = None,
        concate_obs_group: bool = True,
    ):
        """Initializes the wrapper instance.

        Args:
            env: The environment to wrap around.
            rl_device: The device on which agent computations are performed.
            clip_obs: The clipping value for observations.
            clip_actions: The clipping value for actions.
            obs_groups: The remapping from isaaclab observation to rl-games, default to None for backward compatible.
            concate_obs_group: The boolean value indicates if input to rl-games network is dict or tensor. Default to
                True for backward compatible.

        Raises:
            ValueError: The environment is not inherited from :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
            ValueError: If specified, the privileged observations (critic) are not of type :obj:`gym.spaces.Box`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        # store provided arguments
        self._rl_device = rl_device
        self._clip_obs = clip_obs
        self._clip_actions = clip_actions
        self._sim_device = env.unwrapped.device

        # resolve the observation group
        self._concate_obs_groups = concate_obs_group
        self._obs_groups = obs_groups
        if obs_groups is None:
            self._obs_groups = {"obs": ["policy"], "states": []}
            if not self.unwrapped.single_observation_space.get("policy"):
                raise KeyError("Policy observation group is expected if no explicit groups is defined")
            if self.unwrapped.single_observation_space.get("critic"):
                self._obs_groups["states"] = ["critic"]

        if (
            self._concate_obs_groups
            and isinstance(self.state_space, gym.spaces.Box)
            and isinstance(self.observation_space, gym.spaces.Box)
        ):
            self.rlg_num_states = self.state_space.shape[0]
        elif (
            not self._concate_obs_groups
            and isinstance(self.state_space, gym.spaces.Dict)
            and isinstance(self.observation_space, gym.spaces.Dict)
        ):
            space = [space.shape[0] for space in self.state_space.values()]
            self.rlg_num_states = sum(space)
        else:
            raise TypeError(
                "only valid combination for state space is gym.space.Box when concate_obs_groups is True,             "
                "   and gym.space.Dict when concate_obs_groups is False. You have concate_obs_groups:                "
                f" {self._concate_obs_groups}, and state_space: {self.state_space.__class__}"
            )

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return (
            f"<{type(self).__name__}{self.env}>"
            f"\n\tObservations clipping: {self._clip_obs}"
            f"\n\tActions clipping     : {self._clip_actions}"
            f"\n\tAgent device         : {self._rl_device}"
            f"\n\tAsymmetric-learning  : {self.rlg_num_states != 0}"
        )

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.spaces.Box | gym.spaces.Dict:
        """Returns the :attr:`Env` :attr:`observation_space` (``Box`` if concatenated, otherwise ``Dict``)."""
        # note: rl-games only wants single observation space
        space = self.unwrapped.single_observation_space
        clip = self._clip_obs
        if not self._concate_obs_groups:
            policy_space = {grp: gym.spaces.Box(-clip, clip, space.get(grp).shape) for grp in self._obs_groups["obs"]}
            return gym.spaces.Dict(policy_space)
        else:
            shapes = [space.get(group).shape for group in self._obs_groups["obs"]]
            cat_shape, self._obs_concat_fn = make_concat_plan(shapes)
            return gym.spaces.Box(-clip, clip, cat_shape)

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        # note: rl-games only wants single action space
        action_space = self.unwrapped.single_action_space
        if not isinstance(action_space, gymnasium.spaces.Box):
            raise NotImplementedError(
                f"The RL-Games wrapper does not currently support action space: '{type(action_space)}'."
                f" If you need to support this, please modify the wrapper: {self.__class__.__name__},"
                " and if you are nice, please send a merge-request."
            )
        # return casted space in gym.spaces.Box (OpenAI Gym)
        # note: maybe should check if we are a sub-set of the actual space. don't do it right now since
        #   in ManagerBasedRLEnv we are setting action space as (-inf, inf).
        return gym.spaces.Box(-self._clip_actions, self._clip_actions, action_space.shape)

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    @property
    def num_envs(self) -> int:
        """Returns the number of sub-environment instances."""
        return self.unwrapped.num_envs

    @property
    def device(self) -> str:
        """Returns the base environment simulation device."""
        return self.unwrapped.device

    @property
    def state_space(self) -> gym.spaces.Box | gym.spaces.Dict | None:
        """Returns the privileged observation space for the critic (``Box`` if concatenated, otherwise ``Dict``)."""
        # # note: rl-games only wants single observation space
        space = self.unwrapped.single_observation_space
        clip = self._clip_obs
        if not self._concate_obs_groups:
            state_space = {grp: gym.spaces.Box(-clip, clip, space.get(grp).shape) for grp in self._obs_groups["states"]}
            return gym.spaces.Dict(state_space)
        else:
            shapes = [space.get(group).shape for group in self._obs_groups["states"]]
            cat_shape, self._states_concat_fn = make_concat_plan(shapes)
            return gym.spaces.Box(-self._clip_obs, self._clip_obs, cat_shape)

    def get_number_of_agents(self) -> int:
        """Returns number of actors in the environment."""
        return getattr(self, "num_agents", 1)

    def get_env_info(self) -> dict:
        """Returns the Gym spaces for the environment."""
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "state_space": self.state_space,
        }

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self):  # noqa: D102
        obs_dict, _ = self.env.reset()
        # process observations and states
        return self._process_obs(obs_dict)

    def step(self, actions):  # noqa: D102
        # move actions to sim-device
        actions = actions.detach().clone().to(device=self._sim_device)
        # clip the actions
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
        # perform environment step
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        # note: only useful when `value_bootstrap` is True in the agent configuration
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated.to(device=self._rl_device)
        # process observations and states
        obs_and_states = self._process_obs(obs_dict)
        # move buffers to rl-device
        # note: we perform clone to prevent issues when rl-device and sim-device are the same.
        rew = rew.to(device=self._rl_device)
        dones = (terminated | truncated).to(device=self._rl_device)
        extras = {
            k: v.to(device=self._rl_device, non_blocking=True) if hasattr(v, "to") else v for k, v in extras.items()
        }
        # remap extras from "log" to "episode"
        if "log" in extras:
            extras["episode"] = extras.pop("log")

        return obs_and_states, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _process_obs(self, obs_dict: VecEnvObs) -> dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]]:
        """Processing of the observations and states from the environment.

        Note:
            States typically refers to privileged observations for the critic function. It is typically used in
            asymmetric actor-critic algorithms.

        Args:
            obs_dict: The current observations from environment.

         Returns:
            A dictionary for RL-Games with keys:
            - ``"obs"``: either a concatenated tensor (``concate_obs_group=True``) or a Dict of group tensors.
            - ``"states"`` (optional): same structure as above when state groups are configured; omitted otherwise.
        """
        # clip the observations
        for key, obs in obs_dict.items():
            obs_dict[key] = torch.clamp(obs, -self._clip_obs, self._clip_obs)

        # process input obs dict
        rl_games_obs = {"obs": {group: obs_dict[group] for group in self._obs_groups["obs"]}}
        if len(self._obs_groups["states"]) > 0:
            rl_games_obs["states"] = {group: obs_dict[group] for group in self._obs_groups["states"]}

        if self._concate_obs_groups:
            rl_games_obs["obs"] = self._obs_concat_fn(list(rl_games_obs["obs"].values()))
            if "states" in rl_games_obs:
                rl_games_obs["states"] = self._states_concat_fn(list(rl_games_obs["states"].values()))

        return rl_games_obs



class RlGamesTrainValVecEnvWrapper(RlGamesVecEnvWrapper):
    """RLGames wrapper with train/val split.

    RLGames sees only num_train environments. Validation envs run with a separate
    player (created via runner.create_player()) that handles RNN states internally.

    IMPORTANT: When using this wrapper, ensure agent_cfg.params.config.num_actors = num_train
    (not total envs). RLGames uses num_actors from config, not vec_env.num_envs.

    Usage:
        env = RlGamesTrainValVecEnvWrapper(env, rl_device, clip_obs, clip_actions, num_train=100)
        runner = load_rlgames_policy(agent_cfg, train_mode=True)
        env.set_rlgames_runner(runner)  # Creates separate val player
    """

    def __init__(self, env, rl_device, clip_obs, clip_actions, obs_groups=None,
                 concate_obs_groups=True, num_train=None):
        self._total_envs = env.unwrapped.num_envs
        self._num_train = num_train if num_train is not None else self._total_envs
        self._num_val = self._total_envs - self._num_train
        self._rlgames_runner = None
        self._val_player = None
        self._last_val_obs = None
        self._last_synced_epoch = -1  # Track when we last synced weights

        super().__init__(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
        print(f"[RlGamesTrainValVecEnvWrapper] Total: {self._total_envs}, Train: {self._num_train}, Val: {self._num_val}")

    def set_rlgames_runner(self, runner):
        """Set RLGames runner and create separate val player for inference.

        Uses runner.create_player() to create a player that handles RNN states
        internally. Weights are synced from training agent every step.
        """
        assert hasattr(runner, "get_eval_player"), "Runner must implement get_eval_player()"
        assert hasattr(runner, "get_train_agent"), "Runner must implement get_train_agent()"
        assert hasattr(runner, "sync_eval_player"), "Runner must implement sync_eval_player()"
        self._rlgames_runner = runner

        if self._num_val == 0:
            return

        # Use cached eval player from runner (dedicated inference model)
        self._val_player = runner.get_eval_player()

        # Configure player for batched val inference
        # (normally player.run() calls get_batch_size() to set these)
        self._val_player.batch_size = self._num_val
        self._val_player.has_batch_dimension = True
        self._val_player.init_rnn()  # Initialize RNN states for num_val batch size
        print(f"[INFO] Created separate val player for {self._num_val} val envs (batch_size={self._num_val})")

    def _get_val_actions(self, obs):
        """Get actions for val envs using separate val player."""
        if self._val_player is None or self._rlgames_runner is None:
            return torch.zeros(self._num_val, self.action_space.shape[0],
                             device=self._rl_device)

        # Only sync weights after training update (when epoch_num changes)
        agent = self._rlgames_runner.get_train_agent()
        current_epoch = getattr(agent, 'epoch_num', 0)
        if current_epoch != self._last_synced_epoch:
            self._rlgames_runner.sync_eval_player()
            self._last_synced_epoch = current_epoch

        # Use player's get_action which handles RNN states internally
        with torch.no_grad():
            actions = self._val_player.get_action(obs, is_deterministic=True)

        return actions

    def _reset_val_rnn_states(self, val_dones):
        """Reset RNN states for done val envs (indices relative to val envs)."""
        if self._val_player is None or not getattr(self._val_player, 'is_rnn', False):
            return
        if self._val_player.states is None:
            return

        # Get indices of done val envs (relative to val slice)
        done_indices = val_dones.nonzero(as_tuple=False).squeeze(-1)
        if done_indices.numel() == 0:
            return

        # Reset RNN states for done envs (states are (num_layers, batch, hidden))
        for state in self._val_player.states:
            state[:, done_indices, :] = 0.0
        
    @property
    def num_envs(self) -> int:
        """Number of training envs (what RLGames sees)."""
        return self._num_train

    @property
    def total_envs(self) -> int:
        """Total number of envs (train + val)."""
        return self._total_envs

    @property
    def num_val_envs(self) -> int:
        """Number of validation envs."""
        return self._num_val

    def _slice_obs(self, obs, start, end):
        """Slice observations (handles both tensor and dict formats)."""
        if isinstance(obs, dict):
            return {k: v[start:end].clone() for k, v in obs.items()}
        return obs[start:end].clone()

    def _filter_extras(self, extras):
        """Filter per-env tensors in extras to train envs only. Handles nested dicts."""
        filtered = {}
        for k, v in extras.items():
            if isinstance(v, torch.Tensor) and len(v.shape) > 0 and v.shape[0] == self._total_envs:
                filtered[k] = v[:self._num_train]
            elif isinstance(v, dict):
                # Recursively filter nested dicts (e.g., extras["episode"])
                filtered[k] = self._filter_extras(v)
            else:
                filtered[k] = v
        return filtered

    def reset(self):
        # Reset all envs via parent (returns obs for all envs)
        obs_and_states = super().reset()

        # Reset val player RNN states on full reset
        if self._num_val > 0 and self._val_player is not None:
            self._val_player.init_rnn()

        # Store val obs BEFORE filtering for next step's action inference
        if self._num_val > 0:
            self._last_val_obs = self._slice_obs(obs_and_states["obs"], self._num_train, None)

        # Filter outputs to train envs only (what RLGames sees)
        obs_and_states["obs"] = self._slice_obs(obs_and_states["obs"], 0, self._num_train)
        if "states" in obs_and_states:
            obs_and_states["states"] = self._slice_obs(obs_and_states["states"], 0, self._num_train)
        return obs_and_states

    def step(self, train_actions):
        # Get val actions using separate player (handles RNN internally)
        if self._num_val > 0:
            val_actions = self._get_val_actions(self._last_val_obs)
            full_actions = torch.cat([train_actions, val_actions], dim=0)
        else:
            full_actions = train_actions

        # Step all envs via parent (returns obs/rew/dones for all envs)
        obs_and_states, rew, dones, extras = super().step(full_actions)

        # Handle val env RNN state resets BEFORE filtering dones
        if self._num_val > 0:
            val_dones = dones[self._num_train:]
            self._reset_val_rnn_states(val_dones)

        # Store val obs BEFORE filtering for next step's action inference
        if self._num_val > 0:
            self._last_val_obs = self._slice_obs(obs_and_states["obs"], self._num_train, None)

        # Filter outputs to train envs only (what RLGames sees)
        obs_and_states["obs"] = self._slice_obs(obs_and_states["obs"], 0, self._num_train)
        if "states" in obs_and_states:
            obs_and_states["states"] = self._slice_obs(obs_and_states["states"], 0, self._num_train)
        rew, dones = rew[:self._num_train], dones[:self._num_train]
        extras = self._filter_extras(extras)
        return obs_and_states, rew, dones, extras



def make_concat_plan(shapes: list[tuple[int, ...]]) -> tuple[tuple[int, ...], Callable]:
    """
    Given per-sample shapes (no batch dim), return:
      - the concatenated per-sample shape
      - a function that concatenates a list of batch tensors accordingly.

    Rules:
      0) Empty -> (0,), No-op
      1) All 1D -> concat features (dim=1).
      2) Same rank > 1:
         2a) If all s[:-1] equal -> concat along last dim (channels-last, dim=-1).
         2b) If all s[1:] equal  -> concat along first dim (channels-first, dim=1).
    """
    if len(shapes) == 0:
        return (0,), lambda x: x
    # case 1: all vectors
    if all(len(s) == 1 for s in shapes):
        return (sum(s[0] for s in shapes),), lambda x: torch.cat(x, dim=1)
    # case 2: same rank > 1
    rank = len(shapes[0])
    if all(len(s) == rank for s in shapes) and rank > 1:
        # 2a: concat along last axis (…C)
        if all(s[:-1] == shapes[0][:-1] for s in shapes):
            out_shape = shapes[0][:-1] + (sum(s[-1] for s in shapes),)
            return out_shape, lambda x: torch.cat(x, dim=-1)
        # 2b: concat along first axis (C…)
        if all(s[1:] == shapes[0][1:] for s in shapes):
            out_shape = (sum(s[0] for s in shapes),) + shapes[0][1:]
            return out_shape, lambda x: torch.cat(x, dim=1)
        else:
            raise ValueError(f"Could not find a valid concatenation plan for rank {[(len(s),) for s in shapes]}")
    else:
        raise ValueError("Could not find a valid concatenation plan, please make sure all value share the same size")


"""
Environment Handler.
"""


class RlGamesGpuEnv(IVecEnv):
    """Thin wrapper to create instance of the environment to fit RL-Games runner."""

    # TODO: Adding this for now but do we really need this?

    def __init__(self, config_name: str, num_actors: int, **kwargs):
        """Initialize the environment.

        Args:
            config_name: The name of the environment configuration.
            num_actors: The number of actors in the environment. This is not used in this wrapper.
        """
        self.env: RlGamesVecEnvWrapper = env_configurations.configurations[config_name]["env_creator"](**kwargs)

    def step(self, action):  # noqa: D102
        return self.env.step(action)

    def reset(self):  # noqa: D102
        return self.env.reset()

    def get_number_of_agents(self) -> int:
        """Get number of agents in the environment.

        Returns:
            The number of agents in the environment.
        """
        return self.env.get_number_of_agents()

    def get_env_info(self) -> dict:
        """Get the Gym spaces for the environment.

        Returns:
            The Gym spaces for the environment.
        """
        return self.env.get_env_info()
