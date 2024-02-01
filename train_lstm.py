import argparse
import os.path as osp
from pathlib import Path
import sys
import time

from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    obs_as_tensor,
    safe_mean,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
import numpy as np
import wandb
from sb3_contrib import RecurrentPPO as RPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

from envs.memory_planning_game import MemoryPlanningGame
from utils.misc import linear_schedule


SelfRecurrentPPO = TypeVar("SelfRecurrentPPO", bound="RecurrentPPO")


class RecurrentPPO(RPPO):
    def learn(
        self: SelfRecurrentPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "RecurrentPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRecurrentPPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max(
                    (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
                )
                fps = int(
                    (self.num_timesteps - self._num_timesteps_at_start) / time_elapsed
                )
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    avg_rew = safe_mean(
                        [ep_info["r"] for ep_info in self.ep_info_buffer]
                    )
                    self.logger.record("rollout/ep_rew_mean", avg_rew)
                    self.logger.record(
                        "rollout/ep_len_mean",
                        safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                    )
                    oracle_reward = (
                        self.env.envs[0].max_episode_steps
                        / self.env.envs[0].oracle_min_num_actions
                    )
                    self.logger.record(
                        "rollout/fraction_of_oracle", avg_rew / oracle_reward
                    )
                    self.logger.record("rollout/oracle_reward", oracle_reward)
                self.logger.record("time/fps", fps)
                self.logger.record(
                    "time/time_elapsed", int(time_elapsed), exclude="tensorboard"
                )
                self.logger.record(
                    "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
                )
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self


def main(config):
    # Parallel environments
    env_kwargs = dict(
        maze_size=config["maze_size"],
        num_maze=config["num_maze"],
        max_episode_steps=config["max_episode_steps"],
        target_reward=1.0,
        per_step_reward=0.0,
        num_labels=config["num_labels"],
        render_mode=None,
        dict_space=False,
        seed=config["seed"],
    )
    if config["num_maze"] > 0:
        env_kwargs["maps"] = MemoryPlanningGame.generate_worlds(**env_kwargs)
    env = make_vec_env(
        MemoryPlanningGame, n_envs=config["n_envs"], env_kwargs=env_kwargs
    )
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    run = wandb.init(
        dir=config["run_dir"],
        project=config["proj_name"],
        group=config["group_name"],
        config=config,
        sync_tensorboard=True,
        save_code=True,
    )
    policy_kwargs = dict(
        n_lstm_layers=config["n_layer"],
        lstm_hidden_size=config["n_hidden"],
        net_arch=[],
        shared_lstm=config["shared_vf"],
        enable_critic_lstm=not config["shared_vf"],
    )
    if config["linear_schedule"]:
        lr = linear_schedule(config["learning_rate"])
        cr = linear_schedule(config["clip_range"])
    else:
        lr = config["learning_rate"]
        cr = config["clip_range"]
    set_random_seed(config["seed"])

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        batch_size=config["batch_size"],
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        policy_kwargs=policy_kwargs,
        max_grad_norm=config["max_grad_norm"],
        learning_rate=lr,
        clip_range=cr,
        verbose=1,
        tensorboard_log=osp.join(run.dir, run.id),
    )
    print("tot num of params:", sum(p.numel() for p in model.policy.parameters()))
    model.learn(total_timesteps=config["total_timesteps"])

    local_path = osp.join("/".join(run.dir.split("/")[:-3]), "local_saves_wandb")
    Path(osp.join(local_path, run.id)).mkdir(parents=True, exist_ok=True)
    model.save(osp.join(local_path, run.id, "model"))
    env.save(osp.join(local_path, run.id, "venv_norm.pkl"))
    if config["upload_to_wandb"]:
        wandb.save(osp.join(local_path, run.id, "model.zip"))
        wandb.save(osp.join(local_path, run.id, "venv_norm.pkl"))
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./")
    parser.add_argument("--proj_name", type=str, default="MemoryPlanning")
    parser.add_argument("--group_name", type=str, default="RecurrentPPO")
    parser.add_argument("--num_labels", type=int, default=10)
    parser.add_argument("--maze_size", type=int, default=6)
    parser.add_argument("--num_maze", type=int, default=32)
    parser.add_argument("--max_episode_steps", type=int, default=100)
    parser.add_argument("--n_envs", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--n_steps", "-K", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=0.25)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-4)
    parser.add_argument("--total_timesteps", type=int, default=40_000_000)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--n_a", type=int, default=5)
    parser.add_argument("--seed", type=int, default=531)
    parser.add_argument("--shared_vf", action="store_true", default=False)
    parser.add_argument("--linear_schedule", action="store_true", default=False)
    parser.add_argument("--upload_to_wandb", "-w", action="store_true", default=False)

    args = parser.parse_args()
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    main(config=vars(args))
