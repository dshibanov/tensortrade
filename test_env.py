import numpy as np
import pytest
import sys, os, time
sys.path.append(os.getcwd())
import pandas as pd
import pytest
import ray
from ray import tune
from ray.tune.registry import register_env
from tensortrade.env.default import *
# import math

import copy
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.rllib.utils import check_env

# TODO: please rename this file to test_env.py
# or better move these tests to existed one

register_env("multy_symbol_env", create_multy_symbol_env)
def eval_fold(params):
    score = 2
    def set_params(config, params):
        # print(params)
        for p in params:
            # print(p)
            config[p] = params[p]
        return config

    framework = 'torch'
    # current_config = set_params(config, params)
    current_config = copy.deepcopy(params)#.copy()
    print('current_config: feed ', current_config["symbols"][0]["feed"])
    symbols = current_config["symbols"]
    test_fold_index = current_config["test_fold_index"]
    print('test_fold_index ', test_fold_index)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     train_feed, test_feed = get_train_test_feed(current_config)
    #     print('FEED ', len(train_feed), len(test_feed))


    eval_config = current_config
    eval_config["test"] = True
    env = create_multy_symbol_env(current_config)
    print('check_env ')
    check_env(env)


    a = env.reset()
    pprint(a)
    # return

    config = (
        # PGConfig()
        DQNConfig()
        # .environment(SimpleCorridor, env_config={"corridor_length": 10})
        # .environment(env="multy_symbol_env", env_config=config)
        .environment(env="multy_symbol_env", env_config=current_config)
        # Training rollouts will be collected using just the learner
        # process, but evaluation will be done in parallel with two
        # workers. Hence, this run will use 3 CPUs total (1 for the
        # learner + 2 more for evaluation workers).
        # .rollouts(num_rollout_workers=0)
        # .evaluation(
        # #     evaluation_num_workers=2,
        # #     # Enable evaluation, once per training iteration.
        # #     evaluation_interval=1,
        # #     # Run 10 episodes each time evaluation runs (OR "auto" if parallel to
        # #     # training).
        # #     # evaluation_duration="auto" if args.evaluation_parallel_to_training else 10,
        # #     evaluation_duration="auto" if evaluation_parallel_to_training else 10,
        # #     # Evaluate parallelly to training.
        # #     # evaluation_parallel_to_training=args.evaluation_parallel_to_training,
        # #     evaluation_parallel_to_training=evaluation_parallel_to_training,
        # #     # evaluation_config=PGConfig.overrides(
        #     # evaluation_config=DQNConfig.overrides(env_config={"test": True}, explore=True
        #     evaluation_config=DQNConfig.overrides(env_config=eval_config
        # #         env_config={
        # #             # Evaluate using LONGER corridor than trained on.
        # #             "corridor_length": 5,
        # #         },
        #     )
        #     custom_evaluation_function=eval_fn,
        # )
        # .framework(args.framework)
        .framework(framework)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    stop = {
        # "training_iteration": args.stop_iters,
        # "timesteps_total": args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
        "training_iteration": 3,
        # "timesteps_total": 2000,
        "episode_reward_mean": 15,
    }

    # pprint({key: value for key, value in config.to_dict().items() if key not in ["symbols"]})
    # return
    tuner = tune.Tuner(
        # "PG",
        "DQN",
        # param_space=config.to_dict(),
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=1, checkpoint_config=CheckpointConfig(checkpoint_frequency=2)),
        # run_config=RunConfig(stop=TrialPlateauStopper(metric="score"))
        # run_config=RunConfig(stop=TrialPlateauStopper(metric="reward"),
        #                      # checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),
        #                      checkpoint_config=CheckpointConfig(checkpoint_frequency=3),
        #                      verbose=2)
    )
    # results = tuner.fit()
    results = tuner.fit().get_dataframe(filter_metric="score", filter_mode="min")
    print(results)


    # 2 make validation here
    #
    # 3 return validation_score

    # return {"score": score}



def test_eval_fold():
    config = {"max_episode_length": 50, # bigger is ok, smaller is not
              "make_folds":True,
              "num_folds": 7,
              "symbols": make_symbols(5, 410),
              "cv_mode": 'proportional',
              "test_fold_index": 3,
              "reward_window_size": 1,
              "window_size": 2,
              "max_allowed_loss": 100,
              "use_force_sell": True,
              "multy_symbol_env": True,
              "test": False
             }


    action = 0 # 0 - buy asset, 1 - sell asset
    config["nn_topology_a"] = 7
    config["nn_topology_b_to_a_ratio"] = 0.3
    config["nn_topology_c_to_b_ratio"] = 0.7
    config["nn_topology_h_to_l_ratio"] = 2
    config = make_folds(config)
    # print('config ', config)
    # print('AFTER ')
    pprint({key: value for key, value in config.items() if key not in ["symbols"]})

    # evaluation_config=DQNConfig.overrides(explore=False)
    # pprint(evaluation_config)
    num_cpus = 2
    local_mode = True
    # framework = 'torch'
    evaluation_parallel_to_training = True
    # # ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
    ray.init(num_cpus=num_cpus or None, local_mode=local_mode)
    eval_fold(config)
    return


def pong():
    import gymnasium as gym
    # env = gym.make("LunarLander-v2", render_mode="human")
    env = gym.make('CartPole-v1')

    import stable_baselines3.common.env_checker as ech
    r = ech.check_env(env)
    print(f'checking is done by stable-baseline3: {r}')
    return

    # env = gym.make('Pong-v0')
    observation, info = env.reset(seed=42)
    for _ in range(5):
        action = env.action_space.sample()  # this is where you would insert your policy
        obs, reward, terminated, truncated, info = env.step(action)

        print(f'step obs: {obs} type(obs): {type(obs)} obs.shape: {np.shape(obs)} ') #, env.env.observer.observation_space)
        # assert np.shape(obs) == env.env.observer.observation_space.shape
        # obs, reward, done, truncated, info = env.step(0)
        # assert np.shape(obs) == env.env.observer.observation_space.shape

        if terminated or truncated:
            observation, info = env.reset()
            print(f'reset obs: {obs} type(obs): {type(obs)} obs.shape: {np.shape(obs)} ') #, env.env.observer.observation_space)
            # print('reset ', obs, np.shape(obs)) #, env.env.observer.observation_space)

    env.close()

if __name__ == "__main__":
    # test_eval_fold()
    pong()
