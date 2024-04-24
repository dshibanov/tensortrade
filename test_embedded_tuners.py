import pandas as pd
import sys, os, time
sys.path.append(os.getcwd())
import numpy as np
import ray
from ray import tune
from ray.tune.registry import register_env

from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.rllib.utils import check_env
from ray.rllib.algorithms.algorithm import Algorithm
from icecream import ic
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch
import random

def test_embedded_tuners():
    ray.shutdown()
    num_cpus = 2
    local_mode = True
    framework = 'torch'
    evaluation_parallel_to_training = True
    ray.init(num_cpus=num_cpus or None, local_mode=local_mode)

    config = {
                "num_iterations": 10,
                "h1": tune.uniform(0, 3),
                "h2": tune.uniform(2, 7)
             }

    # algo = BayesOptSearch(random_search_steps=7)
    # algo = OptunaSearch(metric="score", mode="min")

    def objective(config):
        # return {"score": random.random()}

        def inner_objective(configx):
            return {"score": random.random()}

        conf = {"id": tune.grid_search(np.arange(0,10))}
        tuner2 = tune.Tuner(
            inner_objective,
            param_space=conf,
        )

        ic("ama here")
        results = tuner2.fit()
        df = results.get_dataframe(filter_metric="score", filter_mode="min")
        return {"score":df["score"].mean()}

    tuner1 = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            # metric="episode_reward_mean",
            metric="score",
            mode="max",
            # search_alg=algo,
            num_samples=7
        ),
        param_space=config,
    )

    results = tuner1.fit()
    df = results.get_dataframe(filter_metric="score", filter_mode="min")
    return df


if __name__ == "__main__":
    test_embedded_tuners()
