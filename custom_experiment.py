"""Example of a custom experiment wrapped around an RLlib Algorithm."""
import argparse

import ray
from ray import train, tune
import ray.rllib.algorithms.ppo as ppo
from tensortrade.env.default import *
import copy
from ray.rllib.algorithms.dqn.dqn import DQNConfig, DQN
import pandas as pd
import sys, os, time
from ray.tune.registry import register_env
# from ray.rllib.agents import dqn
# import ray.rllib.algorithms.algorithm_config.AlgorithmConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
parser = argparse.ArgumentParser()
parser.add_argument("--train-iterations", type=int, default=3)

register_env("multy_symbol_env", create_multy_symbol_env)

def shape_to_topology2(a, b_to_a, lh_ratio):
    b = math.ceil(a*b_to_a)
    depth = math.ceil(max(a,b) * lh_ratio)
    topo = [a]
    ic(a,b,depth)
    if depth > 2:
        # layer_delta = math.ceil((b-a) / (depth-2))
        layer_delta = math.ceil((b-a) / (depth-1))
        ic(layer_delta)
        for i in range(depth-2):
            # print(i)
            cur_layer = topo[-1] + layer_delta
            # ic(cur_layer)
            topo.append(cur_layer)

    topo.append(b)
    # print(depth)
    return(topo)

def experiment(config):
    iterations = config.pop("train-iterations")

    algo = ppo.PPO(config=config)
    checkpoint = None
    train_results = {}

    # Train
    for i in range(iterations):
        train_results = algo.train()
        if i % 2 == 0 or i == iterations - 1:
            checkpoint = algo.save(train.get_context().get_trial_dir())
        train.report(train_results)
    algo.stop()

    # Manual Eval
    config["num_workers"] = 0
    eval_algo = ppo.PPO(config=config)
    eval_algo.restore(checkpoint)
    env = eval_algo.workers.local_worker().env

    obs, info = env.reset()
    done = False
    eval_results = {"eval_reward": 0, "eval_eps_length": 0}
    while not done:
        action = eval_algo.compute_single_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        eval_results["eval_reward"] += reward
        eval_results["eval_eps_length"] += 1
    results = {**train_results, **eval_results}
    train.report(results)




# def get_cv_scores(config):
#     num_folds = config["num_folds"]
#     config = make_folds(config)
#     config["test_fold_index"] = tune.grid_search(np.arange(0,num_folds))

#     num_iterations=config.get('num_train_iterations', 2)
#     # stop = {
#     #     # "training_iteration": args.stop_iters,
#     #     # "timesteps_total": args.stop_timesteps,
#     #     # "episode_reward_mean": args.stop_reward,
#     #     "training_iteration": num_iterations,
#     #     # "timesteps_total": 2000,
#     #     # "episode_reward_mean": 15,
#     #     "score": 57
#     # }
#     tuner = tune.Tuner(
#         eval_fold,
#         param_space=config,

#         # run_config=air.RunConfig(stop=stop, verbose=1, checkpoint_config=CheckpointConfig(checkpoint_frequency=2)),
#         # run_config=air.RunConfig(stop=stop, verbose=1),
#         run_config=air.RunConfig(verbose=1),
#         # run_config=RunConfig(stop=TrialPlateauStopper(metric="score"))
#         # run_config=RunConfig(stop=TrialPlateauStopper(metric="reward"),
#         #                      # checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),
#         #                      checkpoint_config=CheckpointConfig(checkpoint_frequency=3),
#         #                      verbose=2)
#     )
#     # results = tuner.fit().get_best_result(metric="score", mode="min")
#     # results = tuner.fit().get_dataframe(filter_metric="score", filter_mode="min")
#     results = tuner.fit()

#     df = results.get_dataframe(filter_metric="score", filter_mode="min")
#     # print('this is results.. ')
#     # print(type(results))
#     # print(results)

#     # ic(df)
#     # cv_score = df["score"]
#     # print(f'cv_score {cv_score}')
#     # return cv_score
#     return df


def get_cv_score(config):

    def inner_evaluate(config):
        # train 

        # iterations = config.pop("train-iterations")
        # iterations = config.get("num_folds", 1)

        # algo = ppo.DQN(config=config)
        # checkpoint = None
        # train_results = {}

        # # Train
        # for i in range(iterations):
        #     train_results = algo.train()
        #     if i % 2 == 0 or i == iterations - 1:
        #         checkpoint = algo.save(train.get_context().get_trial_dir())
        #     train.report(train_results)
        # algo.stop()

        # print(checkpoint)


        # Another train 
        #
        score = 2
        framework = 'torch'
        current_config = copy.deepcopy(config)#.copy()
        # ic(current_config["symbols"][0]["feed"])
        symbols = current_config["symbols"]
        test_fold_index = current_config["test_fold_index"]
        # print('test_fold_index ', test_fold_index)
        ic(test_fold_index)

        eval_config = current_config
        eval_config["test"] = True

        # topo = shape_to_topology2(codict["env_config"]["nnt_a"], codict["env_config"]["nnt_btoa"], codict["env_config"]["nnt_lh_ratio"])
        topo = shape_to_topology2(config["nnt_a"], config["nnt_btoa"], config["nnt_lh_ratio"])
        ic(topo)
        # codict["model"]["fcnet_hiddens"] = topo

        config = (
            DQNConfig()
            .training(hiddens=topo)
            .environment(env="multy_symbol_env", env_config=current_config)
            .framework(framework)
            .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        )


        # TODO: 
            # - make algo first 
            # - should set fc_net hiddens and control it

        ic(config.to_dict())
        return {"score": 666}

        algo = config.build()

        print('oookaa')
        return

        # stop = {
        #     "training_iteration": 5,
        #     "episode_reward_mean": 15,
        # }

        codict = config.to_dict()
        topo = shape_to_topology2(codict["env_config"]["nnt_a"], codict["env_config"]["nnt_btoa"], codict["env_config"]["nnt_lh_ratio"])
        ic(topo)
        codict["model"]["fcnet_hiddens"] = topo
        ic(codict)
        tuner = tune.Tuner(
            # "PG",
            "DQN",
            # param_space=config.to_dict(),
            # param_space=config.to_dict(),
            param_space=codict,
            run_config=air.RunConfig(stop=stop, verbose=1, checkpoint_config=CheckpointConfig(checkpoint_frequency=2)),
            # run_config=RunConfig(stop=TrialPlateauStopper(metric="score"))
            # run_config=RunConfig(stop=TrialPlateauStopper(metric="reward"),
            #                      # checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),
            #                      checkpoint_config=CheckpointConfig(checkpoint_frequency=3),
            #                      verbose=2)
        )

        algo = (
            DQNConfig()
            .rollouts(num_rollout_workers=1)
            .resources(num_gpus=0)
            .environment(env=envs["trains"][0])
            .build()
        )


        # results = tuner.fit()
        results = tuner.fit()#.get_dataframe(filter_metric="score", filter_mode="min")
        best_result = results.get_best_result(metric="loss", mode="min")
        best_checkpoint = best_result.checkpoint
        algo = Algorithm.from_checkpoint(best_checkpoint)
        evaluated = algo.evaluate()

        # -------------------------
        # test
        score = 1#...
        return {"score": score}

    results = []
    for f in range(config["num_folds"]):
        print(f)
        results.append(inner_evaluate(config)["score"])#, f)

    return results


def test_get_cv_score():
    ray.shutdown()
    num_cpus = 2
    local_mode = True
    framework = 'torch'
    evaluation_parallel_to_training = True
    # ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
    ray.init(num_cpus=num_cpus or None, local_mode=local_mode)

    config = {
              "max_episode_length": 50, # bigger is ok, smaller is not
              "min_episode_length": 5, # bigger is ok, smaller is not
              "make_folds":True,
              "num_folds": 3,
              "symbols": make_symbols(5, 41),
              # "symbols": make_symbols(2, 40),

              "cv_mode": 'proportional',
              "test_fold_index": 1,
              "reward_window_size": 1,
              "window_size": 2,
              "max_allowed_loss": 100,
              "use_force_sell": True,
              "multy_symbol_env": True,
              "num_train_iterations": 15,
              "test": False,
              "nnt_a": 34,
              "nnt_btoa": 0.3,
              "nnt_lh_ratio": 0.005
             }

    config = make_folds(config)
    scores = get_cv_score(config)
    # ic(f'we got scores {scores}')
    print(scores)


def test_model():

    env_config = {
              "max_episode_length": 50, # bigger is ok, smaller is not
              "min_episode_length": 5, # bigger is ok, smaller is not
              "make_folds":True,
              "num_folds": 3,
              "symbols": make_symbols(7, 1411),
              # "symbols": make_symbols(2, 40),

              "cv_mode": 'proportional',
              "test_fold_index": 1,
              "reward_window_size": 1,
              "window_size": 2,
              "max_allowed_loss": 100,
              "use_force_sell": True,
              "multy_symbol_env": True,
              "num_train_iterations": 15,
              "test": False,
              "nnt_a": 34,
              "nnt_btoa": 0.3,
              "nnt_lh_ratio": 0.005
             }

    ray.shutdown()
    num_cpus = 2
    local_mode = True
    framework = 'torch'
    evaluation_parallel_to_training = True
    # ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
    ray.init(num_cpus=num_cpus or None, local_mode=local_mode)

    framework = 'torch'
    env_config = make_folds(env_config)
    config = (
        DQNConfig()
        # .environment(SimpleCorridor, env_config={"corridor_length": 10})
        .environment(env="multy_symbol_env", env_config=env_config)
        # .environment(env="multy_symbol_env", env_config=current_config)
        # Training rollouts will be collected using just the learner
        # process, but evaluation will be done in parallel with two
        # workers. Hence, this run will use 3 CPUs total (1 for the
        # learner + 2 more for evaluation workers).
        # .rollouts(num_rollout_workers=0)
        .framework(framework)
        .training(
            model={
                "fcnet_hiddens" : [24, 24],
            }
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    ic('konkon')
    algo = config.build()
    print('dd')
    ic('nice')
    print(algo)


    checkpoint = None
    train_results = {}

    iterations = 20
    print('heyyyaaaaa')
    # Train
    mean_rewardz=[]
    for i in range(iterations):
        train_results = algo.train()
        print(f'i {i} ')
        # pprint(train_results)
        mean_rewardz.append(train_results['episode_reward_mean'])

        if i % 2 == 0 or i == iterations - 1:
            checkpoint = algo.save(train.get_context().get_trial_dir())

        # train.report(train_results)
    algo.stop()

    print(mean_rewardz)

    # conf = restored_algo.config["env_config"]
    # model = restored_algo.config.model
    # policy = restored_algo.get_policy(policy_id="default_policy")

    conf = algo.config["env_config"]
    model = algo.config.model
    policy = algo.get_policy(policy_id="default_policy")
    print('policy')
    print(policy)
    print(conf["test"])
    conf["test"] = True
    env = create_multy_symbol_env(conf)
    get_action_scheme(env).portfolio.exchanges[0].options.max_trade_size = 100000000000

    # return
    # obs = np.array([0.0, 0.1, 0.2, 0.3])  # individual CartPole observation
    obs,_ = env.reset()
    # info = env.env.informer.info(env.env)
    info = get_info(env)
    action = policy.compute_single_action(obs)[0]
    print("action: ", action)
    done = False
    step = 0
    volumes=[]
    instruments=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))

    obss=[]
    reward=float('NaN')
    # observations=[np.append(obs[-1], np.append([action, info['net_worth']], volumes))]
    print('>>>>>>>>>>> START SIMULATION >>>>>>>>>>>>')
    observations=[]
    while done == False:
        wallets = [w.total_balance for w in get_action_scheme(env).portfolio.wallets]
        print(f' step {step}, close {obs[-1]} action {action} info {info}, wallets {wallets}')
        non_zero_wallets=0
        for w in wallets:
            if w != 0:
                non_zero_wallets+=1
        assert non_zero_wallets == 1

        if step == 90:
            print('hoi')
        # print(step)
        obss.append(obs)
        action = policy.compute_single_action(obs)[0]
        # print(f"step {step}, action {action}, info {info} ")
        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
        row = np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], np.append(volumes, [reward])))
        # row = np.append(obs[-1], np.append([action, info['net_worth']], np.append(volumes, [reward])))
        observations.append(row)
        obs, reward, done, truncated, info = env.step(action)
        step += 1

    # for i in range(len(obss)):
    #     print(i, obss[i])

    # return

    track = pd.DataFrame(observations)
    track.columns = ['close', 'symbol_code',  'end_of_episode', 'action', 'net_worth'] + instruments + ['reward']
    # get model/policy from last checkpoint or last iteration
    # get model or policy from algo
    #
    print(track)
    print(np.mean(track["reward"].dropna()))



    # ic(algo)
    ic('ice')
    return
    # ic(config.to_dict())
    config["num_workers"] = 1
    config["hiddens"] = [512,128]

    # conf = AlgorithmConfig.from_dict(config)
    conf = AlgorithmConfig.from_dict(DQNConfig().to_dict())
    return
    algo = DQN(conf)

    ic(algo)
    return
    # trainer = dqn.DQNTrainer(config=config,env="AirRaid-ram-v0")
    model = trainer.get_policy().model
    model.base_model.summary()
    model.q_value_head.summary()

if __name__ == "__main__":
    test_model()
    # test_get_cv_score()

    if False:
        args = parser.parse_args()
        ray.init(num_cpus=3)
        config = ppo.PPOConfig().environment("CartPole-v1")
        config = config.to_dict()
        config["train-iterations"] = args.train_iterations
        tune.Tuner(
            tune.with_resources(experiment, ppo.PPO.default_resource_request(config)),
            param_space=config,
        ).fit()




