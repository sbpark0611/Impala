"""
Example of using an RL agent (default: PPO) with an AttentionNet model,
which is useful for environments where state is important but not explicitly
part of the observations.

For example, in the "repeat after me" environment (default here), the agent
needs to repeat an observation from n timesteps before.
AttentionNet keeps state of previous observations and uses transformers to
learn a policy that successfully repeats previous observations.
Without attention, the RL agent only "sees" the last observation, not the one
n timesteps ago and cannot learn to repeat this previous observation.

AttentionNet paper: https://arxiv.org/abs/1506.07704

This example script also shows how to train and test a PPO agent with an
AttentionNet model manually, i.e., without using Tune.

---
Run this example with defaults (using Tune and AttentionNet on the "repeat
after me" environment):
$ python attention_net.py
Then run again without attention:
$ python attention_net.py --no-attention
Compare the learning curve on TensorBoard:
$ cd ~/ray-results/; tensorboard --logdir .
There will be a huge difference between the version with and without attention!

Other options for running this example:
$ python attention_net.py --help
"""
import argparse
import os

import numpy as np

import ray
from ray import air, tune
from ray.rllib.algorithms import impala
from ray.rllib.examples.env.look_and_push import LookAndPush, OneHot
from ray.rllib.examples.env.repeat_after_me_env import RepeatAfterMeEnv
from ray.rllib.examples.env.repeat_initial_obs_env import RepeatInitialObsEnv
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune import registry
from ray.tune.logger import pretty_print

from memory_planning_game import MemoryPlanningGame
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train.torch import TorchCheckpoint, TorchPredictor
import time

tf1, tf, tfv = try_import_tf()
SUPPORTED_ENVS = [
    "RepeatAfterMeEnv",
    "RepeatInitialObsEnv",
    "LookAndPush",
    "StatelessCartPole",
    "MemoryPlanningGame",
]


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    # example-specific args
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Do NOT use attention. For comparison: The agent will not learn.",
    )
    parser.add_argument("--env", choices=SUPPORTED_ENVS, default="MemoryPlanningGame")

    # general args
    parser.add_argument(
        "--run", default="IMPALA", help="The RLlib-registered algorithm to use."
    )
    parser.add_argument("--num-cpus", type=int)
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument(
        "--stop-iters", type=int, default=1000000, help="Number of iterations to train."
    )
    '''
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=500000,
        help="Number of timesteps to train.",
    )
    '''
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=80.0,
        help="Reward at which we stop training.",
    )
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test: --stop-reward must "
        "be achieved within --stop-timesteps AND --stop-iters.",
    )
    parser.add_argument(
        "--no-tune",
        default=False,
        action="store_true",
        help="Run without Tune using a manual train loop instead. Here,"
        "there is no TensorBoard support.",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args

def mlflow_log_metrics(metrics: dict, step: int):
    import mlflow
    while True:
        try:
            mlflow.log_metrics(metrics, step=step)
            break
        except:
            print('Error logging metrics - will retry.')
            time.sleep(10)

def drawGraph(predictor, test_len):
    # create env and policy
    env = MemoryPlanningGame()

    # simulate and get steps per task
    steps_per_task = [[] for _ in range(test_len)]
    for i in range(test_len):
        obs = env.reset()
        done = False
        prev_steps = 0
        info(f'Drawing graph: episode {i}')
        while not done:
            action, mets = predictor.predict(obs)
            obs, reward, done, inf = env.step(action)
            if reward == 1:
                steps_per_task[i].append(inf["episode_steps"] - prev_steps)
                prev_steps = inf["episode_steps"]

    # calculate mean steps per task and log by mlflow
    mean_steps_per_task = []
    task = 0
    while True:
        lst = []
        for i in range(test_len):
            if len(steps_per_task[i]) > task:
                lst.append(steps_per_task[i][task])
        task += 1
        
        if len(lst) < 2:
            break

        mean_steps_per_task.append(sum(lst) / len(lst))

    for i in range(len(mean_steps_per_task)):
        mlflow_log_metrics({"number_of_steps_to_goal": mean_steps_per_task[i]}, step=i+1)
    
    info(f'Drawing graph finished')


if __name__ == "__main__":
    args = get_cli_args()

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    # register custom environments
    registry.register_env("RepeatAfterMeEnv", lambda c: RepeatAfterMeEnv(c))
    registry.register_env("RepeatInitialObsEnv", lambda _: RepeatInitialObsEnv())
    registry.register_env("LookAndPush", lambda _: OneHot(LookAndPush()))
    registry.register_env("StatelessCartPole", lambda _: StatelessCartPole())
    registry.register_env("MemoryPlanningGame", lambda _: MemoryPlanningGame())

    # main part: RLlib config with AttentionNet model
    config = (
        impala.ImpalaConfig()
        .environment(
            args.env,
            env_config={},
        )
        .training(
            lr=0.0003, #tune.grid_search([0.0001, 0.0003]),
            grad_clip=20.0,
            model={
                "use_attention": not args.no_attention,
                "max_seq_len": 10,
                "attention_num_transformer_units": 1,
                "attention_dim": 32,
                "attention_memory_inference": 10,
                "attention_memory_training": 10,
                "attention_num_heads": 1,
                "attention_head_dim": 32,
                "attention_position_wise_mlp_dim": 32,
            },
            # TODO (Kourosh): Enable when LSTMs are supported.
            _enable_learner_api=False,
        )
        .framework(args.framework)
        .rollouts(num_envs_per_worker=20)
        .resources(
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", 0))
        )
        .rl_module(_enable_rl_module_api=False)
    )

    stop = {
        "training_iteration": args.stop_iters,
        #"timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # Manual training loop (no Ray tune).
    if args.no_tune:
        # manual training loop using PPO and manually keeping track of state
        if args.run != "IMPALA":
            raise ValueError("Only support --run IMPALA with --no-tune.")
        algo = config.build()
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = algo.train()
            print(pretty_print(result))
            # stop training if the target train steps or reward are reached
            if (
                result["timesteps_total"] >= args.stop_timesteps
                or result["episode_reward_mean"] >= args.stop_reward
            ):
                break

        # Run manual test loop (only for RepeatAfterMe env).
        if args.env == "MemoryPlanningGame":
            print("Finished training. Running manual test/inference loop.")
            # prepare env
            env = MemoryPlanningGame()
            obs, info = env.reset()
            done = False
            total_reward = 0
            # start with all zeros as state
            num_transformers = config["model"]["attention_num_transformer_units"]
            state = algo.get_policy().get_initial_state()
            # run one iteration until done
            while not done:
                action, state_out, _ = algo.compute_single_action(obs, state)
                next_obs, reward, done, _, _ = env.step(action)
                print(f"Obs: {obs}, Action: {action}, Reward: {reward}")
                obs = next_obs
                total_reward += reward
                state = [
                    np.concatenate([state[i], [state_out[i]]], axis=0)[1:]
                    for i in range(num_transformers)
                ]
            print(f"Total reward in test episode: {total_reward}")

    # Run with Tune for auto env and algorithm creation and TensorBoard.
    else:
        tuner = tune.Tuner(
            args.run,
            param_space=config.to_dict(),
            run_config=air.RunConfig(
                stop=stop, verbose=2,
                name="mlflow",
                callbacks=[
                    MLflowLoggerCallback(
                        experiment_name="mlflow_callback_example",
                        save_artifact=True,
                    )
                ],
                storage_path="./storage",
            ),
        )
        results = tuner.fit()
        '''
        best_result = results.get_best_result("reward")

        checkpoint: TorchCheckpoint = best_result.checkpoint

        # Create a Predictor using the best result's checkpoint
        predictor = TorchPredictor.from_checkpoint(checkpoint)
        drawGraph(predictor, 10)
        '''
        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()