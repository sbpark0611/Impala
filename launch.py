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
from ray.air.integrations.mlflow import setup_mlflow
import mlflow
from memory_maze import tasks
import wandb
import flatdict

from collections.abc import MutableMapping

from ray.air.integrations.wandb import WandbLoggerCallback
import shutil

tf1, tf, tfv = try_import_tf()
SUPPORTED_ENVS = [
    "RepeatAfterMeEnv",
    "RepeatInitialObsEnv",
    "LookAndPush",
    "StatelessCartPole",
    "MemoryPlanningGame",
    "MemoryMaze",
]


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    # example-specific args
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
        "--n-steps", type=int, default=100000, help="Number of iterations to train."
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
        "--tune",
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

def mlflow_log_metrics(mlflow, metrics: dict, step: int):
    while True:
        try:
            mlflow.log_metrics(metrics, step=step)
            break
        except:
            print('Error logging metrics - will retry.')
            time.sleep(10)

def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='/') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def drawGraphWithAlgo(mlflow, envName, algo, test_len):
    # prepare env
    if envName == "MemoryPlanningGame":
        env = MemoryPlanningGame()
    elif envName == "MemoryMaze":
        env = tasks.memory_maze_9x9()
    
    # simulate and get steps per task
    steps_per_task = [[] for _ in range(test_len)]
    for i in range(test_len):
        obs, info = env.reset()
        done = False

        # start with all zeros as state
        num_transformers = config["model"]["attention_num_transformer_units"]
        state = algo.get_policy().get_initial_state()
        state = np.array([state])
        # run one iteration until done

        prev_steps = 0
        episode_steps = 0
        print(f'Drawing graph: episode {i}')
        while not done:
            action, state_out, _ = algo.compute_single_action(obs, state)
            next_obs, reward, done, _, _ = env.step(action)
            episode_steps += 1
            if reward == 1:
                steps_per_task[i].append(episode_steps - prev_steps)
                prev_steps = episode_steps
            obs = next_obs
            state = [
                np.concatenate([state[i], [state_out[i]]], axis=0)[1:] for i in range(num_transformers)
            ]
    
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
        if mlflow:
            mlflow_log_metrics(mlflow, {"number_of_steps_to_goal": mean_steps_per_task[i]}, step=i+1)
        else:
            wandb.log({"number_of_steps_to_goal": mean_steps_per_task[i], 'custom_step': step})
    
    print(f'Drawing graph finished')

def drawGraph(mlflow, envName, predictor, test_len):
    # create env and policy
    if envName == "MemoryPlanningGame":
        env = MemoryPlanningGame()
    elif envName == "MemoryMaze":
        os.environ['MUJOCO_GL'] = 'glfw'
        env = tasks.memory_maze_9x9()

    # simulate and get steps per task
    steps_per_task = [[] for _ in range(test_len)]
    for i in range(test_len):
        obs = env.reset()
        done = False
        prev_steps = 0
        episode_steps = 0
        print(f'Drawing graph: episode {i}')
        while not done:
            action, mets = predictor.predict(obs)
            obs, reward, done, _, inf = env.step(action)
            episode_steps += 1
            if reward == 1:
                steps_per_task[i].append(episode_steps - prev_steps)
                prev_steps = episode_steps

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
        mlflow_log_metrics(mlflow, {"number_of_steps_to_goal": mean_steps_per_task[i]}, step=i+1)
    
    print(f'Drawing graph finished')


if __name__ == "__main__":
    args = get_cli_args()
    if args.env == "MemoryMaze":
        os.environ['MUJOCO_GL'] = 'glfw'

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    # register custom environments
    registry.register_env("RepeatAfterMeEnv", lambda c: RepeatAfterMeEnv(c))
    registry.register_env("RepeatInitialObsEnv", lambda _: RepeatInitialObsEnv())
    registry.register_env("LookAndPush", lambda _: OneHot(LookAndPush()))
    registry.register_env("StatelessCartPole", lambda _: StatelessCartPole())
    registry.register_env("MemoryPlanningGame", lambda _: MemoryPlanningGame())
    registry.register_env("MemoryMaze", lambda _: tasks.memory_maze_9x9())

    # main part: RLlib config with AttentionNet model
    config = (
        impala.ImpalaConfig()
        .environment(
            args.env,
            env_config={},
        )
        .training(
            entropy_coeff = 0.001, # 0.01
            gamma = 0.9, # 0.95
            lr=0.00001, #0.0004
            epsilon = 0.0001,
            momentum = 0,
            decay = 0.99,
            grad_clip=20.0,
            model={
                "use_attention": True,
                "max_seq_len": 50,
                "attention_num_transformer_units": 1,
                "attention_dim": 128,
                "attention_memory_inference": 100,
                "attention_memory_training": 100,
                "attention_num_heads": 1,
                "attention_head_dim": 64,
                "attention_position_wise_mlp_dim": 64,
            },
            # TODO (Kourosh): Enable when LSTMs are supported.
            _enable_learner_api=False,
        )
        .framework(args.framework)
        .rollouts(num_envs_per_worker=1)
        .resources(
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", 0)),
        )
        .rl_module(_enable_rl_module_api=False)
    )

    stop = {
        "training_iteration": args.n_steps,
        #"timesteps_total": args.stop_timesteps,
        #"episode_reward_mean": args.stop_reward,
    }

    # Manual training loop (no Ray tune).
    if not args.tune:

        # manual training loop using PPO and manually keeping track of state
        if args.run != "IMPALA":
            raise ValueError("Only support --run IMPALA with --no-tune.")
        
        '''
        experiment_name = "impala_GTrXl"
        mlflow.set_experiment(experiment_name)
        myMlflow = setup_mlflow(config.to_dict(), run_name = "memory_maze", experiment_name=experiment_name)
        '''

        wandb.init()
        wandb.run.name = 'impala_GTrXl'
        wandb.run.save()
        wandb.define_metric("custom_step")
        wandb.define_metric("*", step_metric="custom_step")
        
        checkpoint_interval = 10
        algo = config.build()
        # run manual training loop and print results after each iteration
        for step in range(1, args.n_steps+1):
            print("train step:", step)

            result = algo.train()
            print(pretty_print(result)) 
            
            #myMlflow.log_metric('episode_reward_mean', result['episode_reward_mean'], step=step)
            
            if step % checkpoint_interval == 0:
                path_to_checkpoint = algo.save()
                print(
                    "An Algorithm checkpoint has been created inside directory: "
                    f"'{path_to_checkpoint}'."
                )
                shutil.make_archive(path_to_checkpoint, "zip", path_to_checkpoint)
                wandb.save(f"{path_to_checkpoint}.zip")
                


            fd = flatten_dict(result)
            fd['custom_step'] = step
            wandb.log(fd)

            '''
            # stop training if the target train steps or reward are reached
            if (
                result["timesteps_total"] >= args.stop_timesteps
                or result["episode_reward_mean"] >= args.stop_reward
            ):
                break
            '''
        
        drawGraphWithAlgo(None, args.env, algo, 10)

    # Run with Tune for auto env and algorithm creation and TensorBoard.
    else:
        tuner = tune.Tuner(
            args.run,
            param_space=config.to_dict(),
            run_config=air.RunConfig(
                stop=stop, verbose=1,
                name="mlflow",
                callbacks=[
                    WandbLoggerCallback(
                        project="impala",
                        save_checkpoints=True,
                    )
                ],
                storage_path = "./ray_results", #/proj/internal_group/dscig/kdkyum/workdir/Impala/ray_results
            ),
        )
        results = tuner.fit()
        
        #best_result = results.get_best_result("reward")

        #checkpoint: TorchCheckpoint = best_result.checkpoint

        #algo = config.build()

        # Create a Predictor using the best result's checkpoint
        #predictor = TorchPredictor.from_checkpoint(checkpoint, algo.get_policy())
        #drawGraph(predictor, 10)
        
        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()

