# MIT License

# Copyright (c) 2024 Vindula Jayawardana, Baptiste Freydt, Ao Qu, Cameron Hickert, Zhongxia Yan, Cathy Wu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import ray
import wandb
import argparse

from pathlib import Path
from datetime import datetime
from ray.rllib.algorithms.ppo import PPOConfig
from sumo.constants import REGULAR
from env.config import IntersectionZooEnvConfig
from env.environment import IntersectionZooEnv
from env.task_context import PathTaskContext
from env.rllib_callback import MetricsCallback
import pufferlib.emulation
import pufferlib.wrappers
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.examples._old_api_stack.models.centralized_critic_models import (
    TorchCentralizedCriticModel,
)
from algorithm.mappo import CentralizedCritic

parser = argparse.ArgumentParser(description='Model arguments')
parser.add_argument('--dir', default='/Users/bfreydt/MIT_local/IntersectionZoo/wd/new_exp', type=str, help='Result directory')
parser.add_argument('--intersection_dir', default='dataset/salt-lake-city', type=str, help='Path to intersection dataset')
parser.add_argument('--wandb_project', default='intersectionzoo', type=str, help='Weights and biases project name')
parser.add_argument('--wandb_entity', default='vindula', type=str, help='Weights and biases entity name')
parser.add_argument('--num_workers', default=10, type=str, help='Number of workers')
parser.add_argument('--num_gpus', default=0, type=str, help='Number of GPUs')
parser.add_argument('--save_frequency', default=5, type=str, help='Frequency of saving checkpoints')
parser.add_argument('--rollouts', default=500, type=str, help='Number of rollouts')

parser.add_argument('--penetration', default=0.33, type=str, help='Eco drive adoption rate')
parser.add_argument('--temperature_humidity', default='68_46', type=str, help='Temperature and humidity for evaluations')

args = parser.parse_args()
print(args)

# wandb.init(project=args.wandb_project, entity=args.wandb_entity)

ray.init(ignore_reinit_error=True, num_cpus=args.num_workers + 15)

tasks = PathTaskContext(
    dir=Path(args.intersection_dir),                    
    single_approach=True,
    penetration_rate=args.penetration,
    temperature_humidity=args.temperature_humidity,
    electric_or_regular=REGULAR,
)

env_conf = IntersectionZooEnvConfig(
    task_context=tasks.sample_task(),
    working_dir=Path(args.dir),
    moves_emissions_models=[args.temperature_humidity],
    fleet_reward_ratio=1,
)

# def postprocess_fn(policy, sample_batch, other_agent_batches, episode):
#     agents = ["agent_1", "agent_2", "agent_3"]  # simple example of 3 agents
#     global_obs_batch = np.stack(
#         [other_agent_batches[agent_id][1]["obs"] for agent_id in agents],
#         axis=1)
#     # add the global obs and global critic value
#     sample_batch["global_obs"] = global_obs_batch
#     sample_batch["central_vf"] = self.sess.run(
#         self.critic_network, feed_dict={"obs": global_obs_batch})
#     return sample_batch


register_env("my_env", lambda config: pufferlib.emulation(env=IntersectionZooEnv(config)))

def curriculum_fn(train_results, task_settable_env, env_ctx):
    return tasks.sample_task()

ModelCatalog.register_custom_model(
        "cc_model",
        TorchCentralizedCriticModel,
    )

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=args.num_workers, sample_timeout_s=3600, \
        batch_mode="complete_episodes", rollout_fragment_length=400)
    .resources(num_gpus=args.num_gpus)
    .training(model={"custom_model": "cc_model"})
    .evaluation(evaluation_num_workers=1, evaluation_duration=1, \
        evaluation_duration_unit='episodes', evaluation_force_reset_envs_before_iteration=True)
    .environment(
        env=IntersectionZooEnv,
        env_config={"intersectionzoo_env_config": env_conf},
        env_task_fn=curriculum_fn,
    )
    .callbacks(MetricsCallback)
    .build()
)

for i in range(args.rollouts):
    
    result = CentralizedCritic(algo).train()
    
    print(f"iteration {i} completed.")
    
    sampler_results = result['sampler_results']
    custom_results = result['custom_metrics']
    metrics = {key: sampler_results[key] for key in sampler_results.keys()}
    metrics.update({key: custom_results[key] for key in custom_results.keys()})
    # wandb.log(metrics, step=i)
    
    if i % args.save_frequency == 0:
        save_dir = f'{args.dir}/runs/{str(i)}/{datetime.now().strftime("%Y%m%d_%H%M")}'
        checkpoint_dir = algo.save(save_dir).checkpoint.path
        print(f"Checkpoint saved at {checkpoint_dir}")

# wandb.finish()


