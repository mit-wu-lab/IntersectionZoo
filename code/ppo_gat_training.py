from collections import OrderedDict
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
from ray.rllib.models import ModelCatalog

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch_geometric.utils import dense_to_sparse
import numpy as np

import sys
sys.path.append('/home/vindula/Desktop/IntersectionZoo/')
print(sys.path)

parser = argparse.ArgumentParser(description='Model arguments')
parser.add_argument('--dir', default='wd/new_exp', type=str, help='Result directory')
parser.add_argument('--intersection_dir', default='/home/vindula/Desktop/IntersectionZoo/dataset/salt-lake-city', type=str, help='Path to intersection dataset')
parser.add_argument('--wandb_project', default='intersectionzoo', type=str, help='Weights and biases project name')
parser.add_argument('--wandb_entity', default='vindula', type=str, help='Weights and biases entity name')
parser.add_argument('--num_workers', default=10, type=str, help='Number of workers')
parser.add_argument('--num_gpus', default=1, type=str, help='Number of GPUs')
parser.add_argument('--save_frequency', default=5, type=str, help='Frequency of saving checkpoints')
parser.add_argument('--rollouts', default=100, type=str, help='Number of rollouts')

parser.add_argument('--penetration', default=0.33, type=str, help='Eco drive adoption rate')
parser.add_argument('--temperature_humidity', default='68_46', type=str, help='Temperature and humidity for evaluations')

args = parser.parse_args()
print(args)

Path(args.dir).mkdir(parents=True, exist_ok=True)

wandb.init(project=args.wandb_project, entity=args.wandb_entity)

ray.init(ignore_reinit_error=True, num_cpus=args.num_workers + 15)

# Function to convert OrderedDict to matrix
def ordered_dict_to_matrix(ordered_dict):
    matrices = []
    for key, value in ordered_dict.items():
        if isinstance(value, OrderedDict):
            # Recursively handle nested OrderedDict
            sub_matrix = ordered_dict_to_matrix(value)
            matrices.append(sub_matrix)
        else:
            matrices.append(value.cpu())
    
    # Concatenate all matrices along the second axis (columns)
    return np.hstack(matrices)

def transform_features(input_dict, device):
    features_dict = input_dict["obs"]
    # convert the OrderedDict to a matrix
    result_matrix = ordered_dict_to_matrix(features_dict)
    batch_tensor = torch.tensor(result_matrix)
    batch_tensor = batch_tensor.to(device)
    return batch_tensor

class GATModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, num_heads=4):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.values = None
        
        num_features = obs_space.shape[0]
        self.encoder = nn.Linear(num_features, 64)
        self.gat1 = GATConv(64, 64, heads=num_heads, concat=False)
        self.gat2 = GATConv(64, 64, heads=num_heads, concat=False)
        self.actor = nn.Linear(64, num_outputs)
        self.critic = nn.Linear(64, 1)

    def forward(self, input_dict, state, seq_lens):
        # given no extra vehicular connections are know, we assume fully connected graph
        # get the features in a nxm matrix where n is the number of 
        # nodes (in the graph) and m is the number of features
        x = transform_features(input_dict, self.device)
        self._last_obs = x
        # number of nodes in the graph
        num_nodes = x.size(0)
        # create a fully connected edge_index
        dense_adj = torch.ones((num_nodes, num_nodes), dtype=torch.float)
        edge_index, _ = dense_to_sparse(dense_adj)
        edge_index = edge_index.to(self.device)
        # apply encoder
        x = F.relu(self.encoder(x))
        # pass through the first GAT layer
        x = F.relu(self.gat1(x, edge_index))
        # pass through the second GAT layer
        x = F.relu(self.gat2(x, edge_index))
        # apply linear layer for actor and critic
        logits = self.actor(x)
        logits = torch.clamp(logits, -5, 5)
        logits = torch.cat((logits[:, 0].unsqueeze(1), torch.abs(logits[:, 1]).unsqueeze(1)), dim=1)
        values = self.critic(x)
        self.values = values.view(-1)
        return logits, state

    def value_function(self):
        return self.values 
    
ModelCatalog.register_custom_model("gat_model", GATModel)

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

def curriculum_fn(train_results, task_settable_env, env_ctx):
    return tasks.sample_task()

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=args.num_workers, sample_timeout_s=3600, \
        batch_mode="complete_episodes", rollout_fragment_length=400)
    .resources(num_gpus=args.num_gpus)
    .evaluation(evaluation_num_workers=1, evaluation_duration=1, \
        evaluation_duration_unit='episodes', evaluation_force_reset_envs_before_iteration=True)
    .environment(
        env=IntersectionZooEnv,
        env_config={"intersectionzoo_env_config": env_conf},
        env_task_fn=curriculum_fn,
    )
    .multi_agent(
        policies={
            "default_policy": 
                (None, IntersectionZooEnv(config={"intersectionzoo_env_config": env_conf}).observation_space, 
                IntersectionZooEnv(config={"intersectionzoo_env_config": env_conf}).action_space, 
                {"model": {"custom_model": "gat_model"}})
        },
    )
    .training(model={"custom_model": "gat_model"},
              mini_batch_size_per_learner=32,
              num_sgd_iter=10,
              grad_clip=4)
    .callbacks(MetricsCallback)
    .build()
)

for i in range(args.rollouts):
    
    result = algo.train()
    
    print(f"iteration {i} completed.")
    
    sampler_results = result['sampler_results']
    custom_results = result['custom_metrics']
    metrics = {key: sampler_results[key] for key in sampler_results.keys()}
    metrics.update({key: custom_results[key] for key in custom_results.keys()})
    wandb.log(metrics, step=i)
    
    if i % args.save_frequency == 0:
        save_dir = f'{args.dir}/runs/{str(i)}/{datetime.now().strftime("%Y%m%d_%H%M")}'
        checkpoint_dir = algo.save(save_dir).checkpoint.path
        print(f"Checkpoint saved at {checkpoint_dir}")

wandb.finish()