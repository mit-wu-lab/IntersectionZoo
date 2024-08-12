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

from collections import OrderedDict
import ray
import pandas as pd
import argparse

from pathlib import Path
from env.task_context import PathTaskContext
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from sumo.constants import REGULAR

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch_geometric.utils import dense_to_sparse
import numpy as np

parser = argparse.ArgumentParser(description='Model arguments')
parser.add_argument('--dir', default='/home/vindula/Desktop/IntersectionZoo/dataset/', type=str, help='Result directory')
parser.add_argument('--intersection_dir', default='/home/vindula/Desktop/IntersectionZoo/dataset/salt-lake-city', type=str, help='Path to intersection dataset')
parser.add_argument('--num_workers', default=5, type=str, help='Number of workers')
parser.add_argument('--checkpoint', default='/home/vindula/Desktop/IntersectionZoo/code/wd/new_exp/runs/5/20240811_1839', type=str, help='Checkpoint path')
parser.add_argument('--eval_per_task', default=3, type=str, help='How many times to evaluate each task')

parser.add_argument('--penetration', default=1.0, type=str, help='Eco drive adoption rate')
parser.add_argument('--temperature_humidity', default='20_50', type=str, help='Temperature and humidity for evaluations')

args = parser.parse_args()
print(args)

import sys
sys.path.append('/home/vindula/Desktop/IntersectionZoo/')
print(sys.path)

ray.init(ignore_reinit_error=True, num_cpus=args.num_workers + 15)

# function to convert OrderedDict to matrix
def ordered_dict_to_matrix(ordered_dict):
    matrices = []
    for key, value in ordered_dict.items():
        if isinstance(value, OrderedDict):
            # recursively handle nested OrderedDict
            sub_matrix = ordered_dict_to_matrix(value)
            matrices.append(sub_matrix)
        else:
            matrices.append(value.cpu())
    
    # concatenate all matrices along the second axis (columns)
    return np.hstack(matrices)

def transform_features(input_dict, device):
    features_dict = input_dict["obs"]
    # Convert the OrderedDict to a matrix
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

algo = Algorithm.from_checkpoint(args.checkpoint)

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

res_df = pd.DataFrame()

for i, task in enumerate(tasks.list_tasks(False)):
    for _ in range(args.eval_per_task):
    
        algo.evaluation_workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: env.set_task(task)))
        results = algo.evaluate()

        flattened_results = {**flatten_dict(results)}
        results_df = pd.DataFrame([flattened_results])
        res_df = pd.concat([res_df, results_df], ignore_index=True)
        
    print(f'Completed evaluation for task {i+1}/{len(tasks.list_tasks(False))}')

res_df.to_csv(f'{args.dir}/eval_result_pen_rate_{args.penetration}.csv')
print('Evaluation completed')
 