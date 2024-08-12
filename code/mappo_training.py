import ray
import wandb
import argparse

from pathlib import Path
from datetime import datetime
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from sumo.constants import REGULAR
from env.config import IntersectionZooEnvConfig
from env.environment import IntersectionZooEnv
from env.task_context import PathTaskContext
from env.rllib_callback import MetricsCallback
import torch

parser = argparse.ArgumentParser(description='Model arguments')
parser.add_argument('--dir', default='wd/new_exp', type=str, help='Result directory')
parser.add_argument('--intersection_dir', default='/Users/bfreydt/MIT_local/IntersectionZoo/dataset/salt-lake-city', type=str, help='Path to intersection dataset')
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

wandb.init(project=args.wandb_project, entity=args.wandb_entity)

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

def curriculum_fn(train_results, task_settable_env, env_ctx):
    return tasks.sample_task()

# Define a custom centralized critic model
class CentralizedCriticModel(FullyConnectedNetwork):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.central_value_function = torch.nn.Linear(self.num_outputs, 1)

    def forward(self, input_dict, state, seq_lens):
        out, _ = super().forward(input_dict, state, seq_lens)
        return out, state

    def central_value(self, observations):
        """
        Compute the centralized value function for all observations.
        
        Args:
            observations: A dictionary of observations, one per agent.
            max_agents: The maximum number of agents expected (for padding).
        
        Returns:
            A single value for the entire observation set.
        """
        max_agents = 1000
        # Extract the observations from the dictionary
        agent_obs = list(observations.values())

        # Concatenate the observations
        concatenated_obs = torch.cat(agent_obs, dim=-1)

        # Determine the current number of agents
        current_agents = len(agent_obs)

        # Padding if necessary
        if max_agents and current_agents < max_agents:
            padding_size = (0, (max_agents - current_agents) * agent_obs[0].size(-1))
            concatenated_obs = torch.nn.functional.pad(concatenated_obs, padding_size)

        # Convert to torch tensor
        obs_tensor = convert_to_torch_tensor(concatenated_obs)

        # Pass through the central value function
        return self.central_value_function(obs_tensor).squeeze(1)

# Register the model
ModelCatalog.register_custom_model("centralized_critic_model", CentralizedCriticModel)

def policy_mapping_fn(agent_id, ss, worker):
    return "shared_policy"

# MAPPO Config
algo = (
    PPOConfig()
    .framework("torch")
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
    .training(
        model={"custom_model": "centralized_critic_model"},
    )
    .multi_agent(
            policies= {
                "shared_policy": (
                    None,  # Use default observation/action spaces
                    IntersectionZooEnv(config={"intersectionzoo_env_config": env_conf}).observation_space,
                    IntersectionZooEnv(config={"intersectionzoo_env_config": env_conf}).action_space,
                    {
                        "model": {
                            "custom_model": "centralized_critic_model",
                        },
                    },
                )
            },
            policy_mapping_fn= policy_mapping_fn,
            policies_to_train= ["shared_policy"],
    )
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