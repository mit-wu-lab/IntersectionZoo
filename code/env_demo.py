import argparse
from pathlib import Path

import numpy as np
from env.config import IntersectionZooEnvConfig
from env.task_context import PathTaskContext
from env.environment import IntersectionZooEnv
from sumo.constants import REGULAR

parser = argparse.ArgumentParser(description='Demo run arguments')
parser.add_argument('--dir', default='/Users/bfreydt/MIT_local/IntersectionZoo/wd/new_exp', type=str, help='Result directory')
parser.add_argument('--intersection_dir', default='/Users/bfreydt/MIT_local/IntersectionZoo/dataset/salt-lake-city', type=str, help='Path to intersection dataset')
parser.add_argument('--penetration', default=0.33, type=str, help='Eco drive adoption rate')
parser.add_argument('--temperature_humidity', default='68_46', type=str, help='Temperature and humidity for evaluations')

args = parser.parse_args()
print(args)

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

# Create the environment
env = IntersectionZooEnv({"intersectionzoo_env_config": env_conf})

def filter_obs(obs: dict):
    def simplify(v):
        if isinstance(v, np.ndarray):
            if len(v) == 1:
                return v[0]
            else:
                return v.tolist()
        else:
            return v

    return {k: {
        k2: simplify(v2) for k2, v2 in v.items() if k2 in ["speed", "relative_distance", "tl_phase"]
    } for k,v in obs.items() if k != "mock"}

def filter_rew(rew: dict):
    return {k: v for k,v in rew.items() if k != "mock"}

# Reset the environment
obs, _ = env.reset()
terminated = {"__all__": False}
while not terminated["__all__"]:
    # Send a constant action for all agents
    action = {agent: [1] for agent in obs.keys()}

    # Take a step in the environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Print the observations and reward
    print("Observations:", filter_obs(obs))
    print("Reward:", filter_rew(reward))
    input("Press Enter to continue...")

# Close the environment
env.close()