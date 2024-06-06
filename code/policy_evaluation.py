import ray
import pandas as pd
import argparse

from pathlib import Path
from env.task_context import PathTaskContext
from ray.rllib.algorithms.algorithm import Algorithm
from sumo.constants import REGULAR

parser = argparse.ArgumentParser(description='Model arguments')
parser.add_argument('--dir', default='/home/gridsan/vindula/neurips24/scenarioenv', type=str, help='Result directory')
parser.add_argument('--intersection_dir', default='/home/gridsan/vindula/neurips24/scenarioenv/dataset/test', type=str, help='Path to intersection dataset')
parser.add_argument('--num_workers', default=5, type=str, help='Number of workers')
parser.add_argument('--checkpoint', default='/home/gridsan/vindula/neurips24/scenarioenv/wd/ppo_run/runs/0/20240530_2354', type=str, help='Checkpoint path')
parser.add_argument('--eval_per_task', default=3, type=str, help='How many times to evaluate each task')

parser.add_argument('--penetration', default=1.0, type=str, help='Eco drive adoption rate')
parser.add_argument('--temperature_humidity', default='20_50', type=str, help='Temperature and humidity for evaluations')

args = parser.parse_args()
print(args)

ray.init(ignore_reinit_error=True, num_cpus=args.num_workers + 15)

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
 