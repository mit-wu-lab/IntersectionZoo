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

import random
import numpy as np
import pandas as pd

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Optional, Set, Tuple
from pathlib import Path

from env.config import IntersectionZooEnvConfig
from env.task_context import PathTaskContext, TaskContext
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Discrete
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict, MultiEnvDict
from sumo.constants import (CYAN, ELECTRIC, GLOBAL_MAX_LANE_LENGTH,
                            GLOBAL_MAX_SPEED, LANE_LENGTH_NORMALIZATION,
                            MAX_TL_CYCLE, RED, REGULAR, SPEED_NORMALIZATION,
                            TL_CYCLE_NORMALIZATION)
from sumo.sumo import start_sumo
from sumo.traffic_state import TrafficState, Vehicle
from sumo.utils import get_remaining_time, is_internal_lane, is_rl


class IntersectionZooEnv(MultiAgentEnv, TaskSettableEnv):
    """
    RL env interfacing Rllib multi agent API with SUMO.
    """

    def __init__(self, config: EnvContext):
        super().__init__()
        self.config: IntersectionZooEnvConfig = config["intersectionzoo_env_config"]
        self.task_context: TaskContext | None = self.config.task_context
        self.prefix: str = str(config.worker_index) if hasattr(config, "worker_index") else "0"
        self.traci = None
        self.traffic_state: Optional[TrafficState] = None
        self._curr_step = 0
        self.vehicle_metrics = defaultdict(lambda: defaultdict(lambda: []))
        self.uncontrolled_region_metrics = defaultdict(lambda: [])
        self.warmup_vehicles: Set[str] = set()
        self.agents: Set[str] = {0}
        self._agent_ids = self.agents
        self.max_num_agents = 1000
        self.possible_agents = list(range(self.max_num_agents))
        self.agents_id_mapping = {'mock': 0}
        self.inverse_agents_id_mapping = {0: 'mock'}
        self.num_agents_appeard = 1

        self.action_spaces = defaultdict(lambda:
            GymDict(
                {
                    # the boundaries of box are not followed, -20 and 10 should include all possible vehicle types
                    # boundary conditions are set seperately.
                    "accel": Box(low=-20, high=10, shape=(1,), dtype=np.float32),
                    "lane_change": Discrete(3),  # left, stay, right
                }
            )
            if self.config.control_lane_change
            else Discrete(30)
        )
        self.action_space = self.action_spaces[0]

        speed_space = Box(npa(0), npa(GLOBAL_MAX_SPEED / SPEED_NORMALIZATION))

        other_vehicle_space = GymDict(
            {
                "speed": speed_space,
                "relative_position": Box(
                    npa(-1), npa(GLOBAL_MAX_LANE_LENGTH / LANE_LENGTH_NORMALIZATION)
                ),
                "blinker_left": Discrete(2),
                "blinker_right": Discrete(2),
            }
        )

        self.observation_spaces = defaultdict(lambda: GymDict(
            {
                "speed": speed_space,
                "relative_distance": Box(
                    npa(-GLOBAL_MAX_LANE_LENGTH / LANE_LENGTH_NORMALIZATION),
                    npa(GLOBAL_MAX_LANE_LENGTH / LANE_LENGTH_NORMALIZATION),
                ),
                "tl_phase": Discrete(3),
                "time_remaining": Box(
                    npa(0), npa(MAX_TL_CYCLE / TL_CYCLE_NORMALIZATION)
                ),
                "time_remaining2": Box(-npa(2 * MAX_TL_CYCLE), npa(2 * MAX_TL_CYCLE)),
                "time_remaining3": Box(-npa(3 * MAX_TL_CYCLE), npa(3 * MAX_TL_CYCLE)),
                "edge_id": Discrete(
                    3
                ),  # 4 incoming, 4 outgoing, and 1 for internal lanes
                "follower": other_vehicle_space,
                "leader": other_vehicle_space,
                "lane_index": Box(npa(0), npa(1)),
                "destination": Discrete(3),  # left, straight or right
                "leader_left": other_vehicle_space,
                "leader_right": other_vehicle_space,
                "follower_left": other_vehicle_space,
                "follower_right": other_vehicle_space,
                # context, stay constant
                "penetration_rate": Box(npa(0), npa(1)),
                "lane_length": Box(
                    npa(0), npa(GLOBAL_MAX_LANE_LENGTH / LANE_LENGTH_NORMALIZATION)
                ),
                "speed_limit": speed_space,
                "green_phase": Box(npa(0), npa(120)),
                "red_phase": Box(npa(0), npa(120)),
                "temperature": Box(npa(0), npa(100)),
                "humidity": Box(npa(0), npa(100)),
                "electric": Discrete(2),
            }
        ))
        self.observation_space = self.observation_spaces[0]

    @property
    def num_agents(self) -> int:
        return len(self.agents)
    
    # def action_space(self, agent: AgentID = None) -> GymDict:
    #     return self.action_spaces[agent]
    
    # def observation_space(self, agent: AgentID = None) -> GymDict:
    #     return self.observation_spaces[agent]

    def state(self) -> None:
        raise NotImplementedError

    def reset(
        self,
        seed: int | None = None,
        options: Dict | None = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """
        Custom Env class reset method.
        """
        super().reset(seed=seed, options=options)
        self.traci = start_sumo(
            self.config,
            self.traci,
            self.prefix,
            self.get_task(),
            seed if seed is not None else random.randint(0, 9999),
        )
        self.traffic_state = TrafficState(
            self.config,
            self.traci,
            self.task_context,
            self.config.working_dir / f"sumo/net{self.prefix}.net.xml",
        )

        self._curr_step = 0
        self.vehicle_metrics = defaultdict(lambda: defaultdict(lambda: []))
        self.uncontrolled_region_metrics = defaultdict(lambda: [])
        self.warmup_vehicles = set()
        self.agents = {0}
        self._agent_ids = self.agents
        self.agents_id_mapping = {'mock': 0}
        self.inverse_agents_id_mapping = {0: 'mock'}
        self.num_agents_appeard = 1

        obs, _, _, _, _ = self.step({}, warmup=True)
        for _ in range(self.config.warmup_steps):
            obs, _, _, _, _ = self.step({}, warmup=True)

        return obs, {}

    def step(
        self, action_dict: MultiAgentDict, warmup: bool = False
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        if not warmup:
            for v_id, action in action_dict.items():
                if v_id != 0:
                    v_id = self.inverse_agents_id_mapping[v_id]
                    vehicle = self.traffic_state.vehicles[v_id]

                    sumo_speed = self.traci.vehicle.getSpeedWithoutTraCI(v_id)
                    diff = sumo_speed - vehicle.previous_step_idm_speed
                    vehicle.previous_step_idm_speed = (
                        self.traffic_state.get_idm_accel(vehicle)
                        * self.config.sim_step_duration
                        + vehicle.speed
                    )

                    if self.config.control_lane_change:
                        if not is_internal_lane(vehicle.lane_id):
                            vehicle.change_lane_relative(action["lane_change"] - 1)
                        self.traffic_state.accel(
                            vehicle, action["accel"]/5 - 3, use_speed_factor=False
                        )
                    else:
                        # We only apply RL accel to the vehicle if SUMO doesn't slow down the vehicle to do a lane change
                        # We identify if SUMO slows down to do lane change by computing theoretical IDM speed, and taking the
                        # diff with the speed SUMO actually wants to apply.
                        if abs(diff) < 0.5 or abs(sumo_speed - vehicle.speed) < 0.5:
                            self.traffic_state.accel(
                                vehicle, action/5-3, use_speed_factor=False
                            )

                    # color vehicles according to whether they'll turn at the intersection
                    if vehicle.direction != 0:
                        self.traffic_state.set_color(vehicle, CYAN)
                    else:
                        self.traffic_state.set_color(vehicle, RED)

        self.traffic_state.step()
        self._curr_step += 1
        self._collect_metrics(warmup)

        reward = self._get_reward(action_dict.keys())

        for v_id in self.traffic_state.current_vehicles:
            if v_id not in self.agents_id_mapping:
                if self.num_agents_appeard >= self.max_num_agents:
                    raise NotImplementedError("Too many agents appeared")
                self.agents_id_mapping[v_id] = self.num_agents_appeard
                self.inverse_agents_id_mapping[self.num_agents_appeard] = v_id
                self.num_agents_appeard += 1

        self.agents = {
            self.agents_id_mapping[v_id]
            for v_id, vehicle in self.traffic_state.current_vehicles.items()
            if vehicle.is_rl
        }.union({0})
        self._agent_ids = self.agents

        obs = self._get_obs({
            v_id
            for v_id, vehicle in self.traffic_state.current_vehicles.items()
            if vehicle.is_rl
        }.union({'mock'}))

        v_finished_during_step = {
            v_id for v_id in action_dict.keys() if v_id not in self.agents
        }

        # We consider the simulation over is too many vehicles are present, and they are too slow
        if (
            not warmup
            and len(self.agents) > 150
            and self._get_average_speed() < 0.5
        ):
            termination = True
            truncation = True
        elif (
            self._curr_step
            > self.config.simulation_duration / self.config.sim_step_duration
        ):
            termination = True
            truncation = True
        else:
            termination = False
            truncation = False

        terminated = {
            **{v_id: termination for v_id in self.agents},
            **{v_id: True for v_id in v_finished_during_step},
        }
        terminated["__all__"] = termination
        truncated = {v_id: truncation for v_id in terminated}

        # if self._curr_step < self.config.warmup_steps + 1:
        #     self._agent_ids.add('mock')
        # elif self._curr_step == self.config.warmup_steps + 1:
        #     return self.observation_space_sample(), {'mock': 0}, {'mock': True}, {'mock': False}, {}
        return obs, reward, terminated, truncated, {}

    def close(self):
        """
        Closes the env.
        """
        self.traci.close()

    def get_metrics(self) -> Dict[str, any]:
        """
        Supposed to be called at the end of the episode, returns the metric of the episode and resets the internal state
        """
        metrics = self._filtered_metrics()

        num_vehicles = len(metrics)

        approach_speed_means = []
        for _, series in metrics.items():
            if series["approach_speed"] != []:
                approach_speed_means.append(mean(series["approach_speed"]))

        int_speed_means = []
        for _, series in metrics.items():
            if series["int_speed"] != []:
                int_speed_means.append(mean(series["int_speed"]))

        leaving_speed_means = []
        for _, series in metrics.items():
            if series["leaving_speed"] != []:
                leaving_speed_means.append(mean(series["leaving_speed"]))

        if approach_speed_means != []:
            approach_speed_mean = mean(approach_speed_means)
        else:
            approach_speed_mean = 10e6

        if int_speed_means != []:
            int_speed_mean = mean(int_speed_means)
        else:
            int_speed_mean = 10e6

        if leaving_speed_means != []:
            leaving_speed_mean = mean(leaving_speed_means)
        else:
            leaving_speed_mean = 10e6

        aggregated_metrics = (
            {
                "num_vehicle_completed": num_vehicles,
                "num_vehicle": sum(
                    series["int_crossed"][-1] for _, series in metrics.items()
                ),
                "vehicle_speed": mean(
                    mean(series["speed"]) for _, series in metrics.items()
                ),
                "approach_vehicle_speed": approach_speed_mean,
                "int_vehicle_speed": int_speed_mean,
                "leaving_vehicle_speed": leaving_speed_mean,
                "stopping_time": mean(
                    sum(1 for s in series["speed"] if s < 1)
                    * self.config.sim_step_duration
                    for _, series in metrics.items()
                ),
                **{
                    f"regular_vehicle_emission_{condition}": mean(
                        sum(series[f"emission_{condition}_{REGULAR}"])
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == REGULAR
                },
                **{
                    f"electric_vehicle_emission_{condition}": mean(
                        sum(series[f"emission_{condition}_{ELECTRIC}"])
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == ELECTRIC
                },
                **{
                    f"regular_approach_vehicle_emission_{condition}": mean(
                        sum(series[f"approach_emission_{condition}_{REGULAR}"])
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == REGULAR
                },
                **{
                    f"regular_approach_vehicle_emission_idling_{condition}": mean(
                        sum(series[f"approach_idling_emission_{condition}_{REGULAR}"])
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == REGULAR
                },
                **{
                    f"electric_approach_vehicle_emission_{condition}": mean(
                        sum(series[f"approach_emission_{condition}_{ELECTRIC}"])
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == ELECTRIC
                },
                **{
                    f"electric_approach_vehicle_emission_idling_{condition}": mean(
                        sum(series[f"approach_idling_emission_{condition}_{ELECTRIC}"])
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == ELECTRIC
                },
                **{
                    f"regular_leaving_vehicle_emission_{condition}": mean(
                        sum(series[f"leaving_emission_{condition}_{REGULAR}"])
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == REGULAR
                },
                **{
                    f"regular_leaving_vehicle_emission_idling_{condition}": mean(
                        sum(series[f"leaving_idling_emission_{condition}_{REGULAR}"])
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == REGULAR
                },
                **{
                    f"electric_leaving_vehicle_emission_{condition}": mean(
                        sum(series[f"leaving_emission_{condition}_{ELECTRIC}"])
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == ELECTRIC
                },
                **{
                    f"electric_leaving_vehicle_emission_idling_{condition}": mean(
                        sum(series[f"leaving_idling_emission_{condition}_{ELECTRIC}"])
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == ELECTRIC
                },
                **{
                    f"regular_intersection_vehicle_emission_{condition}": mean(
                        sum(series[f"intersection_emission_{condition}_{REGULAR}"])
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == REGULAR
                },
                **{
                    f"regular_intersection_vehicle_emission_idling_{condition}": mean(
                        sum(
                            series[
                                f"intersection_idling_emission_{condition}_{REGULAR}"
                            ]
                        )
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == REGULAR
                },
                **{
                    f"electric_intersection_vehicle_emission_{condition}": mean(
                        sum(series[f"intersection_emission_{condition}_{ELECTRIC}"])
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == ELECTRIC
                },
                **{
                    f"electric_intersection_vehicle_emission_idling_{condition}": mean(
                        sum(
                            series[
                                f"intersection_idling_emission_{condition}_{ELECTRIC}"
                            ]
                        )
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == ELECTRIC
                },
                **{
                    f"effective_uncontrolled_regular_vehicle_emission_{condition}": mean(
                        sum(
                            series[
                                f"effective_uncontrolled_emission_{condition}_{REGULAR}"
                            ]
                        )
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == REGULAR
                },
                **{
                    f"effective_uncontrolled_electric_vehicle_emission_{condition}": mean(
                        sum(
                            series[
                                f"effective_uncontrolled_emission_{condition}_{ELECTRIC}"
                            ]
                        )
                        for _, series in metrics.items()
                    )
                    for i, condition in enumerate(self.config.moves_emissions_models)
                    if self.config.moves_emissions_models_conditions[i] == ELECTRIC
                },
                "vehicle_fuel": mean(
                    sum(series["fuel"]) for _, series in metrics.items()
                ),
                "vehicle_accel_squared": mean(
                    mean(acc**2 for acc in series["accel"])
                    for _, series in metrics.items()
                ),
                "lane_changes": mean(
                    sum(series["lane_change"]) for _, series in metrics.items()
                ),
                "removed_vehicles": sum(
                    sum(series["removed"]) for _, series in metrics.items()
                ),
            }
            if num_vehicles > 0
            else {"num_vehicle": num_vehicles}
        )

        if (
            self.task_context.penetration_rate != 1
            and len([v_id for v_id in metrics if v_id.startswith("rl")]) > 0
        ):

            rl_vehicle_accel_mean = []
            rl_vehicle_decel_mean = []
            rl_vehicle_decel_max = []
            rl_vehicle_accel_max = []
            rl_vehicle_stopping_time = []

            for v_id, series in metrics.items():
                if is_rl(v_id):
                    try:
                        acc = mean(acc for acc in series["accel"] if acc >= 0)
                        rl_vehicle_accel_mean.append(acc)
                    except:
                        continue

                    try:
                        decel = mean(acc for acc in series["accel"] if acc < 0)
                        rl_vehicle_decel_mean.append(decel)
                    except:
                        continue

                    try:
                        decel_max = min(acc for acc in series["accel"] if acc < 0)
                        rl_vehicle_decel_max.append(decel_max)
                    except:
                        continue

                    try:
                        accel_max = max(acc for acc in series["accel"] if acc >= 0)
                        rl_vehicle_accel_max.append(accel_max)
                    except:
                        continue

                    try:
                        stopping_time = (
                            sum(1 for s in series["speed"] if s < 1)
                            * self.config.sim_step_duration
                        )
                        rl_vehicle_stopping_time.append(stopping_time)
                    except:
                        continue

            aggregated_metrics.update(
                {
                    "rl_vehicle_speed": mean(
                        mean(series["speed"])
                        for v_id, series in metrics.items()
                        if is_rl(v_id)
                    ),
                    **{
                        f"rl_vehicle_emission_{condition}": mean(
                            sum(series[f"emission_{condition}_{REGULAR}"])
                            for v_id, series in metrics.items()
                            if is_rl(v_id)
                        )
                        for i, condition in enumerate(
                            self.config.moves_emissions_models
                        )
                        if self.config.moves_emissions_models_conditions[i] == REGULAR
                    },
                    **{
                        f"rl_approaching_vehicle_emission_{condition}": mean(
                            sum(series[f"approach_emission_{condition}_{REGULAR}"])
                            for v_id, series in metrics.items()
                            if is_rl(v_id)
                        )
                        for i, condition in enumerate(
                            self.config.moves_emissions_models
                        )
                        if self.config.moves_emissions_models_conditions[i] == REGULAR
                    },
                    **{
                        f"rl_leaving_vehicle_emission_{condition}": mean(
                            sum(series[f"leaving_emission_{condition}_{REGULAR}"])
                            for v_id, series in metrics.items()
                            if is_rl(v_id)
                        )
                        for i, condition in enumerate(
                            self.config.moves_emissions_models
                        )
                        if self.config.moves_emissions_models_conditions[i] == REGULAR
                    },
                    **{
                        f"rl_intersection_vehicle_emission_{condition}": mean(
                            sum(series[f"intersection_emission_{condition}_{REGULAR}"])
                            for v_id, series in metrics.items()
                            if is_rl(v_id)
                        )
                        for i, condition in enumerate(
                            self.config.moves_emissions_models
                        )
                        if self.config.moves_emissions_models_conditions[i] == REGULAR
                    },
                    "rl_vehicle_fuel": mean(
                        sum(series["fuel"])
                        for v_id, series in metrics.items()
                        if is_rl(v_id)
                    ),
                    "rl_vehicle_accel_squared": mean(
                        mean(acc**2 for acc in series["accel"])
                        for v_id, series in metrics.items()
                        if is_rl(v_id)
                    ),
                }
            )

            if len(rl_vehicle_accel_mean) > 0:
                aggregated_metrics.update(
                    {"rl_vehicle_accel_mean": mean(rl_vehicle_accel_mean)}
                )
            if len(rl_vehicle_decel_mean) > 0:
                aggregated_metrics.update(
                    {"rl_vehicle_decel_mean": mean(rl_vehicle_decel_mean)}
                )
            if len(rl_vehicle_decel_max) > 0:
                aggregated_metrics.update(
                    {"rl_vehicle_decel_max": mean(rl_vehicle_decel_max)}
                )
            if len(rl_vehicle_accel_max) > 0:
                aggregated_metrics.update(
                    {"rl_vehicle_accel_max": mean(rl_vehicle_accel_max)}
                )

        if (
            self.task_context.penetration_rate != 1
            and len([v_id for v_id in metrics if not v_id.startswith("rl")]) > 0
        ):

            non_rl_vehicle_accel_mean = []
            non_rl_vehicle_decel_mean = []
            non_rl_vehicle_decel_max = []
            non_rl_vehicle_accel_max = []
            non_rl_vehicle_stopping_time = []

            for v_id, series in metrics.items():
                if not is_rl(v_id):
                    try:
                        acc = mean(acc for acc in series["accel"] if acc >= 0)
                        non_rl_vehicle_accel_mean.append(acc)
                    except:
                        continue

                    try:
                        decel = mean(acc for acc in series["accel"] if acc < 0)
                        non_rl_vehicle_decel_mean.append(decel)
                    except:
                        continue

                    try:
                        decel_max = min(acc for acc in series["accel"] if acc < 0)
                        non_rl_vehicle_decel_max.append(decel_max)
                    except:
                        continue

                    try:
                        accel_max = max(acc for acc in series["accel"] if acc >= 0)
                        non_rl_vehicle_accel_max.append(accel_max)
                    except:
                        continue

                    try:
                        stopping_time = (
                            sum(1 for s in series["speed"] if s < 1)
                            * self.config.sim_step_duration
                        )
                        non_rl_vehicle_stopping_time.append(stopping_time)
                    except:
                        continue

            aggregated_metrics.update(
                {
                    "non_rl_vehicle_speed": mean(
                        mean(series["speed"])
                        for v_id, series in metrics.items()
                        if not is_rl(v_id)
                    ),
                    **{
                        f"non_rl_vehicle_emission_{condition}": mean(
                            sum(series[f"emission_{condition}_{REGULAR}"])
                            for v_id, series in metrics.items()
                            if not is_rl(v_id)
                        )
                        for i, condition in enumerate(
                            self.config.moves_emissions_models
                        )
                        if self.config.moves_emissions_models_conditions[i] == REGULAR
                    },
                    **{
                        f"non_rl_approaching_vehicle_emission_{condition}": mean(
                            sum(series[f"approach_emission_{condition}_{REGULAR}"])
                            for v_id, series in metrics.items()
                            if not is_rl(v_id)
                        )
                        for i, condition in enumerate(
                            self.config.moves_emissions_models
                        )
                        if self.config.moves_emissions_models_conditions[i] == REGULAR
                    },
                    **{
                        f"non_rl_leaving_vehicle_emission_{condition}": mean(
                            sum(series[f"leaving_emission_{condition}_{REGULAR}"])
                            for v_id, series in metrics.items()
                            if not is_rl(v_id)
                        )
                        for i, condition in enumerate(
                            self.config.moves_emissions_models
                        )
                        if self.config.moves_emissions_models_conditions[i] == REGULAR
                    },
                    **{
                        f"non_rl_intersection_vehicle_emission_{condition}": mean(
                            sum(series[f"intersection_emission_{condition}_{REGULAR}"])
                            for v_id, series in metrics.items()
                            if not is_rl(v_id)
                        )
                        for i, condition in enumerate(
                            self.config.moves_emissions_models
                        )
                        if self.config.moves_emissions_models_conditions[i] == REGULAR
                    },
                    "non_rl_vehicle_accel_squared": mean(
                        mean(acc**2 for acc in series["accel"])
                        for v_id, series in metrics.items()
                        if not is_rl(v_id)
                    ),
                }
            )
            if len(non_rl_vehicle_accel_mean) > 0:
                aggregated_metrics.update(
                    {"non_rl_vehicle_accel_mean": mean(non_rl_vehicle_accel_mean)}
                )
            if len(non_rl_vehicle_decel_mean) > 0:
                aggregated_metrics.update(
                    {"non_rl_vehicle_decel_mean": mean(non_rl_vehicle_decel_mean)}
                )
            if len(non_rl_vehicle_decel_max) > 0:
                aggregated_metrics.update(
                    {"non_rl_vehicle_decel_max": mean(non_rl_vehicle_decel_max)}
                )
            if len(non_rl_vehicle_accel_max) > 0:
                aggregated_metrics.update(
                    {"non_rl_vehicle_accel_max": mean(non_rl_vehicle_accel_max)}
                )

        if self.config.report_uncontrolled_region_metrics and num_vehicles > 0:
            aggregated_metrics.update(
                {
                    "uncontrolled_waiting": sum(
                        self.uncontrolled_region_metrics["waiting"]
                    )
                    / num_vehicles,
                    **{
                        f"regular_uncontrolled_emission_{condition}": mean(
                            sum(series[f"emission_{condition}_{REGULAR}"])
                            for _, series in metrics.items()
                        )
                        for i, condition in enumerate(
                            self.config.moves_emissions_models
                        )
                        if self.config.moves_emissions_models_conditions[i] == REGULAR
                    },
                    **{
                        f"electric_uncontrolled_emission_{condition}": mean(
                            sum(series[f"emission_{condition}_{REGULAR}"])
                            for _, series in metrics.items()
                        )
                        for i, condition in enumerate(
                            self.config.moves_emissions_models
                        )
                        if self.config.moves_emissions_models_conditions[i] == ELECTRIC
                    },
                }
            )
            
        def map_str_to_int(s: str) -> int:
            if s == 'A': return 0
            elif s == 'B': return 1
            elif s == 'C': return 2
            elif s == 'D': return 3
            elif s == 'E': return 4
            elif s == 'F': return 5
            elif s == 'G': return 6
            elif s == 'H': return 7
            elif s == 'I': return 8
            elif s == ELECTRIC : return 100
            elif s == REGULAR: return 200
            else: return -1
            
        aggregated_metrics.update(
            {"temperature": int(self.task_context.temperature_humidity.split("_")[0])},)
        aggregated_metrics.update(    
            {"humidity": int(self.task_context.temperature_humidity.split("_")[1])},)
        aggregated_metrics.update(
            {"electric": map_str_to_int(self.task_context.electric_or_regular == ELECTRIC)},)
        aggregated_metrics.update(
            {"penetration_rate": self.task_context.penetration_rate},)
        aggregated_metrics.update(
            {"single_approach": map_str_to_int(self.task_context.single_approach)},)
        if isinstance(self.task_context, PathTaskContext):
            aggregated_metrics.update(
                {"intersection_id": int((self.task_context.dir).name)},)
        else:
            aggregated_metrics.update(
                {"base_id": int((self.task_context.base_id))},)
            aggregated_metrics.update(
                {"inflow": int((self.task_context.inflow))},)
            aggregated_metrics.update(
                {"green_phase": int((self.task_context.green_phase))},)
            aggregated_metrics.update(
                {"red_phase": int((self.task_context.red_phase))},)
            aggregated_metrics.update(
                {"lane_length": int((self.task_context.lane_length))},)
            aggregated_metrics.update(
                {"speed_limit": int((self.task_context.speed_limit))},)
            aggregated_metrics.update(
                {"offset": int((self.task_context.offset))},)

        return aggregated_metrics

    def export_moves_csv(self, file: Path) -> None:
        """
        Export run data to the format used by the MOVES script.
        """
        metrics = self._filtered_metrics()

        def keep_point(index: int):
            step_duration = self.config.sim_step_duration
            timing = index * step_duration
            return int(timing) < int(timing + step_duration)

        pd.concat(
            [
                pd.DataFrame(
                    {
                        "trajectory_id": i,
                        "drive_cycle": str(
                            [
                                s
                                for j, s in enumerate(vehicle_metrics["speed"])
                                if keep_point(j)
                            ]
                        ),
                        "grade": mean(vehicle_metrics["slope"]),
                        "vehicle_type": 21,
                        "age": 5,
                    },
                    index=[i],
                )
                for i, vehicle_metrics in enumerate(metrics.values())
            ]
        ).to_csv(file)

    def get_task(self) -> TaskContext:
        assert (
            self.task_context is not None
        ), "You must define a TaskContext for the env to function."
        return self.task_context

    def set_task(self, task: TaskContext) -> None:
        self.task_context = task

    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        return {v_id: self.action_space.sample() for v_id in self.agents}

    def observation_space_sample(self, agent_ids: list = None) -> MultiEnvDict:
        return {v_id: self.observation_space.sample() for v_id in self.agents}

    def action_space_contains(self, x: MultiAgentDict) -> bool:
        return all(self.action_space.contains(v) for v in x.values())

    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        return all(self.observation_space.contains(v) for v in x.values())

    def _get_average_speed(self) -> float:
        """
        Return the average speed of the vehicles in the main intersection. Does NOT work during warmup.
        """
        return mean(
            self.vehicle_metrics[v]["speed"][-1]
            for v in self.traffic_state.current_vehicles
        )

    def _get_obs(self, current_rl_vehicle_list: set[str]) -> Dict[AgentID, dict]:
        time = self.traffic_state.traci_module.simulation.getTime()
        return {
            self.agents_id_mapping[vehicle_id]: self._get_vehicle_obs(
                self.traffic_state.vehicles[vehicle_id], time
            ) if vehicle_id != 'mock' else self.observation_space.sample()
            for vehicle_id in current_rl_vehicle_list
        }

    def _get_vehicle_obs(self, vehicle: Vehicle, time: float) -> Dict[str, float]:
        """
        Returns the obs dict for a single vehicle
        """
        current_phase = self.traffic_state.get_phase("TL")
        lane_index = vehicle.lane_index

        if lane_index > 0:
            leader_right = self.traffic_state.get_leader(vehicle, side_lane=-1)
            follower_right = self.traffic_state.get_follower(vehicle, side_lane=-1)
        else:
            leader_right = vehicle
            follower_right = vehicle

        if (
            vehicle.edge_id not in self.traffic_state.lane_counts
            or lane_index >= self.traffic_state.lane_counts[vehicle.edge_id] - 1
        ):
            leader_left = vehicle
            follower_left = vehicle
        else:
            leader_left = self.traffic_state.get_leader(vehicle, side_lane=1)
            follower_left = self.traffic_state.get_follower(vehicle, side_lane=1)

        def get_other_vehicle_obs(other_veh: Optional[Vehicle]) -> Dict[str, any]:
            if other_veh is None:
                return {
                    "speed": npa(GLOBAL_MAX_SPEED / SPEED_NORMALIZATION),
                    "relative_position": npa(
                        GLOBAL_MAX_LANE_LENGTH / LANE_LENGTH_NORMALIZATION
                    ),
                    "blinker_left": False,
                    "blinker_right": False,
                }
            elif other_veh.id == vehicle.id:
                return {
                    "speed": npa(vehicle.speed / SPEED_NORMALIZATION),
                    "relative_position": npa(0),
                    "blinker_left": False,
                    "blinker_right": False,
                }
            else:
                return {
                    "speed": npa(other_veh.speed / SPEED_NORMALIZATION),
                    "relative_position": npa(
                        self.traffic_state.get_linear_distance(other_veh, vehicle)
                        / LANE_LENGTH_NORMALIZATION
                    ),
                    "blinker_left": other_veh.turn_signal == 1,
                    "blinker_right": other_veh.turn_signal == -1,
                }

        green_phase_transition = (
            get_remaining_time(vehicle.green_phase_timings, time)
            if current_phase == vehicle.green_phase_index
            else get_remaining_time(vehicle.green_phase_timings, time)
            - vehicle.green_phase_timings[1]
        )
        obs = {
            "speed": npa(vehicle.speed / SPEED_NORMALIZATION),
            "relative_distance": npa(
                vehicle.relative_distance / LANE_LENGTH_NORMALIZATION
            ),
            "tl_phase": (
                0
                if current_phase == vehicle.green_phase_index
                else (1 if current_phase == vehicle.green_phase_index + 1 else 2)
            ),
            "time_remaining": npa(get_remaining_time(vehicle.green_phase_timings, time))
            / TL_CYCLE_NORMALIZATION,
            "time_remaining2": npa(
                green_phase_transition
                + sum(vehicle.green_phase_timings) / TL_CYCLE_NORMALIZATION
            ),
            "time_remaining3": npa(
                green_phase_transition
                + sum(vehicle.green_phase_timings) * 2 / TL_CYCLE_NORMALIZATION
            ),
            # E and W: 0, S and N: 1, exiting lanes: 2, internal lanes: 3
            "edge_id": (
                0
                if vehicle.edge_id.endswith("2TL")
                else (1 if is_internal_lane(vehicle.lane_id) else 2)
            ),
            "follower": get_other_vehicle_obs(self.traffic_state.get_follower(vehicle)),
            "leader": get_other_vehicle_obs(self.traffic_state.get_leader(vehicle)),
            "lane_index": npa(
                min(
                    lane_index
                    / max(self.traffic_state.lane_counts[vehicle.edge_id] - 1, 1),
                    1,
                )
                if vehicle.edge_id in self.traffic_state.lane_counts
                else 0.5
            ),
            # 0 for right turn, 1 for no turn necessary, 2 for left turn
            "destination": vehicle.direction + 1,
            "leader_right": get_other_vehicle_obs(leader_right),
            "follower_right": get_other_vehicle_obs(follower_right),
            "leader_left": get_other_vehicle_obs(leader_left),
            "follower_left": get_other_vehicle_obs(follower_left),
            # context, stays constant
            "penetration_rate": npa(self.task_context.penetration_rate),
            "green_phase": npa(vehicle.green_phase_timings[1]),
            "red_phase": npa(
                vehicle.green_phase_timings[0] + vehicle.green_phase_timings[2]
            ),
            "speed_limit": npa(
                self.traffic_state.get_speed_limit(vehicle.origin + "2TL_0")
                / SPEED_NORMALIZATION
            ),
            "lane_length": npa(
                self.traffic_state.get_lane_length(vehicle.origin + "2TL_0")
                / LANE_LENGTH_NORMALIZATION
            ),
            "temperature": npa(self.task_context.temperature_humidity.split("_")[0]),
            "humidity": npa(self.task_context.temperature_humidity.split("_")[1]),
            "electric": self.task_context.electric_or_regular == ELECTRIC,
        }
        return obs

    def _get_reward(self, vehicle_list: set[str]) -> MultiAgentDict:
        """Compute the reward of the previous action."""

        def individual_reward(veh: Vehicle) -> float:
            penalty = 0

            speed = veh.speed
            threshold = self.config.threshold
            emission = 0

            if self.config.stop_penalty is not None and speed < threshold:
                penalty += self.config.stop_penalty * (threshold - speed) / threshold

            if self.config.accel_penalty is not None:
                penalty += self.config.accel_penalty * abs(veh.accel)

            if self.config.emission_penalty is not None:
                penalty += self.config.emission_penalty * emission

            return (speed - penalty) / SPEED_NORMALIZATION

        fleet_rewards = {
            k1: {k2: mean(individual_reward(v) for v in v2) for k2, v2 in v1.items()}
            for k1, v1 in self.traffic_state.current_vehicles_sorted_lists.items()
        }

        result = {}
        num_stopped_vehicles = 0

        for v_id in vehicle_list:
            real_v_id = self.inverse_agents_id_mapping[v_id]
            if real_v_id == 'mock':
                result[v_id] = 0
            else:
                vehicle = self.traffic_state.vehicles[real_v_id]
                num_stopped_vehicles += vehicle.speed < self.config.threshold
                result[v_id] = (
                    fleet_rewards[vehicle.platoon][vehicle.lane_index]
                    if (
                        random.random() < self.config.fleet_reward_ratio
                        and vehicle.platoon in fleet_rewards
                        and vehicle.lane_index in fleet_rewards[vehicle.platoon]
                    )
                    else individual_reward(vehicle)
                )
                if self.config.fleet_stop_penalty is not None:
                    result[v_id] -= (
                        self.config.fleet_stop_penalty
                        * num_stopped_vehicles
                        / len(vehicle_list)
                    )

        return result

    def _filtered_metrics(self) -> dict:
        """
        Returns the metrics for vehicles outside the warmup and who completely crossed the intersection.
        """
        return {
            v_id: series
            for v_id, series in self.vehicle_metrics.items()
            if (
                v_id not in self.warmup_vehicles
                and self.traffic_state.vehicles[v_id]
                in self.traffic_state.completed_vehicle
            )
        }

    def _collect_metrics(self, warmup) -> None:
        """
        Supposed to be called at each step, updates the internal metrics
        """
        if warmup:
            for v_id in self.traffic_state.current_vehicles:
                self.warmup_vehicles.add(v_id)
        else:
            for v_id, vehicle in self.traffic_state.current_vehicles.items():
                self.vehicle_metrics[v_id]["speed"].append(vehicle.speed)
                if (
                    vehicle.lane_id.startswith("A2TL")
                    or vehicle.lane_id.startswith("B2TL")
                    or vehicle.lane_id.startswith("C2TL")
                    or vehicle.lane_id.startswith("D2TL")
                    or vehicle.lane_id.startswith("E2TL")
                    or vehicle.lane_id.startswith("F2TL")
                ):
                    self.vehicle_metrics[v_id]["approach_speed"].append(vehicle.speed)
                elif vehicle.lane_id.startswith("TL"):
                    self.vehicle_metrics[v_id]["int_speed"].append(vehicle.speed)
                else:
                    self.vehicle_metrics[v_id]["leaving_speed"].append(vehicle.speed)

                self.vehicle_metrics[v_id]["accel"].append(vehicle.accel)
                self.vehicle_metrics[v_id]["fuel"].append(vehicle.fuel_consumption)

                if (
                    vehicle.lane_id.startswith("A2TL")
                    or vehicle.lane_id.startswith("B2TL")
                    or vehicle.lane_id.startswith("C2TL")
                    or vehicle.lane_id.startswith("D2TL")
                    or vehicle.lane_id.startswith("E2TL")
                    or vehicle.lane_id.startswith("F2TL")
                ):
                    self.vehicle_metrics[v_id]["int_crossed"] = [0]
                else:
                    self.vehicle_metrics[v_id]["int_crossed"] = [1]

                if vehicle.has_changed_lane:
                    self.vehicle_metrics[v_id]["lane_change"].append(1)
                if vehicle.closest_optim_lane_distance is not None:
                    self.vehicle_metrics[v_id]["removed"] = [0]
                for condition, emission in vehicle.co2_emission.items():
                    self.vehicle_metrics[v_id][f"emission_{condition}"].append(emission)
                    if (
                        vehicle.lane_id.startswith("A2TL")
                        or vehicle.lane_id.startswith("B2TL")
                        or vehicle.lane_id.startswith("C2TL")
                        or vehicle.lane_id.startswith("D2TL")
                        or vehicle.lane_id.startswith("E2TL")
                        or vehicle.lane_id.startswith("F2TL")
                    ):
                        self.vehicle_metrics[v_id][
                            f"approach_emission_{condition}"
                        ].append(emission)
                        if vehicle.speed < 0.1:
                            self.vehicle_metrics[v_id][
                                f"approach_idling_emission_{condition}"
                            ].append(emission)
                        elif vehicle.accel > 0:
                            self.vehicle_metrics[v_id][
                                f"approach_accelaration_emission_{condition}"
                            ].append(emission)
                        elif vehicle.accel < 0:
                            self.vehicle_metrics[v_id][
                                f"approach_decelaration_emission_{condition}"
                            ].append(emission)
                        else:
                            self.vehicle_metrics[v_id][
                                f"approach_const_speed_emission_{condition}"
                            ].append(emission)
                    elif vehicle.lane_id.startswith("TL"):
                        self.vehicle_metrics[v_id][
                            f"leaving_emission_{condition}"
                        ].append(emission)
                        if vehicle.speed < 0.1:
                            self.vehicle_metrics[v_id][
                                f"leaving_idling_emission_{condition}"
                            ].append(emission)
                        elif vehicle.accel > 0:
                            self.vehicle_metrics[v_id][
                                f"leaving_accelaration_emission_{condition}"
                            ].append(emission)
                        elif vehicle.accel < 0:
                            self.vehicle_metrics[v_id][
                                f"leaving_decelaration_emission_{condition}"
                            ].append(emission)
                        else:
                            self.vehicle_metrics[v_id][
                                f"leaving_const_speed_emission_{condition}"
                            ].append(emission)
                    else:
                        self.vehicle_metrics[v_id][
                            f"intersection_emission_{condition}"
                        ].append(emission)
                        if vehicle.speed < 0.1:
                            self.vehicle_metrics[v_id][
                                f"intersection_idling_emission_{condition}"
                            ].append(emission)
                        elif vehicle.accel > 0:
                            self.vehicle_metrics[v_id][
                                f"intersection_accelaration_emission_{condition}"
                            ].append(emission)
                        elif vehicle.accel < 0:
                            self.vehicle_metrics[v_id][
                                f"intersection_decelaration_emission_{condition}"
                            ].append(emission)
                        else:
                            self.vehicle_metrics[v_id][
                                f"intersection_const_speed_emission_{condition}"
                            ].append(emission)

                if (
                    vehicle.lane_id.startswith("A2TL")
                    or vehicle.lane_id.startswith("B2TL")
                    or vehicle.lane_id.startswith("C2TL")
                    or vehicle.lane_id.startswith("D2TL")
                    or vehicle.lane_id.startswith("E2TL")
                    or vehicle.lane_id.startswith("F2TL")
                ):
                    if self.vehicle_metrics[v_id]["start_speed"] == []:
                        self.vehicle_metrics[v_id]["start_speed"].append(vehicle.speed)
                self.vehicle_metrics[v_id]["slope"].append(vehicle.slope)

            for v_id, vehicle in self.traffic_state.vehicles.items():
                if (
                    vehicle not in self.traffic_state.completed_vehicle
                    and v_id not in self.traffic_state.current_vehicles
                    and vehicle not in self.warmup_vehicles
                ):
                    for condition, emission in vehicle.co2_emission.items():
                        self.vehicle_metrics[v_id][
                            f"effective_uncontrolled_emission_{condition}"
                        ].append(emission)

            if self.config.report_uncontrolled_region_metrics:
                blocked_vehicles_number = len(
                    self.traci.simulation.getPendingVehicles()
                )
                self.uncontrolled_region_metrics["waiting"].append(
                    blocked_vehicles_number
                )

                self.uncontrolled_region_metrics["fuel"].append(
                    blocked_vehicles_number * 0
                )

                for v_id, vehicle in self.traffic_state.vehicles.items():
                    if (
                        vehicle not in self.traffic_state.completed_vehicle
                        and v_id not in self.traffic_state.current_vehicles
                        and vehicle not in self.warmup_vehicles
                    ):
                        self.uncontrolled_region_metrics["waiting"][-1] += 1
                        self.uncontrolled_region_metrics["fuel"][
                            -1
                        ] += vehicle.fuel_consumption
                        for condition, emission in vehicle.co2_emission.items():
                            self.uncontrolled_region_metrics[
                                f"emission_{condition}"
                            ].append(emission)


def npa(x: float | int) -> np.ndarray:
    return np.array([x], np.float32)