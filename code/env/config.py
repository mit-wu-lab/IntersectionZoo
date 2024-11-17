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

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

from env.task_context import TaskContext
from sumo.constants import REGULAR


class IntersectionZooEnvConfig(NamedTuple):
    """ """

    working_dir: Path
    """ where to retrieve and store artifacts """

    task_context: TaskContext | None = None
    """ defines the environment, see the TaskContext class. Optional, can also directly be given to the env."""
    ### Outputs and monitoring
    logging: bool = True
    """ Verbose logging """
    visualize_sumo: bool = False
    """ whether to use SUMO's GUI"""
    moves_output: Optional[Path] = None
    """ If specified exports vehicle trajectories in the given dir to be processed by MOVES to get accurate emissions"""
    trajectories_output: bool = False
    """ Generates trajectories_emissions.xml file (made for single evals) for time-space diagrams """

    ### Simulation settings
    moves_emissions_models: List[str] = ["68_46"]
    """ Which (if any) MOVES surrogate to use """
    moves_emissions_models_conditions: List[str] = [REGULAR]
    """ What is the condition for each MOVES surrogate """
    """ Can be either REGULAR or ELECTRIC. If ELECTRIC, the emission model is only used for non-electric 
    vehicles and electric vehicles have zero emissions. Note that only controlled vehicles can be electric. 
    The human driven vehicles are always internal combustion engine vehicles. """
    report_uncontrolled_region_metrics: bool = False
    """ Reports metrics both for controlled and uncontrolled region """
    csv_output_custom_name: str = "results"
    """ Name of the csv file to store the metrics """
    control_lane_change: bool = False
    """ Whether to control controlled vehicles Lane change """
    sim_step_duration: float = 0.5
    """ Duration of SUMO steps """
    warmup_steps: int = 50  # in multi lane cannot do warmup
    """
    Number of Warmup steps at the beginning of the episode
    where vehicles are not controlled and metrics not collected
    """
    action_noise_sigma: float = 0
    """ Standard deviation of the action noise. The action noise is sample from a guassian and multipled by the intended acceleration."""

    ### Reward
    stop_penalty: Optional[float] = 35
    threshold: float = 1
    accel_penalty: Optional[float] = None
    emission_penalty: Optional[float] = 3
    lane_change_penalty: Optional[float] = None
    optim_lane_penalty: Optional[float] = None
    fleet_reward_ratio: float = 1
    fleet_stop_penalty: Optional[float] = 0

    simulation_duration: int = 1000
    """ Duration of the simualtion in sec, including warmup """

    def update(self, config_fields: Dict[str, any]) -> "IntersectionZooEnvConfig":
        """
        Returns a NEW config object with the given fields
        """
        return IntersectionZooEnvConfig(**{**self._asdict(), **config_fields})
