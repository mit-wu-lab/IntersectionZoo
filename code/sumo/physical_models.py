from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from env.config import IntersectionZooEnvConfig
from sumo.constants import ELECTRIC, MOVES_ROAD_GRADE_RESOLUTION, REGULAR
from sumo.utils import is_rl
from sumo.vehicle import Vehicle
from torch import nn

"""
Mapping between MOVES fuel models ids (used in file names) and their name in our vehicle emission type IDs
"""


def mapping_moves(veh_emissions_type: str) -> str:
    veh_type, fuel_type, age = veh_emissions_type.split(".")
    year = 2019 - min(int(age), 10)  # only have fuel models for ages < 11,
    # thus we consider older vehicles to be 10 years old
    return f"{fuel_type}/{year}_{veh_type}"


def conditions_mapping(condition: str) -> Tuple[float, float]:
    temp, humidity = condition.split("_")
    return round(float(temp), 0), round(float(humidity), 0)


class FuelEmissionsModels:
    """
    Stateful container for fuel models that retrieves the necessary data at initialization and keeps it in memory.
    """

    def __init__(self, config: IntersectionZooEnvConfig):
        self.config = config

        self.moves_emissions_models: Dict[str, SurrogateNN] = {}
        """ mapping from vehicle emission type to the corresponding neural net """
        self.moves_emissions_models_cache: Dict[
            Tuple[str, float, float, float, float, float], float
        ] = {}
        """ cache for above decision trees,
            indexed by (in order) vehicle emissions type, speed, accel, road grade, temp, humidity"""
        minimums_csv = pd.read_csv(Path("resources/fuel_models/idling_consumption.csv"))
        # dicts of fuel_type, year, v type, temp, humidity
        self.minimums = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))
        )
        temps = [int(int(model[:2]) / 2) * 2 for model in config.moves_emissions_models]
        humid = [
            int(int(model[-2:]) / 2) * 2 for model in config.moves_emissions_models
        ]

        for row in minimums_csv.query(
            "temperature == @temps & humidity == @humid"
        ).iterrows():
            data = row[1]
            self.minimums[data["fuel_type"]][data["year"]][data["type"]][
                data["temperature"]
            ][data["humidity"]] = data["idling_consumption"]

    def get_fuel(
        self, vehicle: Vehicle, speed: float, accel: float, road_grade: float
    ) -> float:
        return vtcpmf_fuel_model(speed, accel) * self.config.sim_step_duration

    def get_default_fuel(self) -> float:
        return vtcpmf_fuel_model(0, 0) * self.config.sim_step_duration

    def get_emissions(
        self, vehicle: Vehicle, speed: float, accel: float, road_grade: float
    ) -> Dict[str, float]:
        return {
            condition
            + "_"
            + self.config.moves_emissions_models_conditions[
                i
            ]: self.get_emissions_single_condition(
                vehicle.emissions_type,
                speed,
                accel,
                road_grade,
                condition,
                self.config.moves_emissions_models_conditions[i],
                is_rl(vehicle.id),
            )
            for i, condition in enumerate(self.config.moves_emissions_models)
        }

    def get_emissions_single_condition(
        self,
        vehicle_emissions_type: str,
        speed: float,
        accel: float,
        road_grade: float,
        condition: str,
        emission_condition: str,
        is_rl: bool,
    ) -> float:

        if (
            self.moves_emissions_models is not None and emission_condition == REGULAR
        ) or (
            self.moves_emissions_models is not None
            and emission_condition == ELECTRIC
            and not is_rl
        ):

            effective_grade = (
                round(road_grade / MOVES_ROAD_GRADE_RESOLUTION)
                * MOVES_ROAD_GRADE_RESOLUTION
            )
            moves_veh_type = mapping_moves(vehicle_emissions_type)
            temp, humidty = conditions_mapping(condition)

            query = (
                moves_veh_type,
                round(speed, 1),
                round(accel, 1),
                effective_grade,
                temp,
                humidty,
            )
            if query not in self.moves_emissions_models_cache:
                if moves_veh_type not in self.moves_emissions_models:
                    self.moves_emissions_models[moves_veh_type] = SurrogateNN()
                    self.moves_emissions_models[moves_veh_type].load_state_dict(
                        torch.load(
                            Path(f"resources/fuel_models/{moves_veh_type}.pt"),
                            map_location=torch.device("cpu"),
                        )
                    )

                self.moves_emissions_models_cache[query] = max(
                    float(
                        self.moves_emissions_models[moves_veh_type](
                            torch.Tensor([*query[1:]])
                        )
                    ),
                    self.minimums[moves_veh_type.split("/")[0]][
                        int(moves_veh_type[-7:-3])
                    ][int(moves_veh_type[-2:])][int(query[-2] / 2) * 2][
                        int(query[-1] / 2) * 2
                    ],
                )
            return (
                self.moves_emissions_models_cache[query] * self.config.sim_step_duration
            )
        else:
            return 0

    def get_default_emissions(self) -> Dict[str, float]:
        return {
            condition + "_" + self.config.moves_emissions_models_conditions[i]: 0
            for i, condition in enumerate(self.config.moves_emissions_models)
        }


def vtcpmf_fuel_model(v_speed, v_accel):
    """
    VT-CPFM Fuel Model
    """

    r_f = (
        1.23 * 0.6 * 0.98 * 3.28 * (v_speed**2)
        + 9.8066 * 3152 * (1.75 / 1000) * 0.033 * v_speed
        + 9.8066 * 3152 * (1.75 / 1000) * 0.033
        + 9.8066 * 3152 * 0
    )

    power = ((r_f + 1.04 * 3152 * v_accel) / (3600 * 0.92)) * v_speed

    if power >= 0:
        return 0.00078 + 0.000019556 * power + 0.000001 * (power**2)
    else:
        return 0.00078


class SurrogateNN(nn.Module):
    def __init__(self):
        super(SurrogateNN, self).__init__()
        # our neural net has one input and one output layer with tanh activations
        self.relu_activation = nn.Tanh()
        # 'speed','accel', 'grade', 'temperature', 'humidity'
        self.layer1 = nn.Linear(5, 32)
        self.layer2 = nn.Linear(32, 1)

    def forward(self, x):
        # given an input x, this function makes the prediction
        x = self.relu_activation(self.layer1(x))
        x = self.layer2(x)
        return x
