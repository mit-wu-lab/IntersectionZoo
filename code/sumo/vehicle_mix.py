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
from typing import Dict

import pandas as pd

RL_VEHICLE = {
    "speedFactor": "1",
    "accel": "2.6",
    "decel": "4.5",
    "tau": "1.0",
    "minGap": "2.0",
    "length": "5",
    "carFollowModel": "IDM",
    "color": "1,0,0",
    # 'lcKeepRight': "0",
    # 'lcStrategic': "0.1",
    # 'lcSpeedGain': '0',
    # 'lcLookaheadLeft': '0.1',
    # 'lcCooperative': '0.1',
}


class VehicleTypeParamsSampler:
    def __init__(self):
        self.precomputed_samples = {
            vehicle_type: pd.read_csv(
                Path("resources/idm_params") / f"{vehicle_type}.csv"
            )
            .rename(
                columns={
                    "mu_0": "velocity",
                    "mu_1": "minGap",
                    "mu_2": "tau",
                    "mu_3": "accel",
                    "mu_4": "decel",
                }
            )
            .drop(columns=["velocity"])
            for vehicle_type in ["car", "bus", "tru"]
        }

    def sample_idm_params(self, v_type_id: str) -> Dict[str, str]:
        """
        Samples the vehicle attributes for each vehicle based on the vehicle type.
        """
        return {
            "speedFactor": "1",
            "carFollowModel": "IDM",
            "length": self._veh_type_mix_mapping(v_type_id, "length"),
            "guiShape": self._veh_type_mix_mapping(v_type_id, "shape"),
            "color": "1,1,0",
            "lcStrategic": "1.0",
            "lcLookaheadLeft": "2.0",
            # 'accel': '1.0',
            # 'decel': '1.5',
            # 'tau': '1.0',
            # 'minGap': '2.0',
            **{
                k: str(v)
                for k, v in self.precomputed_samples[
                    self._veh_type_mix_mapping(v_type_id, "idm")
                ]
                .sample(1)
                .to_dict("records")[0]
                .items()
            },
        }

    def get_vehicle_mix(self) -> Dict[str, float]:
        data = pd.read_csv(Path("resources/vehicle_mix.csv"))
        return pd.Series(data["probability"].values, index=data["name"]).to_dict()

    def _veh_type_mix_mapping(self, v_type_id: str, usecase: str) -> str:
        """
        Maps from the type id in the vehicle mix file to the params used for other configs
        """
        # sumo defaults for lengths
        if usecase == "length":
            return {
                "21": "5",
                "31": "6",
                "32": "7.1",
                "42": "12",
            }[v_type_id[:2]]

        if usecase == "shape":
            return {
                "21": "passenger",
                "31": "passenger",
                "32": "truck",
                "42": "bus",
            }[v_type_id[:2]]

        if usecase == "idm":
            return {
                "21": "car",
                "31": "car",
                "32": "tru",
                "42": "bus",
            }[v_type_id[:2]]
