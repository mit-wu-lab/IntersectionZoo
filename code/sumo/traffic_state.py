import logging
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

import numpy as np
import traci
from env.config import IntersectionZooEnvConfig
from env.task_context import TaskContext
from sumo.physical_models import FuelEmissionsModels
from sumo.sumo import TRACI_VARS, SubscriptionManager
from sumo.utils import (get_directions, get_edge_id, get_lane_counts,
                        get_splited_edges, is_main_lane)
from sumo.vehicle import Vehicle

# Do not use the root logger in Ray workers
logger = logging.getLogger("sumo")


class TrafficState:
    """
    Keeps relevant parts of SUMO simulation state in Python for easy access.

    Given that querying SUMO is slow, we keep in cache as much information as possible.
    """

    def __init__(
        self,
        config: IntersectionZooEnvConfig,
        traci_module: traci,
        task_context: TaskContext,
        net_file: Path,
    ):
        """
        Initialize and populate container objects
        """
        self.config = config
        self.traci_module = traci_module
        self.subscriptions: Dict[str, SubscriptionManager] = {}
        self.vehicles: Dict[str, Vehicle] = {}
        self.completed_vehicle: Set[Vehicle] = set()
        self.driver_variety = {}
        self.current_vehicles: Dict[str, Vehicle] = {}
        self.current_vehicles_sorted_lists: DefaultDict[
            str, DefaultDict[int, List[Vehicle]]
        ] = defaultdict(lambda: defaultdict(lambda: []))
        self.net_infos = defaultdict(lambda: {})
        self.junctions = get_directions(net_file)
        self.edges = sorted(
            [edge.format(j) for edge in ("{}2TL", "TL2{}") for j in self.junctions]
        )
        self.splited_edges = get_splited_edges(net_file)

        self.lane_counts = get_lane_counts(self.edges, net_file, True)
        self.task_context = task_context
        self._setup_subscriptions()
        self.fuel_emissions_models = FuelEmissionsModels(config)

    def step(self):
        """
        Take a simulation step and update state
        """
        # Actual SUMO step
        self.traci_module.simulationStep()
        sim_res = self.subscriptions["sim"].get()

        # subscribe to newly departed vehicles
        for v_id in sim_res["departed_vehicles_ids"]:
            self.subscriptions["veh"].subscribe(v_id)

        # update vehicle states
        veh_info = self.subscriptions["veh"].get_all()
        for v_id, vehicle in self.vehicles.items():
            if vehicle not in self.completed_vehicle:
                if v_id not in sim_res["arrived_vehicles_ids"]:
                    vehicle.update(
                        veh_info[(v_id, "lane_id")],
                        veh_info[(v_id, "laneposition")],
                        veh_info[(v_id, "speed")],
                        self.fuel_emissions_models.get_fuel(
                            vehicle,
                            veh_info[(v_id, "speed")],
                            veh_info[(v_id, "acceleration")],
                            veh_info[(v_id, "slope")],
                        ),
                        self.fuel_emissions_models.get_emissions(
                            vehicle,
                            veh_info[(v_id, "speed")],
                            veh_info[(v_id, "acceleration")],
                            veh_info[(v_id, "slope")],
                        ),
                        veh_info[(v_id, "acceleration")],
                        veh_info[(v_id, "signals")],
                        veh_info[(v_id, "slope")],
                    )
                else:
                    self.completed_vehicle.add(vehicle)

        for v_id in sim_res["departed_vehicles_ids"]:
            self.vehicles[v_id] = Vehicle(
                v_id,
                self.traci_module,
                self.junctions,
                self.lane_counts,
                self.task_context,
                self.config,
                self,
            )

        self.current_vehicles = {
            v_id: vehicle
            for v_id, vehicle in self.vehicles.items()
            if (
                vehicle not in self.completed_vehicle
                # some vehicles have no lane id but are already loaded
                and vehicle.lane_id != ""
                and is_main_lane(vehicle.lane_id)
            )
        }

        self.current_vehicles_sorted_lists = defaultdict(
            lambda: defaultdict(lambda: [])
        )
        for _, vehicle in self.current_vehicles.items():
            self.current_vehicles_sorted_lists[vehicle.platoon][
                vehicle.lane_index
            ].append(vehicle)

        for _, lane_indices in self.current_vehicles_sorted_lists.items():
            for lane_index, vehicle_list in lane_indices.items():
                lane_indices[lane_index] = sorted(
                    vehicle_list, key=lambda v: v.relative_distance, reverse=True
                )

        self.net_infos["phases"].clear()
        self.net_infos["phases_timing"].clear()

    def remove_vehicle(self, vehicle: Vehicle):
        """
        Removes a vehicle from the simulation and marks it as finished.
        """
        logger.info("REMOVING " + vehicle.id)
        self.completed_vehicle.add(vehicle)
        del self.current_vehicles[vehicle.id]
        self.current_vehicles_sorted_lists[vehicle.platoon][vehicle.lane_index].remove(
            vehicle
        )
        self.traci_module.vehicle.remove(vehicle.id)
        self.traci_module.vehicle.unsubscribe(vehicle.id)

    def get_phase(self, tl_id: str) -> int:
        """
        Returns the current phase index.
        """
        # using net_infos as cache to avoid unnecessary calls to traci
        if not ("phases" in self.net_infos and tl_id in self.net_infos["phases"]):
            self.net_infos["phases"][tl_id] = self.traci_module.trafficlight.getPhase(
                tl_id
            )

        return self.net_infos["phases"][tl_id]

    def remaining_phase_time(self, tl_id: str) -> float:
        """
        Time before the phase becomes green or red (IGNORES YELLOW PHASES, considered to be red).
        """
        # using net_infos as cache to avoid unnecessary calls to traci
        if not (
            "phases_timing" in self.net_infos
            and tl_id in self.net_infos["phases_timing"]
        ):
            self.net_infos["phases_timing"][tl_id] = (
                self.traci_module.trafficlight.getNextSwitch(tl_id)
                - self.traci_module.simulation.getTime()
            )

        return self.net_infos["phases_timing"][tl_id]

    def get_linear_distance(self, v1: Vehicle, v2: Vehicle) -> float:
        """
        Returns the distance between 2 vehicles if it is defined.

        !!! Uses the relative distance, i.e. the distance is not the real one when vehicles are in an internal lane or
        there is one between them. !!!
        """
        assert v1.platoon == v2.platoon

        if v1.relative_distance > v2.relative_distance:
            return v1.relative_distance - v2.relative_distance - v1.length
        else:
            return v2.relative_distance - v1.relative_distance - v2.length

    def get_leader(self, vehicle: Vehicle, side_lane: int = 0) -> Optional[Vehicle]:
        """
        Returns the id of the leading vehicle on the same edge. By default, searches in the current and next lane.
        If side_lane is 1 or -1, returns the leader in the adjacent (respectively left or right) lane,
        raises an error if no such lane exist.
        """
        current_lane_index = vehicle.lane_index

        if side_lane == 0:
            veh_in_lane = self.current_vehicles_sorted_lists[vehicle.platoon][
                current_lane_index
            ]
            try:
                index = veh_in_lane.index(vehicle)
                return veh_in_lane[index - 1] if index > 0 else None
            except ValueError:
                return None
        else:
            veh_in_lane = self.current_vehicles_sorted_lists[vehicle.platoon][
                current_lane_index + side_lane
            ]
            best = None
            for candidate in veh_in_lane:
                if candidate.relative_distance < vehicle.relative_distance:
                    return best
                else:
                    best = candidate
            return best

    def get_follower(self, vehicle: Vehicle, side_lane: int = 0) -> Optional[Vehicle]:
        """
        Returns the vehicle's follower
        """
        current_lane_index = vehicle.lane_index

        if side_lane == 0:
            veh_in_lane = self.current_vehicles_sorted_lists[vehicle.platoon][
                current_lane_index
            ]
            index = veh_in_lane.index(vehicle) + 1 if vehicle in veh_in_lane else 0
            return veh_in_lane[index] if len(veh_in_lane) > index else None
        else:
            veh_in_lane = self.current_vehicles_sorted_lists[vehicle.platoon][
                current_lane_index + side_lane
            ]
            best = None
            for candidate in reversed(veh_in_lane):
                if candidate.relative_distance >= vehicle.relative_distance:
                    return best
                else:
                    best = candidate
            return best

    def set_color(self, vehicle: Vehicle, color: Tuple[int, int, int]) -> None:
        # avoid unnecessary calls to Traci
        if self.config.visualize_sumo:
            self.traci_module.vehicle.setColor(vehicle.id, color + (255,))

    def accel(self, vehicle: Vehicle, acc: float, use_speed_factor: bool) -> bool:
        """
        Accelerates the vehicle by the given value if authorized on the current lane.

        Warning: only checks against the speed limit of the lane (which is not enforced when calling traci).
        The vehicle might also be limited by its internal max speed or its acceleration limits, which are enforced by
        SUMO when using traci.

        Returns whether the acceleration was fully applied (complies with speed limit).
        """
        desired_speed = vehicle.speed + acc * self.config.sim_step_duration
        allowed_speed = self.get_speed_limit(vehicle.lane_id) * (
            vehicle.speed_factor if use_speed_factor else 1
        )
        authorized_target_speed = max(0, min(desired_speed, allowed_speed))

        self.traci_module.vehicle.slowDown(vehicle.id, authorized_target_speed, 1e-3)

        return desired_speed == authorized_target_speed

    def get_speed_limit(self, lane_id: str) -> float:
        """
        Returns the speed limit of the lane in m/s
        """
        # using net_infos as cache to avoid unnecessary calls to traci
        if not (lane_id in self.net_infos and "max_speed" in self.net_infos[lane_id]):
            self.net_infos[lane_id]["max_speed"] = self.traci_module.lane.getMaxSpeed(
                lane_id
            )

        return self.net_infos[lane_id]["max_speed"]

    def get_lane_length(self, lane_id: str) -> float:
        """
        Returns the length of the lane in meters
        """
        edge_id = get_edge_id(lane_id)
        # using net_infos as cache to avoid unnecessary calls to traci
        if not (edge_id in self.net_infos and "length" in self.net_infos[edge_id]):
            if edge_id not in self.splited_edges:
                self.net_infos[edge_id]["length"] = self.traci_module.lane.getLength(
                    edge_id + "_0"
                )
            else:
                self.net_infos[edge_id]["length"] = self.traci_module.lane.getLength(
                    edge_id + "_0"
                ) + self.traci_module.lane.getLength(edge_id + "_intern_0")

        return self.net_infos[edge_id]["length"]

    def get_idm_accel(self, vehicle: Vehicle) -> float:
        """
        Returns the theoretical IDM acceleration, ignoring obstacles.
        For consistency, do NOT use, but rather do nothing, which will result in SUMO setting IDM acceleration itself.
        This makes sures that:
        - lane changing in SUMO works (you can't set the car following here and have SUMO doing the lc independently)
        - the constraints stipulated in the route files are homogeneous
        https://en.wikipedia.org/wiki/Intelligent_driver_model
        """
        v0 = self.get_speed_limit(vehicle.lane_id)
        # we use average idm param values here.
        t = 0.3  # minimum time to front veh
        a = 0.75
        b = 1.66
        delta = 4  # exponent, default value
        s0 = 2.22  # distance to front veh

        v = vehicle.speed

        # if entering the intersection we can use our getLeader which is faster
        if vehicle.edge_id.endswith("2TL"):
            leader = self.get_leader(vehicle)
            distance_to_leader = (
                self.get_linear_distance(leader, vehicle) + 0.1
                if leader is not None
                else 1e-10
            )
        else:
            leader = self.traci_module.vehicle.getLeader(vehicle.id)
            if leader is not None:
                distance_to_leader = leader[1] + 0.1
                leader = self.vehicles[leader[0]]
            else:
                distance_to_leader = 1e10

        if leader is None:  # no car ahead
            s_star = 0
        else:
            s_star = s0 + max(
                0, v * t + (v * (v - leader.speed) / (2 * np.sqrt(a * b)))
            )

        accel = a * (1 - (v / v0) ** delta - (s_star / distance_to_leader) ** 2)

        return accel

    def _setup_subscriptions(self) -> None:
        """
        Add subscriptions for some SUMO state variables
        """
        self.subscriptions["sim"] = SubscriptionManager(
            self.traci_module.simulation,
            [
                TRACI_VARS["departed_vehicles_ids"],
                TRACI_VARS["arrived_vehicles_ids"],
                TRACI_VARS["colliding_vehicles_ids"],
                TRACI_VARS["loaded_vehicles_ids"],
            ],
        ).subscribe()
        self.subscriptions["veh"] = SubscriptionManager(
            self.traci_module.vehicle,
            [
                TRACI_VARS["lane_id"],
                TRACI_VARS["laneposition"],
                TRACI_VARS["speed"],
                TRACI_VARS["acceleration"],
                # TRACI_VARS['fuelconsumption'],
                # TRACI_VARS['co2emission'],
                # TRACI_VARS['distance'],
                TRACI_VARS["signals"],
                TRACI_VARS["slope"],
            ],
        )
