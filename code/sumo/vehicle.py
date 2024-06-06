import logging
from typing import Dict, List, Optional

import traci
from env.config import IntersectionZooEnvConfig
from env.task_context import TaskContext
from sumo.utils import (get_edge_id, get_green_phase_timings,
                        get_lane_index_from_lane_id, get_turn_signal_direction,
                        get_vehicle_emissions_type, is_internal_lane, is_rl,
                        straight_connection)

# Do not use the root logger in Ray workers
logger = logging.getLogger("sumo")


class Vehicle:
    def __init__(
        self,
        _id: str,
        traci_module: traci,
        junctions: List[str],
        lane_counts: Dict[str, int],
        task_context: TaskContext,
        config: IntersectionZooEnvConfig,
        traffic_state,
    ):
        """Vehicle id, same as SUMO's one"""
        self.id = _id
        """ The vehicle's type, used for emissions """
        self.emissions_type = get_vehicle_emissions_type(self.id)
        """ Lane id, same as SUMO's one """
        self.lane_id = ""
        """ Lane index, Internal lanes sometimes don't have consistent lane indices, thus to simplify the logic we
        assume vehicles don't change lane in internal lanes, if they do, it will be updating when they enter the next
        non internal lane """
        self.lane_index = 0
        """ Route ID, assumes the format X_Y where X and Y are single letter junction ids """
        self.route_id = traci_module.vehicle.getRouteID(_id)
        """ SUMO's longitudinal lane position """
        self.lane_position = 0
        """ speed, same as SUMO's one """
        self.speed = 0
        """ accel, same as SUMO's one """
        self.accel = 0
        """ slope, in degree, same as SUMO's one"""
        self.slope = 0
        """ fuel consumed in the last step """  # TODO specify unit
        self.fuel_consumption = 0
        """ CO2 emitted in the last step, in g of C02, for each condition specified in the config """
        self.co2_emission = {}
        """ relative distance to the TL, negative when approaching the TL, 0 in the internal lanes, positive after """
        self.relative_distance = 0.0
        """ bit encoding of vehicle exterior signals, same as https://sumo.dlr.de/docs/TraCI/Vehicle_Signalling.html """
        self.signals = 0
        """ whether the vehicle as changed lane since the last step, see lane_index for precise behaviour """
        self.has_changed_lane = False

        # constants throughout the simulation
        """ whether the vehicle will turn right (-1) go straight (0) or turn left (1) """
        self.direction = self._get_direction(junctions)
        """ the indices of the lanes the vehicle should be in (to realise its turn) """
        self.optimal_lanes = self._get_optimal_lanes(lane_counts, junctions)
        """ length, same as SUMO's one """
        self.length = traci_module.vehicle.getLength(self.id)
        """ speed factor, same as SUMO's one, considered constant """
        self.speed_factor = traci_module.vehicle.getSpeedFactor(self.id)

        v_type = traci_module.vehicle.getTypeID(self.id)

        self.tau = traci_module.vehicletype.getTau(v_type)
        self.min_gap = traci_module.vehicletype.getMinGap(v_type)
        self.max_accel = traci_module.vehicletype.getAccel(v_type)
        self.max_decel = traci_module.vehicletype.getDecel(v_type)

        self._straight_connection = straight_connection(junctions, self.origin)
        self._incoming_lane_length = traffic_state.get_lane_length(
            self.origin + "2TL_0"
        )
        self._traci_module = traci_module

        self.task_context = task_context
        self.config = config

        self.green_phase_index, self.green_phase_timings = get_green_phase_timings(
            self.origin, self.destination, traci_module
        )
        self.previous_step_idm_speed = 0

    def update(
        self,
        lane_id: str,
        lane_position,
        speed,
        fuel_consumption,
        co2emission,
        accel,
        signals,
        slope,
    ):
        # see above comment about lane index logic
        self.has_changed_lane = (
            self.lane_id != ""
            and lane_id != ""
            and not is_internal_lane(lane_id)
            and self.lane_index != get_lane_index_from_lane_id(lane_id)
        )
        if self.has_changed_lane:
            self.lane_index = get_lane_index_from_lane_id(lane_id)

        self.lane_id = lane_id
        self.lane_position = lane_position
        self.speed = speed
        self.accel = accel
        self.fuel_consumption = fuel_consumption
        self.co2_emission = co2emission
        self.slope = slope

        # We use the distance relative to the moment vehicles enter the intersection so that comparison between vehicles
        # with different routes (turning or not) still holds, and set the distance to 0 during the intersection
        if self.edge_id.endswith("2TL"):
            self.relative_distance = lane_position - self._incoming_lane_length
        elif is_internal_lane(self.lane_id):
            self.relative_distance = 0
        else:
            self.relative_distance = lane_position

        self.signals = signals

    @property
    def closest_optim_lane_distance(self) -> Optional[int]:
        """
        Distance (in number of lane change) to the closest optimal lane.
        """
        if self.edge_id.endswith("TL"):
            return min(
                (candidate - self.lane_index for candidate in self.optimal_lanes),
                key=lambda d: abs(d),
            )
        else:
            return None

    @property
    def is_rl(self) -> bool:
        """
        Whether the vehicle is RL controlled or not.
        """
        return is_rl(self.id)

    @property
    def edge_id(self) -> str:
        """
        ID of the current edge.
        """
        return get_edge_id(self.lane_id)

    @property
    def turn_signal(self) -> int:
        """
        Direction given by the blinker, 1 for left, -1 for right and 0 otherwise.
        """
        return get_turn_signal_direction(self.signals)

    @property
    def origin(self) -> str:
        """
        The vehicle's route origin junction
        """
        return self.route_id.split("_")[2][0]

    @property
    def destination(self) -> str:
        """
        The vehicle's route destination junction
        """
        return self.route_id.split("_")[3][0]

    @property
    def platoon(self) -> str:
        """
        Returns the platoon the vehicle is currently in.

        This concepts allows turning vehicles to change the group they belong to for leader computation
        """
        if (
            self.direction == 0
            or is_internal_lane(self.lane_id)
            or not self.edge_id.endswith("2TL")
        ):
            return self.destination
        else:
            return self._straight_connection

    def change_lane_relative(self, direction: int) -> None:
        """
        Changes lane to the right or the left
        """
        assert direction in [-1, 0, 1]
        self._traci_module.vehicle.changeLaneRelative(self.id, int(direction), 100000.0)

    def _get_direction(self, junctions: List[str]) -> int:
        """
        -1 for right, 0 for straight, and 1 for left turn
        """
        depart_index = junctions.index(self.origin)
        dest_index = junctions.index(self.destination)

        straight = junctions.index(straight_connection(junctions, self.origin))

        if dest_index == straight:
            return 0
        # left turn
        if depart_index < dest_index < straight or (
            (not straight < dest_index < depart_index) and straight < depart_index
        ):
            return 1
        # right turn
        else:
            return -1

    def _get_optimal_lanes(
        self, lane_counts: Dict[str, int], junctions: List[str]
    ) -> List[int]:
        """
        Returns the optimal lane(s) index(ces) the vehicle should be in for its turn.
        """
        depart_index = junctions.index(self.origin)

        if self.direction == 0:
            if lane_counts[junctions[depart_index] + "2TL"] == 1:
                return [0]

            return list(range(lane_counts[junctions[depart_index] + "2TL"] - 1))
        elif self.direction == 1:
            return [lane_counts[junctions[depart_index] + "2TL"] - 1]
        else:
            return [0]
