from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from xml.etree import ElementTree

import sumolib.net
import traci


def is_rl(vehicle_id: str) -> bool:
    """
    Returns whether the given vehicle ID corresponds to a RL controlled vehicle
    """
    return vehicle_id.startswith("rl")


def get_vehicle_emissions_type(vehicle_id: str) -> str:
    """
    Returns the vehicle type (car, car from year 2000, truck...) from vehicle ID
    """
    return vehicle_id.split("_")[1]


def get_lane_index_from_lane_id(lane_id: str) -> int:
    """
    Extracts the id of the lane from its id.
    Assumes that the id has the format "something_index".
    """
    return int(lane_id.split("_")[-1])


def is_main_junction(junction_id: str) -> bool:
    """
    Returns whether the given junction id is in the considered region, i.e. is a single letter.
    """
    return len(junction_id) == 1


def is_main_lane(lane_id: str) -> bool:
    """
    Returns whether the given lane id is in the considered region, i.e. goes from the TL to a main junction.
    """
    return any(
        junction_id == "TL" for junction_id in get_edge_id(lane_id).split("2")
    ) or (is_internal_lane(lane_id))


def get_turn_signal_direction(signals_bit_vector: int) -> int:
    """
    Converts the bit vector of teh signal attribute of a vehicle into the direction given by the blinker,
    1 for left, -1 for right and 0 otherwise.
    https://sumo.dlr.de/docs/TraCI/Vehicle_Signalling.html
    """
    bits = bin(signals_bit_vector)
    if bits[-1] == "1":
        return -1
    if bits[-2] == "1":
        return 1
    return 0


def get_edge_id(lane_id: str) -> str:
    """
    Extracts the id of the edge containing the given lane from the lane's id.
    Removes the '_intern' suffix from the actual SUMO edge id, which happens in the datasets

    Assumes that the lane id has the format "edgeId_laneIndex" and lane index < 10.
    If wrong one should use Traci (which is slower).
    """
    return lane_id[:-2].split("_intern")[0]


def is_internal_lane(lane_id: str) -> bool:
    """
    Returns whether the given lane id corresponds to an internal lane.
    """
    return lane_id.startswith(":")


def get_directions(net_file: Path) -> List[str]:
    """
    Returns the list of the directions in the net, excluding 'TL'.

    Assumes the A, B..., outsideA, outsideB..., TL naming scheme !!!
    """
    net = ElementTree.parse(net_file).getroot()

    return sorted(
        [
            j.attrib["id"]
            for j in net.findall("junction")
            if j.attrib["type"] != "internal" and len(j.attrib["id"]) == 1
        ]
    )


def straight_connection(directions: List[str], origin: str) -> str:
    """
    returns the direction corresponding to going straight in the given list of directions

    !!! only tested for an even number of edges, and assumes that the straight direction is in front !!!
    """
    # if len(directions) % 2 == 1:
    #     print('WARNING!!! The direction calculation might be wrong '
    #           'because the environment does not know what the straight direction is')

    depart_index = directions.index(origin)

    return directions[(depart_index + len(directions) // 2) % len(directions)]


def get_lane_counts(
    edges: Iterable[str], net_file: Path, include_left_turn_only_lane: bool
) -> Dict[str, int]:
    """
    Returns the number of lane for each given edge.
    """
    net = ElementTree.parse(net_file).getroot()
    lane_counts = {edge.attrib["id"]: len(edge) for edge in net.findall("edge")}

    return {
        e_id: count
        + (1 if include_left_turn_only_lane and e_id + "_intern" in lane_counts else 0)
        for e_id, count in lane_counts.items()
        if e_id in edges
    }


def get_green_phase_timings(
    origin, destination, traci_module: traci
) -> Tuple[int, Tuple[float, float, float]]:
    """
    Returns the phase index in the TL program that gives green to the vehicle with given origin and dest,
    along with the signal timing (duration in one program cycle before the given phase, during the phase, and after).
    """
    links = traci_module.trafficlight.getControlledLinks("TL")
    logic = traci_module.trafficlight.getAllProgramLogics("TL")[0].getPhases()

    link_index = -1
    for i, link in enumerate(links):
        if (
            len(link) > 0
            and get_edge_id(link[0][0]) == f"{origin}2TL"
            and get_edge_id(link[0][1]) == f"TL2{destination}"
        ):
            link_index = i
            break
    assert link_index > -1

    phase_index = -1
    for i, phase in enumerate(logic):
        if phase.state[link_index].lower() == "g":
            phase_index = i
            break
    assert phase_index > -1

    before = sum(phase.duration for i, phase in enumerate(logic) if i < phase_index)
    during = logic[phase_index].duration
    after = sum(phase.duration for i, phase in enumerate(logic) if i > phase_index)

    return phase_index, (before, during, after)


def get_remaining_time(timings: Tuple[float, float, float], simulation_time: float):
    """
    Given a vehicle phase timings, returns the remaining time, either in the current green phase,
    or until the light turns green.

    !!! Assumes that the TL is pretimed, has never been tempered during the simulation,
    and that the program offset is 0 !!!
    """
    relative_time = simulation_time % sum(timings)

    if relative_time < timings[0]:
        return timings[0] - relative_time
    elif relative_time <= timings[0] + timings[1]:
        return timings[0] + timings[1] - relative_time
    else:
        return sum(timings) - relative_time + timings[0]


def get_splited_edges(net_file: Path) -> List[str]:
    """
    Given a net file, returns the list of edges splited in 2 like A2TL and A2TL_intern
    """
    edges = [
        e.getID()
        for e in sumolib.net.readNet(str(net_file)).getEdges(withInternal=False)
    ]
    return [e for e in edges if e + "_intern" in edges]
