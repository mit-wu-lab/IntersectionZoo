import random
import re
import shutil
from pathlib import Path
from string import ascii_uppercase as ALPHABET
from typing import Dict, List, Tuple, Union
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

import numpy as np
import sumolib
from env.config import IntersectionZooEnvConfig
from env.task_context import NetGenTaskContext, PathTaskContext, TaskContext
from sumo.utils import get_directions, get_lane_counts, get_splited_edges
from sumo.vehicle_mix import RL_VEHICLE, VehicleTypeParamsSampler

SECONDS_PER_HOUR = 3600


def generate_temp_sumo_files(
    config: IntersectionZooEnvConfig, prefix: str, seed: int, task_context: TaskContext
) -> None:
    """
    Generates sumo net and routes according to the given config.
    """
    np.random.seed(seed)
    random.seed(seed)
    (config.working_dir / "sumo").mkdir(exist_ok=True)

    net_output_file = config.working_dir / f"sumo/net{prefix}.net.xml"

    if isinstance(task_context, NetGenTaskContext):
        _create_routes_from_netgentask(
            task_context,
            config,
            prefix
        )

        _modify_net(
            task_context.base_net_file().absolute(),
            task_context.base_id,
            task_context.lane_length,
            task_context.speed_limit,
            task_context.green_phase,
            task_context.red_phase,
            task_context.offset,
            net_output_file,
        )

    elif isinstance(task_context, PathTaskContext):
        _create_routes_from_pathtask(
            task_context,
            config,
            prefix
        )

        shutil.copy(str(task_context.base_net_file()), str(net_output_file))
    else:
        raise ValueError("Task context not recognized")


def _modify_net(
    file: Path,
    base_id: int,
    lane_length: float,
    speed_limit: float,
    green_phase: float,
    red_phase: float,
    offset: float,
    output_path: Path,
) -> None:
    net_raw = (
        open(file, "r")
        .read()
        .replace("13.89", str(speed_limit))
        .replace("<green_phase_duration>", str(green_phase))
        .replace("<red_phase_duration>", str(red_phase))
        .replace("<side_red_phase_duration>", str(red_phase + 6))
        .replace(
            "<offset>",
            str(
                offset
                * (6 + green_phase + red_phase + (30 if base_id // 10 == 4 else 0))
            ),
        )
    )

    text = re.split("\[.{6,7}\]", net_raw)
    numbers = [
        str(
            float(token[1:-1])
            + (1 if float(token[1:-1]) > 0 else -1) * (lane_length - 250)
        )
        for token in re.findall("\[.{6,7}\]", net_raw)
    ]

    net_raw = (
        "".join(token for tokens in zip(text, numbers) for token in tokens) + text[-1]
    )

    open(output_path, "w").write(net_raw)


def _create_routes_from_pathtask(
    task: PathTaskContext, config: IntersectionZooEnvConfig, prefix: str
) -> None:
    splited_edges = get_splited_edges(task.base_net_file())

    approaches = get_directions(task.base_net_file())
    num_lanes_with_leftonly = get_lane_counts(
        [approach + "2TL" for approach in approaches], task.base_net_file(), True
    )
    num_lanes_without_leftonly = get_lane_counts(
        [approach + "2TL" for approach in approaches], task.base_net_file(), False
    )
    approaches = [
        a
        for a in approaches
        if a + "2TL" in num_lanes_with_leftonly
        and num_lanes_with_leftonly[a + "2TL"] != 0
    ]
    inflows = [
        min(
            float(inflow) * task.aadt_conversion_factor, 900
        )  # We cap the inflows to 900 vehs per hour per lane
        for inflow in (task.base_net_file().parent / "inflows.txt")
        .open("r")
        .readlines()
    ]

    sumolib_net = sumolib.net.readNet(str(task.base_net_file()))
    connections = {
        approach: {
            lane: [
                connection.getTo().getToNode().getID()
                for connection in sumolib_net.getLane(
                    f'{approach}2TL{"_intern" if f"{approach}2TL" in splited_edges else ""}_{lane}'
                ).getOutgoing()
            ]
            for lane in range(num_lanes_with_leftonly[approach + "2TL"])
        }
        for approach in approaches
    }

    route_inflows = {
        f"{approach}{depart}_{destination}": (
            (
                f"{approach}{depart}2{approach} {approach}2TL TL2{destination}"
                if f"{approach}2TL" not in splited_edges
                else f"{approach}{depart}2{approach} {approach}2TL {approach}2TL_intern TL2{destination}"
            ),
            list(range(num_lanes_without_leftonly[approach + "2TL"])),
            inflow * destination_prop * depart_prop,
        )
        for approach, inflow in zip(approaches, inflows)
        for destination, destination_prop in get_turns(
            connections[approach], len(approaches), approach, 0.1, 0.1
        ).items()
        for depart, depart_prop in {"A": 0.1, "B": 0.9}.items()
        if (
            task.single_approach is False
            or (task.single_approach is True and approach == "A")
            or approach == task.single_approach
        )
    }

    gen_routes(
        task.penetration_rate,
        RL_VEHICLE,
        route_inflows,
        deterministic=False,
        config=config,
        prefix=prefix
    )


def _create_routes_from_netgentask(
    task: NetGenTaskContext, config: IntersectionZooEnvConfig, prefix: str
) -> None:
    """
    Changes the flows in a new copy of the route file.
    """
    if task.num_phases == 1:
        route_inflows = {
            name: details
            for origin in "ABCD"
            for name, details in {
                f"{origin}B_{_dest(origin, 0)}": (
                    f"{origin}B2{origin} {origin}2TL TL2{_dest(origin, 0)}",
                    list(range(task.num_lanes)),
                    task.inflow * 0.9,
                ),
                f"{origin}A_{_dest(origin, 0)}": (
                    f"{origin}A2{origin} {origin}2TL TL2{_dest(origin, 0)}",
                    list(range(task.num_lanes)),
                    task.inflow * 0.1,
                ),
            }.items()
        }
    else:
        route_inflows = {
            f"{origin}{depart}_{_dest(origin, destination)}": (
                f"{origin}{depart}2{origin} {origin}2TL TL2{_dest(origin, destination)}",
                list(range(task.num_lanes)),
                task.inflow
                * destination_prop
                * depart_prop
                * (2 if task.num_phases == 1 else 1),
            )
            for origin in "ABCD"
            for destination, destination_prop in {-1: 0.1, 0: 0.8, 1: 0.1}.items()
            for depart, depart_prop in {"A": 0.1, "B": 0.9}.items()
        }

    if task.single_approach is not False:
        route_inflows = {
            k: v
            for k, v in route_inflows.items()
            if (task.single_approach is True and k.startswith("A"))
            or (isinstance(task.single_approach, str) and k.startswith("A"))
        }

    gen_routes(
        task.penetration_rate,
        RL_VEHICLE,
        route_inflows,
        deterministic=False,
        config=config,
        prefix=prefix
    )


def gen_routes(
    penetration_rate: float,
    rl_veh_params: Dict[str, str],
    routes_inflows: Dict[str, Tuple[str, List[int], float]],
    config: IntersectionZooEnvConfig,
    prefix: str,
    deterministic: bool = False
) -> None:
    routes = Element("routes")
    trips: List[Element] = []

    params_sampler = VehicleTypeParamsSampler()
    vehicle_mix = params_sampler.get_vehicle_mix()

    assert (
        abs(
            sum(
                vehicle_type_proportion
                for _, vehicle_type_proportion in vehicle_mix.items()
            )
            - 1
        )
        < 0.001
    )

    routes.append(
        Element("vType", attrib={"id": "rl", "color": "1,0,0", **rl_veh_params})
    )
    offset = 0

    for route_id, (route_edges, lanes, inflow) in routes_inflows.items():
        for lane in lanes:
            time = (
                0
                if deterministic
                else np.random.exponential(1 / inflow) * SECONDS_PER_HOUR
            )

            i = 0
            while time < config.simulation_duration or i < 1:
                vehicle_type = random.choices(
                    list(vehicle_mix.keys()), list(vehicle_mix.values()), k=1
                )[0]

                if deterministic:
                    time_interval = (1 / inflow) * SECONDS_PER_HOUR
                    driver = (
                        "rl"
                        if (penetration_rate * i + offset) % 1 < penetration_rate
                        else "human"
                    )
                else:
                    # exponential intervals with 1/lambda expected val -> poisson process with rate of lambda
                    # inflow is per hour
                    time_interval = np.random.exponential(1 / inflow) * SECONDS_PER_HOUR
                    driver = random.choices(
                        ["rl", "human"], [penetration_rate, 1 - penetration_rate], k=1
                    )[0]

                # veh type name varies for RL but has no influence on veh type params !!!
                trip_id = f"{driver}_{vehicle_type}_{route_id}_{lane}_{i}"

                if driver == "human":
                    v_type_id = f"vType_{trip_id}"
                    routes.append(
                        Element(
                            "vType",
                            attrib={
                                "id": v_type_id,
                                "color": "1,1,0",
                                **params_sampler.sample_idm_params(vehicle_type),
                            },
                        )
                    )
                else:
                    v_type_id = "rl"

                route_elem = Element(
                    "route", attrib={"id": trip_id, "edges": route_edges}
                )
                vehicle = Element(
                    "vehicle",
                    attrib={
                        "id": trip_id,
                        "type": v_type_id,
                        "depart": str(time),
                        "departLane": str(lane),
                        # must be lower than the speed limit !
                        "departSpeed": "5",
                    },
                )
                vehicle.append(route_elem)
                trips.append(vehicle)

                time += time_interval
                i += 1

        offset += 1 / len(routes_inflows)

    trips.sort(key=lambda e: float(e.attrib["depart"]))

    # make sure that at least 1 rl vehicle spawns
    trips[0].attrib["depart"] = str(config.warmup_steps * config.sim_step_duration + 10)
    trips[0].attrib['id'] = trips[0].attrib['id'].replace('human', 'rl')
    trips.sort(key=lambda e: float(e.attrib["depart"]))

    routes.extend(trips)

    ElementTree.ElementTree(routes).write(config.working_dir / f"sumo/routes{prefix}.rou.xml",)


def get_turns(
    connections: Dict[int, List[str]],
    num_edges: int,
    incoming_edge: str,
    right_ratio: float,
    left_ratio: float,
):
    """
    Returns left, straight and right names and inflow ratio for the given edge index and connections
    @param connections: dict with format {lane origin: edges destination,} ex: {0: ['A'], 1:['A','B']}
    @param num_edges: number of edges in the intersection (including one way edges)
    @param incoming_edge: edge considered
    @param right_ratio: proportion of vehicles turning right
    @param left_ratio: proportion of vehicles turning left
    """
    # sorted from right to left
    destinations = list(
        {
            destination
            for destinations in connections.values()
            for destination in destinations
        }
    )
    destinations.sort(
        key=lambda x: (ALPHABET.index(x) - ALPHABET.index(incoming_edge)) % num_edges,
        reverse=True,
    )

    assert 0 < len(destinations)

    if len(destinations) == 1:
        return {destinations[0]: 1}
    if len(destinations) == 3:
        return {
            destinations[0]: right_ratio,
            destinations[1]: 1 - left_ratio - right_ratio,
            destinations[2]: left_ratio,
        }
    counts = {
        destination: [
            destination
            for destinations in connections.values()
            for destination in destinations
        ].count(destination)
        for destination in destinations
    }

    if len(destinations) == 2:
        straight = max(counts, key=lambda k: counts[k])
        other = min(counts, key=lambda k: counts[k])

        if counts[straight] == counts[other] or destinations.index(straight) == 0:
            return {destinations[0]: 1 - left_ratio, destinations[1]: left_ratio}
        else:
            return {destinations[0]: right_ratio, destinations[1]: 1 - right_ratio}
    else:
        # Too many, we just give traffic proportionally to the number of lanes
        return {d: counts[d] / sum(counts.values()) for d in destinations}


def _dest(origin, direction) -> str:
    return "ABCD"[("ABCD".index(origin) + 2 + direction) % 4]
