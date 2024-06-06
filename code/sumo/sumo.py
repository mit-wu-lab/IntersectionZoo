import inspect
import logging
import os
import sys
import uuid
from typing import Optional

import traci.constants as traci_constants  # https://sumo.dlr.de/pydoc/traci.constants.html
from env.config import IntersectionZooEnvConfig
from env.task_context import TaskContext
from sumo.constants import GUI_SETTINGS_FILE
from sumo.generate_files import generate_temp_sumo_files

TRACI_VARS = {
    k[4:].lower(): k
    for k, _ in inspect.getmembers(traci_constants, lambda x: not callable(x))
    if k.startswith("VAR_")
}

# Do not use the root logger in Ray workers
logger = logging.getLogger("sumo")

import traci

def start_sumo(
    config: IntersectionZooEnvConfig,
    traci_module: Optional[any],
    prefix: str,
    task_context: TaskContext,
    seed: int,
) -> traci:
    """
    Starts SUMO instance using provided configurations
    """
    # SUMO seems to have randomness issues when running in parallel (setting random in sumo args does not fully randomize the
    # behavior)
    generate_temp_sumo_files(config, prefix, seed, task_context)
    net_path = config.working_dir.absolute() / f"sumo/net{prefix}.net.xml"
    route_path = config.working_dir.absolute() / f"sumo/routes{prefix}.rou.xml"

    # https://sumo.dlr.de/docs/SUMO.html
    sumo_args = {
        "net-file": net_path,
        "route-files": route_path,
        "begin": 0,
        "step-length": config.sim_step_duration,
        "no-step-log": True,
        "time-to-teleport": -1,
        "no-warnings": True,
        "collision.action": "none",  # ignores collisions, more realistic than 'teleport', and 'remove' creates bug
        "collision.check-junctions": True,
        "start": True,
        "seed": seed,
    }

    cmd = ["sumo-gui" if config.visualize_sumo else "sumo"]
    if config.visualize_sumo:
        sumo_args["gui-settings-file"] = GUI_SETTINGS_FILE
    if config.trajectories_output:
        os.makedirs(config.working_dir.absolute() / f"traj_data", exist_ok=True)
        os.makedirs(config.working_dir.absolute() / f"traj_data/traj", exist_ok=True)
        os.makedirs(config.working_dir.absolute() / f"traj_data/fcd", exist_ok=True)
        sumo_args["emission-output"] = (
            config.working_dir.absolute()
            / f"traj_data/traj/{prefix}_trajectories_{task_context.compact_str()}.xml"
        )
        sumo_args["fcd-output"] = (
            config.working_dir.absolute()
            / f"traj_data/fcd/{prefix}_fcd_{task_context.compact_str()}.xml"
        )

    for k, v in sumo_args.items():
        cmd.extend(
            ["--%s" % k, (str(v).lower() if isinstance(v, bool) else str(v))]
            if v is not None
            else []
        )
    logger.info(f"starting SUMO with {cmd}")

    if traci_module is not None:
        traci_module.close()

    session_id = uuid.uuid4()
    traci.start(cmd, label=session_id)
    return traci.getConnection(session_id)


class SubscriptionManager:
    """
    Manages Traci subcriptions
    """

    def __init__(self, traci_module: traci, subs):
        self.traci_module = traci_module
        self.names = [k.split("_", 1)[1].lower() for k in subs]
        self.constants = [getattr(traci_constants, k) for k in subs]

    def subscribe(self, *_id):
        self.traci_module.subscribe(*_id, self.constants)
        return self

    def get(self, *_id):
        res = self.traci_module.getSubscriptionResults(*_id)
        return {n: res[v] for n, v in zip(self.names, self.constants)}

    def get_all(self):
        res = self.traci_module.getAllSubscriptionResults()

        return {
            (y, n): res[y][v]
            for y in res.keys()
            for n, v in zip(self.names, self.constants)
        }
