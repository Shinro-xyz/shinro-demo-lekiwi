"""Pick-and-place integration: phase_list schedule drives base + arm + gripper through RobotSim."""

import numpy as np
import pytest

from shinro.factories import ScenarioFactory

from .helpers.scenario_runner import run_phase_schedule

pytestmark = [pytest.mark.integration]

SCENARIO = "scenarios/pick_and_place.toml"


@pytest.fixture
def scenario(mujoco_available):
    """A freshly built pick-and-place scenario with the shared free-joint MJCF."""
    return ScenarioFactory(SCENARIO).build()


class TestPickAndPlace:
    """The phase_list schedule drives a full reach → grip → lift → drive → release sequence."""

    def test_schedule_runs_without_error(self, scenario):
        """The full multi-plant schedule executes with finite states throughout."""
        records = run_phase_schedule(scenario)
        assert len(records) == len(scenario.trajectory["arm"])
        assert np.all(np.isfinite([r.plant_state[0] for r in records]))

    def test_gripper_closes_and_reopens(self, scenario):
        """The jaw setpoint is applied: gripper opens, closes, then opens again."""
        schedule = scenario.trajectory
        jaw = schedule["jaw"]
        # phases: open(0) -> close(0.5) -> hold(0.5) -> close(0.5) -> open(0)
        assert jaw.min() >= 0.0
        assert jaw.max() > 0.3, "gripper never closed (jaw never exceeded 0.3)"
        assert jaw[0] == 0.0 and jaw[-1] == 0.0, "gripper should open at start and end"
        # The jaw joint must actually move in response to the setpoint — not
        # just the schedule values (the setpoint used to be dropped).
        from shinro.simulation.runner import iter_phase_schedule

        engine = scenario.sim.engine
        qpos = np.asarray([engine.get_joint_qpos("Jaw") for _ in iter_phase_schedule(scenario)])
        assert qpos.max() - qpos.min() > 0.05, f"jaw never moved (range {qpos.max() - qpos.min():.4f})"

    def test_drives_out_and_returns(self, scenario):
        """The base drives out to the drop site and comes back to the start."""
        records = run_phase_schedule(scenario)
        xs = np.asarray([r.plant_state[0] for r in records])
        assert xs.max() > 0.8, f"base never reached the drop site (max x={xs.max():.3f})"
        assert abs(xs[-1]) < 0.15, f"base did not return to the start (final x={xs[-1]:.3f})"

    def test_arm_reaches_down_and_up(self, scenario):
        """The arm pitch/elbow setpoints are exercised without NaN."""
        records = run_phase_schedule(scenario)
        arm_states = np.array([r.plant_state for r in records])
        assert np.all(np.isfinite(arm_states))

    def test_reset_returns_to_origin(self, scenario):
        """After the sequence, a reset restores the zero state."""
        run_phase_schedule(scenario)
        scenario.sim.reset()
        base_state = np.asarray(scenario.sim.get_plant("base").get_state(), dtype=np.float64)
        assert np.allclose(base_state, np.zeros(3), atol=1e-3)


class TestGraspableBlock:
    """The demo's block is picked in front, carried, and left at the drop site."""

    def test_block_is_carried_and_placed(self, mujoco_available):
        from pathlib import Path

        from shinro.factories import TrajectoryFactory
        from shinro.simulation import RobotSim
        from shinro.utils.config_resolver import resolve_config_path
        from shinro_demo_lekiwi import MESH_DIR, MJCF_PATH, inject_free_joint, load_model_assets
        from shinro_demo_lekiwi.scene import (
            GROUND_Z,
            block_joint_addresses,
            carry_block,
            inject_graspable_block,
            park_block,
        )

        pick = (0.20, -0.03, GROUND_Z)
        xml = inject_graspable_block(inject_free_joint(Path(MJCF_PATH).read_text()), pos=pick)
        sim = RobotSim(
            str(resolve_config_path("configs/robot_config.toml")),
            xml_string=xml,
            assets=load_model_assets(MESH_DIR),
        )
        sim.reset()
        qpos_adr, dof_adr = block_joint_addresses(sim.engine.model)
        sched = TrajectoryFactory("configs/trajectories/pick_and_place.toml").create()

        grasped = False
        drop = None
        carried_gap = []
        for i in range(len(sched["arm"])):
            jaw = float(np.asarray(sched["jaw"][i]).ravel()[0])
            sim.arm.step(sched["arm"][i])
            sim.base.step(sched["base"][i])
            sim.engine.set_joint_ctrl("Jaw", jaw)
            if not grasped and jaw > 0.4:
                grasped = True
            elif grasped and jaw < 0.1:
                grasped = False
                ee = np.asarray(sim.arm._get_ee_pos(), dtype=np.float64)
                drop = (float(ee[0]), float(ee[1]), GROUND_Z)
            sim.step()
            if grasped:
                carry_block(sim, qpos_adr, dof_adr)
                ee = np.asarray(sim.arm._get_ee_pos(), dtype=np.float64)
                gap = np.asarray(sim.engine.data.qpos[qpos_adr:qpos_adr + 3]) - ee
                carried_gap.append(float(np.linalg.norm(gap)))
            else:
                park_block(sim, qpos_adr, dof_adr, drop if drop is not None else pick)

        block = np.asarray(sim.engine.data.qpos[qpos_adr:qpos_adr + 3])
        base = np.asarray(sim.base.get_state(), dtype=np.float64)
        assert carried_gap, "block was never grasped"
        assert max(carried_gap) < 1e-6, "block was not rigidly carried by the gripper"
        assert block[0] > 0.8, f"block was not placed at the drop site (x={block[0]:.3f})"
        assert abs(base[0]) < 0.15, f"base did not return to the start (x={base[0]:.3f})"
