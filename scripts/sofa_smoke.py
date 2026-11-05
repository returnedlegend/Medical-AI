#!/usr/bin/env python3
"""Optional upstream LapGym physics smoke; never produces CVS-study outcomes.

Run with scripts/run-sofa-smoke.sh after scripts/setup-sofa.sh. One SOFA scene
per subprocess is required by upstream, so this is intentionally a standalone
worker rather than an import inside the threaded analytic API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/workspace/.sofa-runtime/smoke.json"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--start-grasped", action="store_true", help="Use upstream's bounded scripted grasp during reset")
    parser.add_argument("--topology-mode", choices=("upstream", "derived"), default="derived")
    for name in ("pan", "tilt", "spin", "insertion", "jaw"):
        parser.add_argument(f"--{name}", type=float, default=0, help="Normalized gripper velocity in [-0.5,0.5]")
    args = parser.parse_args()
    if not 1 <= args.steps <= 100 or not 64 <= args.size <= 1024 or not 0 <= args.seed <= 2**32 - 1:
        parser.error("steps must be 1..100, size 64..1024, and seed 0..4294967295")
    commands = [getattr(args, key) for key in ("pan", "tilt", "spin", "insertion", "jaw")]
    if not all(-0.5 <= command <= 0.5 for command in commands):
        parser.error("All normalized gripper commands must be finite and in [-0.5,0.5]")

    started = time.monotonic()
    import Sofa
    import SofaRuntime  # noqa: F401 — initialize the upstream runtime
    import numpy as np
    from PIL import Image
    diagnostics = {"error_count": 0, "warning_count": 0, "topology_error_count": 0, "messages": []}

    class DiagnosticHandler(Sofa.Helper.MessageHandler):
        def process(self, message):
            kind = message["type"]
            if kind in ("Error", "Fatal"):
                diagnostics["error_count"] += 1
                if "topolog" in (message["sender"] + message["message"]).lower():
                    diagnostics["topology_error_count"] += 1
            elif kind == "Warning":
                diagnostics["warning_count"] += 1
            if kind in ("Error", "Fatal", "Warning") and len(diagnostics["messages"]) < 30:
                diagnostics["messages"].append({key: str(message[key]) for key in ("type", "sender", "message")})

    handler = DiagnosticHandler()
    handler.__enter__()
    from sofa_env.scenes.grasp_lift_touch.grasp_lift_touch_env import GraspLiftTouchEnv, ObservationType, RenderMode, Phase

    scene_options = {"scene_path": Path(__file__).with_name("sofa_scene.py")} if args.topology_mode == "derived" else {}
    env = GraspLiftTouchEnv(
        **scene_options,
        observation_type=ObservationType.RGB,
        render_mode=RenderMode.HEADLESS,
        image_shape=(args.size, args.size),
        time_step=0.01,
        settle_steps=10,
        randomize_point_of_interest=False,
        limit_grasping_attempts_in_reset=True,
        start_in_phase=Phase.LIFT if args.start_grasped else Phase.GRASP,
    )
    try:
        observation, _ = env.reset(seed=args.seed)
        assert env.observation_space.contains(observation), "Reset observation is outside the declared space"
        assert np.isfinite(observation).all(), "Reset observation is nonfinite"
        initial_vertices = env.gallbladder.mechanical_object.position.array().copy()
        topology = {}
        for name in ("liver", "gallbladder"):
            tissue = getattr(env, name)
            container = tissue.topology_container
            topology[name] = {
                "component_state": str(container.componentState.value),
                "vertex_count": len(tissue.mechanical_object.position.array()),
                "tetrahedra": container.tetrahedra.array().tolist(),
                "triangles": container.triangles.array().tolist(),
                "edges": container.edges.array().tolist(),
            }
            if args.topology_mode == "derived":
                assert topology[name]["component_state"] != "Invalid", f"{name} topology is Invalid"
        initial_image = np.asarray(env.render()).copy()
        initial_gripper_pose = env.gripper.get_pose().tolist()
        steps = []
        for index in range(args.steps):
            # The first five commands control only the gripper. The cauter
            # remains neutral. Commands are upstream normalized velocities.
            action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            action[:5] = commands
            observation, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(observation).all(), "Step observation is nonfinite"
            force = float(env.gallbladder.get_internal_force_magnitude())
            assert np.isfinite(force), "Force diagnostic is nonfinite"
            positions = env.gallbladder.mechanical_object.position.array()
            assert np.isfinite(positions).all(), "Mesh positions are nonfinite"
            displacement = np.linalg.norm(positions - initial_vertices, axis=1)
            steps.append({
                "step": index + 1, "reward": float(reward), "terminated": bool(terminated),
                "truncated": bool(truncated), "target_visible": bool(info["target_visible"]),
                "grasp_established": bool(env.gripper.grasp_established),
                "internal_force_sum_raw": force,
                "upstream_force_reward_feature": float(info["dynamic_force_on_gallbladder"]),
                "simulation_time_s": (index + 1) * env.time_step * env.frame_skip,
                "mean_vertex_displacement_scene_units": float(displacement.mean()),
                "max_vertex_displacement_scene_units": float(displacement.max()),
                "gripper_pose": env.gripper.get_pose().tolist(),
                "gripper_ptsda_state": env.gripper.get_ptsda_state().tolist(),
            })
            if terminated or truncated:
                break
        image = np.asarray(env.render())
        assert image.shape == (args.size, args.size, 3) and image.dtype == np.uint8
        assert int(image.max()) > int(image.min()), "Renderer returned a blank image"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image_path = args.output.with_suffix(".png")
        Image.fromarray(image).save(image_path)
        initial_image_path = args.output.with_name(args.output.stem + "-initial.png")
        Image.fromarray(initial_image).save(initial_image_path)
        commit_file = Path(os.environ["SOFA_ENV_SOURCE"]) / ".upstream_commit"
        mesh_directory = Path(os.environ["SOFA_ENV_SOURCE"]) / "sofa_env/scenes/grasp_lift_touch/meshes"
        def file_hash(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        provenance = {
            "worker_sha256": file_hash(Path(__file__)),
            "wrapper_sha256": file_hash(Path(__file__).with_name("sofa_scene.py")) if args.topology_mode == "derived" else None,
            "python_lock_sha256": file_hash(Path(__file__).with_name("requirements-sofa.lock")),
            "source_asset_sha256": {path.name: file_hash(path) for path in sorted(mesh_directory.iterdir()) if path.suffix in (".vtk", ".stl")},
            "sofa_archive_sha256_expected": "9d515e2f25f657c744821be8a5361e22803c18947b33af7a0b357c259202236a",
        }
        result = {
            "status": "passed", "purpose": "upstream-physics-smoke-only",
            "diagnostic_state": "needs_review" if diagnostics["error_count"] else "infrastructure_checked",
            "diagnostics": diagnostics,
            "topology_mode": args.topology_mode,
            "topology": topology,
            "provenance": provenance,
            "sofa_version": "24.06.00", "sofa_env_commit": commit_file.read_text().strip(),
            "python_version": sys.version.split()[0], "seed": args.seed,
            "observation_shape": list(observation.shape), "image_shape": list(image.shape),
            "action_shape": list(env.action_space.shape),
            "gallbladder_vertex_count": len(env.gallbladder.mechanical_object.position.array()),
            "steps": steps, "image_path": str(image_path.resolve()),
            "initial_image_path": str(initial_image_path.resolve()),
            "requested_steps": args.steps,
            "time_step_s": env.time_step,
            "commands": dict(zip(("pan", "tilt", "spin", "insertion", "jaw"), commands)),
            "start_grasped": args.start_grasped,
            "initial_gripper_pose": initial_gripper_pose,
            "initial_vertices": initial_vertices.tolist(),
            "final_vertices": env.gallbladder.mechanical_object.position.array().tolist(),
            "elapsed_seconds": time.monotonic() - started,
            "limitations": [
                "Upstream geometry, optionally with derived tetrahedral faces/edges; no patient parameterization, force-limit enforcement, or strategy sweep.",
                "Visibility refers to an upstream target sphere, not the hepatocystic triangle or critical view of safety.",
                "Raw internal-force sum is not a calibrated tool force in newtons; upstream reward scales it by 1e-9.",
                "A scripted upstream grasp and bounded actuator motion do not demonstrate clinical exposure or enforce tissue force limits.",
                "Inspect diagnostics: finite state and rendered frames do not validate mesh topology or physical fidelity.",
            ],
        }
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": "passed", "output": str(args.output), "elapsed_seconds": result["elapsed_seconds"]}))
    finally:
        env.close()
        handler.__exit__(None, None, None)


if __name__ == "__main__":
    main()
