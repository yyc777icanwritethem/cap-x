#!/usr/bin/env python3
"""Audit reset(seed=...) determinism for the remaining Robosuite CaP-Bench tasks.

This script is intentionally initialization-only: no LLM, SAM, grasping, IK,
or other API servers are used.  It checks:
  1) same instance + same seed repeated twice -> identical physical state;
  2) fresh privileged vs visual instances + same seed -> paired physical state.

The state fingerprint contains simulator dynamic state plus physical body / geom /
site poses, while excluding cameras and rendered pixels.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from capx.envs.simulators.robosuite_cube_lift import FrankaRobosuiteCubeLiftLowLevel
from capx.envs.simulators.robosuite_cubes_restack import FrankaRobosuiteCubesRestackLowLevel
from capx.envs.simulators.robosuite_handover import RobosuiteHandoverEnv
from capx.envs.simulators.robosuite_nut_assembly import FrankaRobosuiteNutAssembly
from capx.envs.simulators.robosuite_spill_wipe import FrankaRobosuiteSpillWipeLowLevel
from capx.envs.simulators.robosuite_two_arm_lift import RobosuiteTwoArmLiftEnv


TASKS: dict[str, Callable[..., Any]] = {
    "cube_lift": FrankaRobosuiteCubeLiftLowLevel,
    "cube_restack": FrankaRobosuiteCubesRestackLowLevel,
    "nut_assembly": FrankaRobosuiteNutAssembly,
    "spill_wipe": FrankaRobosuiteSpillWipeLowLevel,
    "two_arm_lift": RobosuiteTwoArmLiftEnv,
    "two_arm_handover": RobosuiteHandoverEnv,
}


def _arr(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1).copy()


def physical_state(env: Any) -> dict[str, np.ndarray]:
    """Capture physical simulator state, excluding cameras / pixels."""
    sim = env.robosuite_env.sim
    data = sim.data
    model = sim.model

    out: dict[str, np.ndarray] = {
        "qpos": _arr(data.qpos),
        "qvel": _arr(data.qvel),
        "body_xpos": _arr(data.body_xpos),
        "body_xquat": _arr(data.body_xquat),
        "geom_xpos": _arr(data.geom_xpos),
        "geom_xmat": _arr(data.geom_xmat),
        "site_xpos": _arr(data.site_xpos),
        "site_xmat": _arr(data.site_xmat),
        # Some tasks randomize model-level placements / markers during reset.
        "model_body_pos": _arr(model.body_pos),
        "model_body_quat": _arr(model.body_quat),
        "model_geom_pos": _arr(model.geom_pos),
        "model_geom_quat": _arr(model.geom_quat),
        "model_site_pos": _arr(model.site_pos),
        "model_site_quat": _arr(model.site_quat),
    }
    return out


def digest(state: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        a = np.ascontiguousarray(state[key], dtype=np.float64)
        h.update(key.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def compare(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> tuple[bool, float, str | None]:
    if set(a) != set(b):
        return False, float("inf"), "key mismatch"
    max_diff = 0.0
    for key in sorted(a):
        if a[key].shape != b[key].shape:
            return False, float("inf"), f"shape mismatch at {key}: {a[key].shape} vs {b[key].shape}"
        if a[key].size:
            d = float(np.max(np.abs(a[key] - b[key])))
            max_diff = max(max_diff, d)
    return max_diff == 0.0, max_diff, None


def close_env(env: Any) -> None:
    try:
        env.robosuite_env.close()
    except Exception:
        pass
    del env
    gc.collect()


def make_env(factory: Callable[..., Any], privileged: bool) -> Any:
    # Match the benchmark's normal CodeExecEnvConfig default (enable_render=True).
    # This also keeps the two-arm lift wrapper on its supported single-camera path;
    # the physical-state fingerprint below still excludes cameras and pixels.
    return factory(privileged=privileged, enable_render=True, viser_debug=False)


def audit_task(name: str, factory: Callable[..., Any], seeds: list[int]) -> dict[str, Any]:
    print(f"\n=== {name} ===")
    result: dict[str, Any] = {"same_instance": [], "cross_mode": []}

    # Same-instance repeat determinism, separately for privileged and visual mode.
    for label, privileged in (("privileged", True), ("visual", False)):
        print(f"[{label}] same-instance repeat")
        env = make_env(factory, privileged)
        try:
            for seed in seeds:
                env.reset(seed=seed)
                s1 = physical_state(env)
                env.reset(seed=seed)
                s2 = physical_state(env)
                match, max_diff, reason = compare(s1, s2)
                row = {
                    "mode": label,
                    "seed": seed,
                    "match": match,
                    "hash1": digest(s1),
                    "hash2": digest(s2),
                    "max_diff": max_diff,
                    "reason": reason,
                }
                result["same_instance"].append(row)
                print(
                    f"  seed={seed}: repeat_match={match} "
                    f"{row['hash1']} -> {row['hash2']} max_diff={max_diff:.3e}"
                    + (f" reason={reason}" if reason else "")
                )
        finally:
            close_env(env)

    # Fresh privileged vs visual instance pairing.
    print("[fresh-instance] privileged vs visual")
    for seed in seeds:
        env_p = make_env(factory, True)
        try:
            env_p.reset(seed=seed)
            sp = physical_state(env_p)
        finally:
            close_env(env_p)

        env_v = make_env(factory, False)
        try:
            env_v.reset(seed=seed)
            sv = physical_state(env_v)
        finally:
            close_env(env_v)

        match, max_diff, reason = compare(sp, sv)
        row = {
            "seed": seed,
            "match": match,
            "privileged_hash": digest(sp),
            "visual_hash": digest(sv),
            "max_diff": max_diff,
            "reason": reason,
        }
        result["cross_mode"].append(row)
        print(
            f"  seed={seed}: paired={match} privileged={row['privileged_hash']} "
            f"visual={row['visual_hash']} max_diff={max_diff:.3e}"
            + (f" reason={reason}" if reason else "")
        )

    same_ok = all(x["match"] for x in result["same_instance"])
    cross_ok = all(x["match"] for x in result["cross_mode"])
    result["same_instance_deterministic"] = same_ok
    result["cross_mode_paired"] = cross_ok
    result["pass"] = same_ok and cross_ok
    print(f"RESULT {name}: same_instance={same_ok} cross_mode={cross_ok} PASS={result['pass']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument(
        "--output",
        default="outputs/seed_audit/remaining_robosuite_seed_audit.json",
    )
    args = parser.parse_args()

    print("=== Remaining Robosuite seed audit ===")
    print("No LLM or perception/control API servers are used.")
    print("tasks:", ", ".join(args.tasks))
    print("seeds:", args.seeds)

    results: dict[str, Any] = {}
    for name in args.tasks:
        try:
            results[name] = audit_task(name, TASKS[name], args.seeds)
        except Exception as exc:
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
            results[name] = {
                "pass": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    overall = all(x.get("pass", False) for x in results.values())
    payload = {"seeds": args.seeds, "tasks": results, "overall_pass": overall}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("\n=== SUMMARY ===")
    for name in args.tasks:
        r = results[name]
        if "error" in r:
            print(f"{name:18s} ERROR  {r['error']}")
        else:
            print(
                f"{name:18s} same={r['same_instance_deterministic']} "
                f"paired={r['cross_mode_paired']} PASS={r['pass']}"
            )
    print(f"OVERALL PASS: {overall}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
