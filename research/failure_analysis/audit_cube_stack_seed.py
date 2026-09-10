#!/usr/bin/env python3
"""Audit Cube Stack reset determinism and cross-tier physical initialization.

This script deliberately avoids LLM calls and perception/control API servers. It
instantiates only the low-level Robosuite Cube Stack environment and compares the
initial simulator state produced by S1--S4-style settings.

Checks for each seed:
1. Same tier + same environment instance + same seed, reset twice.
2. Fresh environment instances across S1/S2/S3/S4 with the same seed.

The benchmark runner currently calls ``env.reset(seed=trial)``. This audit tests
whether that seed actually controls the underlying Robosuite randomization.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Must be set before importing Robosuite / MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import numpy as np

from capx.envs.simulators.robosuite_cubes import FrankaRobosuiteCubesLowLevel


TIER_SPECS = {
    "S1": {"privileged": True},
    "S2": {"privileged": False},
    "S3": {"privileged": False},
    "S4": {"privileged": False},
}


def _arr(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def snapshot(env: FrankaRobosuiteCubesLowLevel, seed: int) -> dict[str, Any]:
    """Reset once and capture physical state only (no model/API calls)."""
    env.reset(seed=seed, options={"trial": seed})
    rs_obs = env.robosuite_env._get_observations(force_update=True)

    fields = {
        "cubeA_pos": _arr(rs_obs["cubeA_pos"]),
        "cubeA_quat": _arr(rs_obs["cubeA_quat"]),
        "cubeB_pos": _arr(rs_obs["cubeB_pos"]),
        "cubeB_quat": _arr(rs_obs["cubeB_quat"]),
        "robot0_joint_pos": _arr(rs_obs["robot0_joint_pos"]),
        "sim_qpos": _arr(env.robosuite_env.sim.data.qpos.copy()),
    }

    # Two hashes: object-only is the key pairing criterion; full also includes robot state.
    object_vec = np.concatenate(
        [fields["cubeA_pos"], fields["cubeA_quat"], fields["cubeB_pos"], fields["cubeB_quat"]]
    )
    full_vec = np.concatenate([object_vec, fields["robot0_joint_pos"], fields["sim_qpos"]])

    return {
        "seed": seed,
        "object_hash": hashlib.sha256(object_vec.tobytes()).hexdigest()[:16],
        "full_hash": hashlib.sha256(full_vec.tobytes()).hexdigest()[:16],
        "fields": {k: v.tolist() for k, v in fields.items()},
    }


def compare(a: dict[str, Any], b: dict[str, Any], atol: float = 1e-10) -> dict[str, Any]:
    out: dict[str, Any] = {}
    max_abs = 0.0
    all_match = True
    for key in a["fields"]:
        av = _arr(a["fields"][key])
        bv = _arr(b["fields"][key])
        if av.shape != bv.shape:
            out[key] = {"match": False, "shape_a": list(av.shape), "shape_b": list(bv.shape)}
            all_match = False
            continue
        diff = float(np.max(np.abs(av - bv))) if av.size else 0.0
        match = bool(np.allclose(av, bv, rtol=0.0, atol=atol))
        max_abs = max(max_abs, diff)
        all_match = all_match and match
        out[key] = {"match": match, "max_abs_diff": diff}
    return {"all_match": all_match, "max_abs_diff": max_abs, "fields": out}


def make_env(privileged: bool) -> FrankaRobosuiteCubesLowLevel:
    # enable_render=False keeps the privileged tier headless. The non-privileged
    # wrapper still creates the camera observations required by its benchmark setting.
    return FrankaRobosuiteCubesLowLevel(
        privileged=privileged,
        enable_render=False,
        viser_debug=False,
    )


def cleanup_env(env: FrankaRobosuiteCubesLowLevel) -> None:
    try:
        close = getattr(env.robosuite_env, "close", None)
        if callable(close):
            close()
    except Exception:
        pass
    del env
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/seed_audit/cube_stack_seed_audit.json"),
    )
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "task": "cube_stack",
        "criterion": "same physical initialization for the same seed",
        "seeds": args.seeds,
        "tiers": {},
        "cross_tier": {},
    }

    print("=== Cube Stack seed audit ===")
    print("No LLM or perception/control API servers are used.\n")

    # 1) Same-instance repeated-reset determinism for each tier style.
    for tier, spec in TIER_SPECS.items():
        result["tiers"][tier] = {}
        print(f"[{tier}] privileged={spec['privileged']}")
        for seed in args.seeds:
            env = make_env(spec["privileged"])
            first = snapshot(env, seed)
            second = snapshot(env, seed)
            cmp = compare(first, second, atol=args.atol)
            result["tiers"][tier][str(seed)] = {
                "first": first,
                "second": second,
                "same_instance_same_seed": cmp,
            }
            print(
                f"  seed={seed}: repeat_match={cmp['all_match']} "
                f"object {first['object_hash']} -> {second['object_hash']} "
                f"max_diff={cmp['max_abs_diff']:.3e}"
            )
            cleanup_env(env)
        print()

    # 2) Fresh-instance cross-tier comparison. S2 is the reference tier.
    print("=== Cross-tier fresh-instance comparison (reference=S2) ===")
    for seed in args.seeds:
        states: dict[str, Any] = {}
        for tier, spec in TIER_SPECS.items():
            env = make_env(spec["privileged"])
            states[tier] = snapshot(env, seed)
            cleanup_env(env)

        ref = states["S2"]
        per_seed: dict[str, Any] = {"states": states, "comparisons_to_S2": {}}
        print(f"seed={seed}")
        for tier in TIER_SPECS:
            cmp = compare(ref, states[tier], atol=args.atol)
            per_seed["comparisons_to_S2"][tier] = cmp
            print(
                f"  S2 vs {tier}: match={cmp['all_match']} "
                f"S2_obj={ref['object_hash']} {tier}_obj={states[tier]['object_hash']} "
                f"max_diff={cmp['max_abs_diff']:.3e}"
            )
        result["cross_tier"][str(seed)] = per_seed

    repeat_ok = all(
        result["tiers"][tier][str(seed)]["same_instance_same_seed"]["all_match"]
        for tier in TIER_SPECS
        for seed in args.seeds
    )
    cross_ok = all(
        result["cross_tier"][str(seed)]["comparisons_to_S2"][tier]["all_match"]
        for seed in args.seeds
        for tier in TIER_SPECS
    )
    result["summary"] = {
        "same_instance_same_seed_all_match": repeat_ok,
        "cross_tier_same_seed_all_match": cross_ok,
        "paired_initialization_pass": repeat_ok and cross_ok,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(f"same-instance deterministic: {repeat_ok}")
    print(f"cross-tier paired:            {cross_ok}")
    print(f"PAIRED INITIALIZATION PASS:   {repeat_ok and cross_ok}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
