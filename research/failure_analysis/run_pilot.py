"""Staged runner for the adaptive CaP-Bench failure-analysis pilot.

The runner is dry-run by default. Add --execute to launch trials.

Examples:
  python research/failure_analysis/run_pilot.py --phase cube-stack-smoke
  python research/failure_analysis/run_pilot.py --phase cube-stack-smoke --execute
  python research/failure_analysis/run_pilot.py --phase remaining-seed1 --execute

The initial pilot is intentionally limited to seed 1. CaP-X maps trial id 1 to
seed 1, so every command below uses --total-trials 1.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from capx.llm.client import VLM_MODELS

MODEL_DEFAULT = "openrouter/qwen/qwen3.8-27b"
VDM_MODEL_DEFAULT = "google/gemini-3.1-pro-preview"
SERVER_DEFAULT = "http://127.0.0.1:8110/chat/completions"

TIERS = ("S2", "S3", "M1", "M2", "M3", "M4")

MATRIX: dict[str, dict[str, str]] = {
    "cube_stack": {
        "S2": "env_configs/cube_stack/franka_robosuite_cube_stack.yaml",
        "S3": "env_configs/cube_stack/franka_robosuite_cube_stack_reduced_api.yaml",
        "M1": "env_configs/cube_stack/franka_robosuite_cube_stack_multiturn.yaml",
        "M2": "env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vf.yaml",
        "M3": "env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml",
        "M4": "env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm_reduced_api.yaml",
    },
    "nut_assembly": {
        "S2": "env_configs/nut_assembly/franka_robosuite_nut_assembly.yaml",
        "S3": "env_configs/nut_assembly/franka_robosuite_nut_assembly_reduced_api.yaml",
        "M1": "env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn.yaml",
        "M2": "env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn_vf.yaml",
        "M3": "env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn_vdm.yaml",
        "M4": "env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn_vdm_reduced_api.yaml",
    },
    "spill_wipe": {
        "S2": "env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml",
        "S3": "env_configs/spill_wipe/franka_robosuite_spill_wipe_reduced_api.yaml",
        "M1": "env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn.yaml",
        "M2": "env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn_vf.yaml",
        "M3": "env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn_vdm.yaml",
        "M4": "env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn_vdm_reduced_api.yaml",
    },
    "two_arm_handover": {
        "S2": "env_configs/two_arm_handover/two_arm_handover.yaml",
        "S3": "env_configs/two_arm_handover/two_arm_handover_reduced.yaml",
        "M1": "env_configs/two_arm_handover/two_arm_handover_multiturn.yaml",
        "M2": "env_configs/two_arm_handover/two_arm_handover_multiturn_vf.yaml",
        "M3": "env_configs/two_arm_handover/two_arm_handover_multiturn_vdm.yaml",
        "M4": "env_configs/two_arm_handover/hillclimb/two_arm_handover_multiturn_vdm_reduced_api.yaml",
    },
}


def selected_conditions(phase: str) -> list[tuple[str, str, str]]:
    if phase == "cube-stack-smoke":
        tasks = ["cube_stack"]
    elif phase == "remaining-seed1":
        tasks = ["nut_assembly", "spill_wipe", "two_arm_handover"]
    elif phase == "all-seed1":
        tasks = list(MATRIX)
    else:
        raise ValueError(phase)
    return [(task, tier, MATRIX[task][tier]) for task in tasks for tier in TIERS]


def make_command(
    task: str,
    tier: str,
    config_path: str,
    *,
    model: str,
    vdm_model: str,
    server_url: str,
    vdm_server_url: str,
    max_tokens: int,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "--active",
        "capx/envs/launch.py",
        "--config-path",
        config_path,
        "--model",
        model,
        "--server-url",
        server_url,
        "--max-tokens",
        str(max_tokens),
        "--total-trials",
        "1",
        "--num-workers",
        "1",
        "--output-dir",
        f"./outputs/failure_analysis_pilot/{task}/{tier}",
    ]
    if tier in {"M3", "M4"}:
        cmd += [
            "--visual-differencing-model",
            vdm_model,
            "--visual-differencing-model-server-url",
            vdm_server_url,
        ]
    return cmd


def preflight_errors(conditions: list[tuple[str, str, str]], model: str, vdm_model: str) -> list[str]:
    errors: list[str] = []
    if any(tier == "M2" for _, tier, _ in conditions) and model not in VLM_MODELS:
        errors.append(
            f"M2 BLOCKED: coding model {model!r} is not in VLM_MODELS. "
            "Run audit_qwen_groq_vision.py first; only register it after image input is verified."
        )
    if any(tier in {"M3", "M4"} for _, tier, _ in conditions) and vdm_model not in VLM_MODELS:
        errors.append(
            f"M3/M4 BLOCKED: VDM model {vdm_model!r} is not in VLM_MODELS."
        )
    for _, _, config_path in conditions:
        if not Path(config_path).exists():
            errors.append(f"Missing config: {config_path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["cube-stack-smoke", "remaining-seed1", "all-seed1"],
        default="cube-stack-smoke",
    )
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--vdm-model", default=VDM_MODEL_DEFAULT)
    parser.add_argument("--server-url", default=SERVER_DEFAULT)
    parser.add_argument("--vdm-server-url", default=SERVER_DEFAULT)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    conditions = selected_conditions(args.phase)
    print("=== Adaptive CaP-Bench pilot ===")
    print(f"phase:      {args.phase}")
    print(f"model:      {args.model}")
    print(f"vdm model:  {args.vdm_model}")
    print("seed:       1")
    print(f"conditions: {len(conditions)}")
    print(f"mode:       {'EXECUTE' if args.execute else 'DRY RUN'}")
    print()

    commands: list[tuple[str, str, list[str]]] = []
    for task, tier, config_path in conditions:
        cmd = make_command(
            task,
            tier,
            config_path,
            model=args.model,
            vdm_model=args.vdm_model,
            server_url=args.server_url,
            vdm_server_url=args.vdm_server_url,
            max_tokens=args.max_tokens,
        )
        commands.append((task, tier, cmd))
        print(f"[{task:18s} {tier}] {shlex.join(cmd)}")

    errors = preflight_errors(conditions, args.model, args.vdm_model)
    if errors:
        print("\n=== PREFLIGHT ===")
        for error in errors:
            print(f"ERROR: {error}")
        if args.execute:
            print("Refusing to execute until preflight errors are resolved.")
            raise SystemExit(2)
    else:
        print("\nPREFLIGHT: PASS")

    if not args.execute:
        print("\nDry run only; no model calls were made.")
        return

    manifest_dir = Path("outputs/failure_analysis_pilot")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"manifest_{args.phase}.json"
    records: list[dict] = []

    for i, (task, tier, cmd) in enumerate(commands, start=1):
        print(f"\n=== [{i}/{len(commands)}] {task} {tier} seed=1 ===")
        started = time.time()
        result = subprocess.run(cmd, check=False)
        record = {
            "task": task,
            "tier": tier,
            "seed": 1,
            "model": args.model,
            "vdm_model": args.vdm_model if tier in {"M3", "M4"} else None,
            "returncode": result.returncode,
            "elapsed_seconds": time.time() - started,
            "command": cmd,
        }
        records.append(record)
        manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

        if result.returncode != 0:
            print(f"INFRASTRUCTURE/LAUNCH FAILURE: return code {result.returncode}")
            if not args.keep_going:
                print("Stopping so an infrastructure failure is not confused with a benchmark failure.")
                raise SystemExit(result.returncode)

    print(f"\nCompleted {len(records)} conditions.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
