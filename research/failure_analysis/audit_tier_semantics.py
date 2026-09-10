#!/usr/bin/env python3
"""Static semantic audit for the 7x8 Robosuite CaP-Bench matrix.

No simulator, LLM, or API server is started. The script discovers the intended
S1-S4 / M1-M4 YAMLs, excludes skill-library / legacy / debug hillclimb variants,
and checks whether their declared API / feedback semantics match the benchmark
interpretation used in the failure-analysis study.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "env_configs"

TASK_DIRS = [
    "cube_lifting",
    "cube_stack",
    "cube_restack",
    "nut_assembly",
    "spill_wipe",
    "two_arm_lift",
    "two_arm_handover",
]
TIERS = ["S1", "S2", "S3", "S4", "M1", "M2", "M3", "M4"]

EXCLUDE_TOKENS = ("skill_lib", "legacy", "debug_", "ensemble_", "multimodel_")


def classify(path: Path, task_dir: Path) -> str | None:
    """Classify one YAML into the intended 8-condition matrix."""
    name = path.stem
    rel = path.relative_to(task_dir)

    if any(tok in name for tok in EXCLUDE_TOKENS):
        return None

    # S1-S4 and M1-M3 are expected in the task root. M4 may live in hillclimb
    # for two_arm_handover, so M4 is intentionally allowed recursively.
    is_root = len(rel.parts) == 1

    if name.endswith("_multiturn_vdm_reduced_api"):
        return "M4"
    if not is_root:
        return None
    if name.endswith("_privileged"):
        return "S1"
    if name.endswith("_reduced_api_exampleless") or name.endswith("_reduced_exampleless"):
        return "S4"
    if name.endswith("_reduced_api") or name.endswith("_reduced"):
        return "S3"
    if name.endswith("_multiturn_vdm"):
        return "M3"
    if name.endswith("_multiturn_vf"):
        return "M2"
    if name.endswith("_multiturn"):
        return "M1"

    # The only remaining root YAML should be the standard single-turn S2.
    if not any(tok in name for tok in ("privileged", "reduced", "multiturn")):
        return "S2"
    return None


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def info(path: Path) -> dict[str, Any]:
    data = load_yaml(path)
    env_cfg = data.get("env", {}).get("cfg", {}) or {}
    apis = env_cfg.get("apis", []) or []
    if isinstance(apis, str):
        apis = [apis]

    return {
        "path": str(path.relative_to(ROOT)),
        "target": data.get("env", {}).get("_target_"),
        "low_level": env_cfg.get("low_level"),
        "cfg_privileged": env_cfg.get("privileged", False),
        "apis": list(apis),
        "multi_turn_prompt": bool(env_cfg.get("multi_turn_prompt")),
        "use_visual_feedback": bool(data.get("use_visual_feedback", False)),
        "use_img_differencing": bool(data.get("use_img_differencing", False)),
        "use_video_differencing": bool(data.get("use_video_differencing", False)),
        "use_oracle_code": data.get("use_oracle_code"),
    }


def reduced_api(row: dict[str, Any]) -> bool:
    return any("reduced" in str(x).lower() for x in row["apis"])


def exampleless_api(row: dict[str, Any]) -> bool:
    return any("exampleless" in str(x).lower() for x in row["apis"])


def privileged_api(row: dict[str, Any]) -> bool:
    return any("privileged" in str(x).lower() for x in row["apis"])


def validate(task: str, tier: str, row: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    is_reduced = reduced_api(row)
    is_exampleless = exampleless_api(row)
    has_mt = row["multi_turn_prompt"]
    vf = row["use_visual_feedback"]
    vdm = row["use_img_differencing"]

    if tier.startswith("S") and has_mt:
        errors.append("single-turn tier unexpectedly has multi_turn_prompt")
    if tier.startswith("M") and not has_mt:
        errors.append("multi-turn tier missing multi_turn_prompt")

    if tier in {"S3", "S4", "M4"} and not is_reduced:
        errors.append("tier requires reduced/low-level API but API name is not reduced")
    if tier not in {"S3", "S4", "M4"} and is_reduced:
        errors.append("tier should use high-level API but exposes reduced API")

    if tier == "S4" and not is_exampleless:
        errors.append("S4 should use an exampleless reduced API")
    if tier != "S4" and is_exampleless:
        errors.append("only S4 should use the exampleless API in the main matrix")

    if tier == "M2":
        if not vf:
            errors.append("M2 missing use_visual_feedback=true")
        if vdm:
            errors.append("M2 unexpectedly enables image differencing")
    elif tier in {"M3", "M4"}:
        if not vdm:
            errors.append(f"{tier} missing use_img_differencing=true")
    else:
        if vf:
            errors.append(f"{tier} unexpectedly enables raw visual feedback")
        if vdm:
            errors.append(f"{tier} unexpectedly enables image differencing")

    if tier == "S1":
        if not (bool(row["cfg_privileged"]) or privileged_api(row)):
            errors.append("S1 is not declared privileged")
    elif privileged_api(row):
        errors.append("non-S1 tier exposes a privileged API")

    # NutAssembly's *_low_level_visual class forcibly sets privileged=False in
    # its constructor. Some YAMLs nevertheless carry cfg.privileged=true.
    # This does not change the actual visual simulator mode, but it is metadata
    # that should not be used to infer the tier.
    if task == "nut_assembly" and tier in {"S2", "S3", "S4", "M1", "M2", "M3", "M4"}:
        ll = str(row["low_level"] or "")
        if ll.endswith("_visual") and row["cfg_privileged"]:
            warnings.append(
                "cfg.privileged=true is misleading; low_level_visual forces privileged=False"
            )

    if "skill_lib" in row["path"]:
        errors.append("skill-library config leaked into main 8-tier matrix")

    return errors, warnings


def main() -> None:
    matrix: dict[str, dict[str, Any]] = {}
    fatal = False

    print("=== CaP-Bench Robosuite tier semantic audit ===")
    print("Expected: 7 tasks x 8 tiers = 56 configs; skill-lib variants excluded.\n")

    for task in TASK_DIRS:
        task_dir = CONFIG_ROOT / task
        candidates: dict[str, list[Path]] = {tier: [] for tier in TIERS}
        for path in sorted(task_dir.rglob("*.yaml")):
            tier = classify(path, task_dir)
            if tier is not None:
                candidates[tier].append(path)

        matrix[task] = {}
        print(f"=== {task} ===")
        for tier in TIERS:
            paths = candidates[tier]
            if len(paths) != 1:
                fatal = True
                print(f"{tier}: DISCOVERY_ERROR candidates={len(paths)}")
                for p in paths:
                    print(f"    {p.relative_to(ROOT)}")
                matrix[task][tier] = {
                    "discovery_error": f"expected 1 candidate, found {len(paths)}",
                    "candidates": [str(p.relative_to(ROOT)) for p in paths],
                }
                continue

            row = info(paths[0])
            errors, warnings = validate(task, tier, row)
            row["errors"] = errors
            row["warnings"] = warnings
            matrix[task][tier] = row
            if errors:
                fatal = True

            api_short = ",".join(row["apis"])
            flags = (
                f"priv={row['cfg_privileged']} "
                f"mt={row['multi_turn_prompt']} "
                f"vf={row['use_visual_feedback']} "
                f"vdm={row['use_img_differencing']}"
            )
            status = "FAIL" if errors else ("WARN" if warnings else "PASS")
            print(f"{tier}: {status:4s}  {row['path']}")
            print(f"    low_level={row['low_level']}  api={api_short}")
            print(f"    {flags}")
            for msg in errors:
                print(f"    ERROR: {msg}")
            for msg in warnings:
                print(f"    WARN:  {msg}")
        print()

    # Pairwise semantic checks that matter for causal comparisons.
    print("=== Pairwise comparison checks ===")
    for task in TASK_DIRS:
        rows = matrix[task]
        if any("discovery_error" in rows.get(t, {}) for t in TIERS):
            continue

        def apis(t: str) -> list[str]:
            return rows[t]["apis"]

        checks = {
            "S2/M1 same high-level API": apis("S2") == apis("M1"),
            "M1/M2 same high-level API": apis("M1") == apis("M2"),
            "M1/M3 same high-level API": apis("M1") == apis("M3"),
            "S3/M4 same reduced API": apis("S3") == apis("M4"),
        }
        print(task)
        for name, ok in checks.items():
            print(f"  {'PASS' if ok else 'WARN'} {name}")
        # API-name mismatches are worth reviewing, but not automatically fatal:
        # some tasks may use semantically equivalent task-specific wrappers.
        if not all(checks.values()):
            matrix[task]["pairwise_warning"] = [k for k, v in checks.items() if not v]

    out = ROOT / "outputs/tier_audit/robosuite_tier_semantics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")

    print("\n=== SUMMARY ===")
    print("matrix configs:", sum(1 for task in TASK_DIRS for tier in TIERS if "path" in matrix[task].get(tier, {})))
    print("semantic hard errors:", sum(len(matrix[task].get(tier, {}).get("errors", [])) for task in TASK_DIRS for tier in TIERS))
    print("warnings:", sum(len(matrix[task].get(tier, {}).get("warnings", [])) for task in TASK_DIRS for tier in TIERS))
    print("MAIN MATRIX SEMANTICS PASS:", not fatal)
    print("saved:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
