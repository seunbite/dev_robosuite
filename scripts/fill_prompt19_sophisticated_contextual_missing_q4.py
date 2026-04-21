from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"
TARGET_PATH = SEED / "motion_configs_prompt_v19_sophisticated_contextual.json"
PROMPT_PATH = SEED / "q4_contrastive_experiment" / "prompt_v19_sophisticated_q4_contrastive_full.txt"
SHOTS_PATH = SEED / "q4_contrastive_experiment" / "shot_configs_v19_sophisticated_q4_contrastive.json"
WORK_DIR = SEED / "q4_fill_sophisticated_contextual"
GENERATED_PATH = WORK_DIR / "motion_configs_prompt_v19_sophisticated_contextual_missing_q4_generated.json"
MANIFEST_PATH = WORK_DIR / "manifest.json"


def _load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(rows, key=lambda x: int(x["idx"])), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "adhoc" / "robotarm"))
    from config_gen_single import generate_motion_config  # noqa: E402

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    current_rows = _load_rows(TARGET_PATH)
    missing = [(int(r["idx"]), r["cue"]) for r in current_rows if "# Q4:" not in str(r.get("reasoning", ""))]

    for idx, cue in missing:
        generate_motion_config(
            cue_name=cue,
            cue_idx=idx,
            model_name="gemini-2.5-pro",
            prompt_file=str(PROMPT_PATH),
            shots_json=str(SHOTS_PATH),
            config_json=str(GENERATED_PATH),
            max_handmade_examples=10,
            max_correction_examples=10,
            temperature=None,
            use_shots=True,
            require_reasoning=True,
        )

    generated_rows = _load_rows(GENERATED_PATH)
    generated_map = {(int(r["idx"]), r["cue"]): r for r in generated_rows}
    merged = []
    replaced = []
    for row in current_rows:
        key = (int(row["idx"]), row["cue"])
        if key in generated_map:
            merged.append(generated_map[key])
            replaced.append(key)
        else:
            merged.append(row)

    backup_dir = SEED / "backups" / f"prompt19_sophisticated_contextual_q4_fill_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / TARGET_PATH.name
    shutil.copy2(TARGET_PATH, backup_path)
    _write_rows(TARGET_PATH, merged)

    manifest = {
        "target_path": str(TARGET_PATH),
        "backup_path": str(backup_path),
        "prompt_path": str(PROMPT_PATH),
        "shots_path": str(SHOTS_PATH),
        "generated_path": str(GENERATED_PATH),
        "missing_count": len(missing),
        "replaced_count": len(replaced),
        "missing": [{"idx": idx, "cue": cue} for idx, cue in missing],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Generated missing rows: {len(generated_rows)}")
    print(f"Replaced rows: {len(replaced)}")
    print(f"Updated target: {TARGET_PATH}")
    print(f"Backup: {backup_path}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
