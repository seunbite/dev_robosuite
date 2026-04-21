import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import fire

REPO_ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")

import sys

sys.path.append(str(REPO_ROOT / "adhoc" / "robotarm"))
from config_gen_single import generate_motion_config  # noqa: E402


def _load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {path}")
    return data


def _first_n_cues(config_json: str, limit: int) -> List[Dict]:
    rows = _load_json(config_json)
    return sorted(rows, key=lambda r: int(r.get("idx", -1)))[:limit]


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _make_one_shot_prompt(base_prompt: str) -> str:
    text = base_prompt
    text = text.replace(
        "[Required Planning Output]\nBefore writing JSON, you MUST answer the following questions as `#` comment lines.\nDo not skip any question.\n\n# Q1: How would a human express this iconic motion?\n# Q2: Which initial anchor pose directions are the strongest candidates, ranked by iconic readability?\n  - Compare multiple plausible initial directions first. Direction is the top priority.\n  - Q2 must rank at least 2 candidate directions for the first pose.\n  - For the strongest candidates, include `gripper_orientation`, and include `x`, `y`, `z` when they matter to readability.\n  - Explicitly state which candidate wins and why it wins over the others.\n# Q3: Which step structure best preserves the cue after that initial pose?\n  - Starting from the Q2 winning initial pose, compare multiple candidate motion structures.\n  - Try different primitive combinations such as `pose > movement`, `pose > path`, `pose > pose`, or other compact alternatives when appropriate.\n  - Specify the parameters that is important to manifest that gesture such as repetition, speed, hold time, angle size, pose placement, axis choice, gripper orientation.\n  - Explicitly choose the single winning structure and explain why it reads more clearly than the rejected alternatives.\n",
        "[Internal Planning]\nThink through human expression, initial pose, and primitive structure internally, but do not output planning comment lines.\nOutput only the final JSON object.\n",
    )
    replacements = {
        "- [ ] Q1-Q3 discuss motion semantics, not JSON formatting\n": "",
        "- [ ] Q2 ranks multiple plausible initial directions and clearly picks one winner\n": "",
        "- [ ] Q3 compares multiple primitive structures and clearly picks one winner\n": "",
        "5. Output the reasoning first as `#` comment lines, then output exactly one JSON object.\n": "5. Output exactly one JSON object and no planning lines.\n",
        "6. Do not use markdown code blocks.\n": "6. Do not use markdown code blocks.\n",
        "7. Do not add extra explanation before or after the JSON.\n": "7. Do not add extra explanation before or after the JSON.\n",
        "[Strict Output Format]\n- You must output exactly three planning lines:\n  - `# Q1: ...`\n  - `# Q2: ...`\n  - `# Q3: ...`\n- Q2 must be a compact ranked comparison, for example: `# Q2: candidates=P1) front+vertical+x80,y50,z55, P2) up+horizontal+z85; winner=1 because ...`\n- Q3 must be a compact structure comparison, for example: `# Q3: options=M1 P1>movement(wrist y repeat), M2 P1>path(arc xy), M3 P2>pose>movement; winner=M3 because ...`\n- Do not write any prose paragraphs, bullet lists, or plain text outside those three `#` lines.\n- After `# Q3: ...`, the very next non-empty line must be `{`.\n- The JSON object must be the only non-comment content in the entire response.\n": "[Strict Output Format]\n- Output exactly one JSON object.\n- Do not output planning lines, prose paragraphs, bullet lists, or extra text.\n- The very first non-empty character in the response must be `{`.\n",
        "### Required Response Skeleton\n# Q1: <one sentence>\n# Q2: <one sentence>\n# Q3: <one sentence>\n{\n": "### Required Response Skeleton\n{\n",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _make_no_pose_compare_prompt(base_prompt: str) -> str:
    text = base_prompt
    text = text.replace(
        "# Q2: Which initial anchor pose directions are the strongest candidates, ranked by iconic readability?\n  - Compare multiple plausible initial directions first. Direction is the top priority.\n  - Q2 must rank at least 2 candidate directions for the first pose.\n  - For the strongest candidates, include `gripper_orientation`, and include `x`, `y`, `z` when they matter to readability.\n  - Explicitly state which candidate wins and why it wins over the others.\n",
        "# Q2: Which single initial anchor pose should the motion start from?\n  - Choose one direction directly instead of comparing multiple candidates.\n  - Include `gripper_orientation`, and include `x`, `y`, `z` when they matter to readability.\n  - Explain why this one pose is the clearest anchor.\n",
    )
    text = text.replace(
        "- [ ] Q2 ranks multiple plausible initial directions and clearly picks one winner\n",
        "- [ ] Q2 picks one clear initial anchor pose directly\n",
    )
    text = text.replace(
        "- Q2 must be a compact ranked comparison, for example: `# Q2: candidates=P1) front+vertical+x80,y50,z55, P2) up+horizontal+z85; winner=1 because ...`\n",
        "- Q2 must describe a single chosen initial pose, for example: `# Q2: chosen=front+vertical+x80,y50,z55 because ...`\n",
    )
    return text


def _variant_specs(prompt_text: str) -> Dict[str, Dict]:
    return {
        "one_shot_no_reasoning": {
            "prompt_text": _make_one_shot_prompt(prompt_text),
            "use_shots": True,
            "require_reasoning": False,
        },
        "no_fewshot": {
            "prompt_text": prompt_text,
            "use_shots": False,
            "require_reasoning": True,
        },
        "no_pose_compare": {
            "prompt_text": _make_no_pose_compare_prompt(prompt_text),
            "use_shots": True,
            "require_reasoning": True,
        },
    }


def main(
    base_prompt_file: str = "/Users/sb/Downloads/workspace/dev_robosuite/data/seed/prompt/prompt_v18.txt",
    source_config_json: str = "/Users/sb/Downloads/workspace/dev_robosuite/data/seed/motion_configs_prompt_v18.json",
    output_dir: str = "/Users/sb/Downloads/workspace/dev_robosuite/data/seed/ablation_v18",
    limit: int = 10,
    model_name: str = "gemini-2.5-flash",
    temperature: float | None = 0.2,
    prepare_only: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    prompt_text = Path(base_prompt_file).read_text(encoding="utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompts_dir = Path(output_dir) / f"prompts_{ts}"
    variants = _variant_specs(prompt_text)
    cues = _first_n_cues(source_config_json, limit=limit)

    manifest = {
        "timestamp": ts,
        "source_config_json": source_config_json,
        "base_prompt_file": base_prompt_file,
        "model_name": model_name,
        "temperature": temperature,
        "limit": limit,
        "cues": [{"idx": row["idx"], "cue": row["cue"]} for row in cues],
        "variants": {},
    }

    for variant_name, spec in variants.items():
        prompt_path = _write_text(prompts_dir / f"{variant_name}.txt", spec["prompt_text"])
        output_json = str(Path(output_dir) / f"motion_configs_prompt_v18_{variant_name}_{ts}.json")
        manifest["variants"][variant_name] = {
            "prompt_file": prompt_path,
            "output_json": output_json,
            "use_shots": spec["use_shots"],
            "require_reasoning": spec["require_reasoning"],
        }
        print(f"\n=== Variant: {variant_name} ===")
        print(f"prompt_file={prompt_path}")
        print(f"output_json={output_json}")

        if prepare_only:
            continue

        for row in cues:
            print(f"[{variant_name}] c{row['idx']}: {row['cue']}")
            generate_motion_config(
                cue_name=row["cue"],
                cue_idx=int(row["idx"]),
                model_name=model_name,
                prompt_file=prompt_path,
                shots_json="data/seed/shot_configs.json",
                config_json=output_json,
                temperature=temperature,
                use_shots=bool(spec["use_shots"]),
                require_reasoning=bool(spec["require_reasoning"]),
            )

    manifest_path = Path(output_dir) / f"ablation_manifest_{ts}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    print(f"prepare_only={prepare_only}")


if __name__ == "__main__":
    fire.Fire(main)
