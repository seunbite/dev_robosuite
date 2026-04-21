import glob
import json
import os
import random
from pathlib import Path


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED_DIR = ROOT / "data" / "seed"
MOTION_DIR = ROOT / "data" / "motions"
OUT_DIR = ROOT / "data" / "human_eval" / "sophisticated_v1"
PROMPT_PATH = OUT_DIR / "labeler_prompt_ko.md"
SEED = 20260404
NUM_LABELERS = 8
GROUP_SIZE = 13
BATCH_SIZE = 26


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_match(patterns: list[str]) -> str | None:
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    matches = [m for m in matches if not m.endswith("_preview.gif")]
    if not matches:
        return None
    return sorted(set(matches), key=os.path.getmtime, reverse=True)[0]


def _find_latest_single_gif(render_dir: Path, cue: str) -> str | None:
    safe_cue = cue.replace("/", "_").replace("\\", "_").replace(" ", "_")
    patterns = [
        str(render_dir / f"*_{safe_cue}_p*.gif"),
        str(render_dir / f"*_{safe_cue}_*.gif"),
    ]
    return _latest_match(patterns)


def _cue_tokens(cue: str) -> set[str]:
    return {tok for tok in cue.lower().split("_") if tok}


def _cue_key(testset: str, idx: int) -> str:
    return f"{testset}_c{idx}"


def _collect_cues() -> list[dict]:
    specs = [
        ("iconic", SEED_DIR / "motion_configs_prompt_v19_sophisticated.json", MOTION_DIR / "v19_sophisticated" / "IIWA"),
        ("contextual", SEED_DIR / "motion_configs_prompt_v19_sophisticated_contextual.json", MOTION_DIR / "v19_sophisticated_contextual" / "IIWA"),
    ]
    rows = []
    for testset, config_path, motion_dir in specs:
        for cfg in _load_json(config_path):
            cue = cfg["cue"]
            gif_path = _find_latest_single_gif(motion_dir, cue)
            if not gif_path:
                raise FileNotFoundError(f"Missing top1 gif for {testset} {cfg['idx']} {cue}")
            rows.append(
                {
                    "cue_key": _cue_key(testset, int(cfg["idx"])),
                    "testset": testset,
                    "cue_idx": int(cfg["idx"]),
                    "cue": cue,
                    "cue_tokens": sorted(_cue_tokens(cue)),
                    "gif_path": gif_path,
                }
            )
    if len(rows) != 104:
        raise ValueError(f"Expected 104 cues, got {len(rows)}")
    return rows


def _build_false_mapping(cues: list[dict]) -> dict[str, dict]:
    rng = random.Random(SEED)
    compat: dict[str, list[dict]] = {}
    for src in cues:
        src_tokens = set(src["cue_tokens"])
        choices = [t for t in cues if t["cue_key"] != src["cue_key"] and src_tokens.isdisjoint(set(t["cue_tokens"]))]
        rng.shuffle(choices)
        compat[src["cue_key"]] = choices

    match_to_src: dict[str, str] = {}

    def _dfs(src_key: str, seen: set[str]) -> bool:
        for target in compat[src_key]:
            tkey = target["cue_key"]
            if tkey in seen:
                continue
            seen.add(tkey)
            if tkey not in match_to_src or _dfs(match_to_src[tkey], seen):
                match_to_src[tkey] = src_key
                return True
        return False

    for src in sorted(cues, key=lambda x: len(compat[x["cue_key"]])):
        if not _dfs(src["cue_key"], set()):
            raise RuntimeError(f"Could not assign false cue for {src['cue_key']}")

    lookup = {row["cue_key"]: row for row in cues}
    false_map: dict[str, dict] = {}
    for false_key, src_key in match_to_src.items():
        false_map[src_key] = lookup[false_key]
    return false_map


def _split_groups(cues: list[dict]) -> list[list[dict]]:
    rng = random.Random(SEED)
    shuffled = list(cues)
    rng.shuffle(shuffled)
    groups = [shuffled[i * GROUP_SIZE:(i + 1) * GROUP_SIZE] for i in range(NUM_LABELERS)]
    if any(len(group) != GROUP_SIZE for group in groups):
        raise ValueError("Group sizing failed")
    return groups


def _labeler_plan(groups: list[list[dict]]) -> list[tuple[int, int]]:
    # L1 A yes / B no, L2 C yes / D no, L3 E yes / F no, L4 G yes / H no
    # L5 B yes / A no, L6 D yes / C no, L7 F yes / E no, L8 H yes / G no
    return [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (1, 0),
        (3, 2),
        (5, 4),
        (7, 6),
    ]


def _build_batches(groups: list[list[dict]], false_map: dict[str, dict]) -> list[list[dict]]:
    rng = random.Random(SEED)
    plan = _labeler_plan(groups)
    batches: list[list[dict]] = []

    for batch_idx, (yes_group_idx, no_group_idx) in enumerate(plan, start=1):
        items = []
        labeler = f"labeler_{batch_idx}"

        for cue in groups[yes_group_idx]:
            items.append(
                {
                    "assignment_id": f"batch{batch_idx:02d}_{len(items)+1:02d}",
                    "batch_id": batch_idx,
                    "suggested_labeler": labeler,
                    "pair_id": cue["cue_key"],
                    "testset": cue["testset"],
                    "cue_idx": cue["cue_idx"],
                    "source_cue": cue["cue"],
                    "shown_cue": cue["cue"],
                    "question_type": "binary_match",
                    "ground_truth": "yes",
                    "gif_path": cue["gif_path"],
                    "metadata": {
                        "task_variant": "true",
                        "group_role": "yes_group",
                        "group_index": yes_group_idx,
                        "false_cue": false_map[cue["cue_key"]]["cue"],
                    },
                }
            )

        for cue in groups[no_group_idx]:
            false_cue = false_map[cue["cue_key"]]
            items.append(
                {
                    "assignment_id": f"batch{batch_idx:02d}_{len(items)+1:02d}",
                    "batch_id": batch_idx,
                    "suggested_labeler": labeler,
                    "pair_id": cue["cue_key"],
                    "testset": cue["testset"],
                    "cue_idx": cue["cue_idx"],
                    "source_cue": cue["cue"],
                    "shown_cue": false_cue["cue"],
                    "question_type": "binary_match",
                    "ground_truth": "no",
                    "gif_path": cue["gif_path"],
                    "metadata": {
                        "task_variant": "false",
                        "group_role": "no_group",
                        "group_index": no_group_idx,
                        "false_cue_key": false_cue["cue_key"],
                        "false_cue_tokens": false_cue["cue_tokens"],
                    },
                }
            )

        rng.shuffle(items)
        for order, item in enumerate(items, start=1):
            item["order_in_batch"] = order
            item["assignment_id"] = f"batch{batch_idx:02d}_{order:02d}"
        batches.append(items)

    return batches


def _validate(batches: list[list[dict]]):
    if len(batches) != NUM_LABELERS:
        raise ValueError("Unexpected batch count")
    if any(len(batch) != BATCH_SIZE for batch in batches):
        raise ValueError("Unexpected batch size")

    pair_counts: dict[str, int] = {}
    truth_counts: dict[str, dict[str, int]] = {}
    for batch in batches:
        seen_in_batch = set()
        yes_count = 0
        no_count = 0
        for item in batch:
            pair_id = item["pair_id"]
            if pair_id in seen_in_batch:
                raise ValueError(f"Same cue appears twice in one batch: {pair_id}")
            seen_in_batch.add(pair_id)
            pair_counts[pair_id] = pair_counts.get(pair_id, 0) + 1
            truth_counts.setdefault(pair_id, {"yes": 0, "no": 0})
            truth_counts[pair_id][item["ground_truth"]] += 1
            if item["ground_truth"] == "yes":
                yes_count += 1
            else:
                no_count += 1
                if not _cue_tokens(item["source_cue"]).isdisjoint(_cue_tokens(item["shown_cue"])):
                    raise ValueError(f"False cue overlaps source cue: {pair_id}")
        if yes_count != 13 or no_count != 13:
            raise ValueError("Each labeler must have 13 yes and 13 no")

    if any(count != 2 for count in pair_counts.values()):
        raise ValueError("Each source gif must appear exactly twice")
    if any(v["yes"] != 1 or v["no"] != 1 for v in truth_counts.values()):
        raise ValueError("Each source gif must have one yes and one no task")


def _write_prompt():
    text = """# Human Evaluation Prompt

## 목적
각 item에서 cue 텍스트와 motion GIF 하나를 보고, 이 motion이 해당 cue로 읽히는지 binary로 판단합니다.

## 레이블링 질문
이 GIF는 제시된 cue로 보이는가?

## 출력 형식
- `yes`
- `no`

## 판단 기준
1. cue 의미가 motion에서 명확하게 읽히는가
2. 핵심 gesture 또는 path가 cue와 맞는가
3. 동작이 자연스럽고 안정적으로 보이는가
4. 일부만 비슷한 것이 아니라 전체적으로 cue 의미가 맞는가

## 주의사항
- 각 item은 독립적으로 판단합니다.
- 현재 보이는 GIF와 현재 적힌 cue만 보고 판단합니다.
- 다른 item과 비교하지 않습니다.

## 한 줄 버전
`이 GIF가 지금 적힌 cue로 읽히면 yes, 아니면 no를 고르세요.`
"""
    PROMPT_PATH.write_text(text, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cues = _collect_cues()
    false_map = _build_false_mapping(cues)
    groups = _split_groups(cues)
    batches = _build_batches(groups, false_map)
    _validate(batches)

    manifest = {
        "seed": SEED,
        "source_gif_count": len(cues),
        "total_binary_items": len(cues) * 2,
        "batch_count": len(batches),
        "batch_size": BATCH_SIZE,
        "group_size": GROUP_SIZE,
        "labeler_plan": {
            "labeler_1": "A yes + B no",
            "labeler_2": "C yes + D no",
            "labeler_3": "E yes + F no",
            "labeler_4": "G yes + H no",
            "labeler_5": "B yes + A no",
            "labeler_6": "D yes + C no",
            "labeler_7": "F yes + E no",
            "labeler_8": "H yes + G no",
        },
        "notes": [
            "Each labeler sees 26 items: 13 yes and 13 no.",
            "No labeler sees the same source gif twice.",
            "False cues share zero tokens with the source cue.",
            "Order within each batch is shuffled.",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for batch_idx, batch in enumerate(batches, start=1):
        payload = {
            "batch_id": batch_idx,
            "suggested_labeler": f"labeler_{batch_idx}",
            "item_count": len(batch),
            "items": batch,
        }
        (OUT_DIR / f"sophisticated_eval_batch_{batch_idx:02d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    _write_prompt()
    print(OUT_DIR)


if __name__ == "__main__":
    main()
