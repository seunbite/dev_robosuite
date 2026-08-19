import json
import math
import re
from pathlib import Path

_REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
RESULT_JSON = _REPO / "data/results/motion_configs/tiago/exp/result_exp1_mobile-map.json"
PLAUSIBLE_HTML = _REPO / "data/results/visualize/tiago/plausible_set/index.html"
OUTPUT_DIR = _REPO / "data/results/visualize/tiago/plausible_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# User's selection of plausible tile indices (1-based)
USER_SELECTION = {
    "front_horizontal": [2, 4, 6, 8, 17, 18, 20],
    "front_vertical": [7, 11, 13, 18, 23],
    "back_horizontal": [1, 19],
    "back_vertical": [15],
    "in_horizontal": [2, 10, 13, 17],
    "in_vertical": [2, 9, 14],
    "out_horizontal": [1, 6, 11],
    "out_vertical": [1, 2, 16],
    "up_horizontal": [1, 2, 10, 15, 16, 22, 24],
    "up_vertical": [1, 2, 11, 14, 18],
    "down_horizontal": [1, 2, 6],
    "down_vertical": [1, 10, 21, 23],
}

def get_nominal_group_key(pose):
    arm = pose.get("arm_position", "front")
    orient = pose.get("gripper_orientation", "horizontal")
    return f"{arm}_{orient}"

def run():
    # 1. Load data
    results = json.loads(RESULT_JSON.read_text())
    html_content = PLAUSIBLE_HTML.read_text()
    
    # 2. Parse HTML for tile metadata
    # <article><h3>front_horizontal #2</h3><p class="meta">xyz=(58.754, 32.399, 70.549)<br/>j=[0, 0, 0, 90, 0, -90]</p>
    pattern = r'<h3>(.*?) #(\d+)</h3>.*?xyz=\((.*?)\).*?j=\[(.*?)\]'
    matches = re.findall(pattern, html_content, re.S)
    
    plausible_metadata = {}
    for group, idx, xyz_str, joints_str in matches:
        idx = int(idx)
        xyz = [float(x.strip()) for x in xyz_str.split(',')]
        # joints = [float(x.strip()) for x in joints_str.split(',')]
        if group not in plausible_metadata: plausible_metadata[group] = {}
        plausible_metadata[group][idx] = {'xyz': xyz}

    # 3. Perform selection for each cue
    selections = []
    for res in results:
        cue = res['cue']
        # Find first pose
        nominal_pose = None
        for m in res['movements']:
            if m['type'] == 'pose':
                nominal_pose = m['parameters']['pose']
                break
        
        if not nominal_pose:
            continue
            
        gk = get_nominal_group_key(nominal_pose)
        
        # Get plausible candidates for this group from parsed metadata
        group_meta = plausible_metadata.get(gk, {})
        if not group_meta:
            print(f"No plausible metadata for group {gk} (cue: {cue})")
            continue
            
        # Target z (nominal)
        target_z = nominal_pose.get('z', 50)
        
        # Manual/Heuristic selection from the available plausible tiles
        candidates = []
        for idx, meta in group_meta.items():
            candidates.append((idx, meta))
            
        if not candidates:
            continue
            
        # Selection logic
        best_idx = candidates[0][0]
        best_meta = candidates[0][1]
        
        # Refinement based on cue semantics:
        if any(w in cue for w in ["victory", "toast", "highfive", "up", "raise"]):
            # Pick the highest one
            best_idx, best_meta = max(candidates, key=lambda x: x[1]['xyz'][2])
        elif any(w in cue for w in ["scratch", "think", "crazy", "face", "head", "yawn", "gasp", "shh", "visor"]):
            # Pick one near head height (z ~ 70-80)
            best_idx, best_meta = min(candidates, key=lambda x: abs(x[1]['xyz'][2] - 75))
        elif any(w in cue for w in ["stomach", "pat", "down", "low"]):
            # Pick one lower (z ~ 30-40)
            best_idx, best_meta = min(candidates, key=lambda x: abs(x[1]['xyz'][2] - 35))
        else:
            # Default: closest in z
            best_idx, best_meta = min(candidates, key=lambda x: abs(x[1]['xyz'][2] - target_z))

        selections.append({
            "cue": cue,
            "group": gk,
            "nominal_pose": (nominal_pose.get('x'), nominal_pose.get('y'), nominal_pose.get('z')),
            "best_tile_idx": best_idx,
            "best_pose_xyz": best_meta['xyz'],
            "reason": f"Selected tile #{best_idx} from {gk} plausible set as most appropriate for '{cue}'."
        })

    # 4. Save results
    (OUTPUT_DIR / "selections.json").write_text(json.dumps(selections, indent=2))
    
    # 5. Generate HTML report
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Tiago - Most Appropriate Plausible Poses</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#0f172a;color:#e2e8f0}}
.wrap{{padding:16px}}h1{{font-size:1.5rem;margin-bottom:8px}}
table{{width:100%;border-collapse:collapse;margin-top:16px}}
th,td{{padding:12px;text-align:left;border-bottom:1px solid #334155}}
th{{background:#1e293b}}
img{{width:150px;height:auto;border-radius:4px}}
.nominal{{color:#94a3b8;font-size:12px}}
</style></head><body><div class="wrap">
<h1>Most Appropriate Plausible Poses for Pilot-40</h1>
<p>Mapping original cues to the best pose from the user-curated plausible set.</p>
<table>
<thead><tr><th>Cue</th><th>Group</th><th>Nominal Pose</th><th>Selected Plausible Pose</th><th>Reason</th></tr></thead>
<tbody>
"""
    for s in selections:
        img_path = f"../plausible_set/tiles/{s['group']}_tile_{s['best_tile_idx']:02d}.png"
        html += f"""
<tr>
    <td><b>{s['cue']}</b></td>
    <td>{s['group']}</td>
    <td class="nominal">xyz={s['nominal_pose']}</td>
    <td>
        <div>Tile #{s['best_tile_idx']}</div>
        <img src="{img_path}"/>
        <div class="nominal">xyz=({s['best_pose_xyz'][0]:.1f}, {s['best_pose_xyz'][1]:.1f}, {s['best_pose_xyz'][2]:.1f})</div>
    </td>
    <td>{s['reason']}</td>
</tr>
"""
    html += "</tbody></table></div></body></html>"
    (OUTPUT_DIR / "index.html").write_text(html)
    print(f"Report generated: {OUTPUT_DIR / 'index.html'}")

if __name__ == "__main__":
    run()
