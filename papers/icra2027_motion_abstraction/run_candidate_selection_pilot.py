"""Six-cue, six-request visual GT/O/L/M selection pilot; cached, no retries.

Reuse existing humanlike90 negatives and rendered clips. PNG rows are chronological
storyboards, not still poses. Model sees no code, foil labels, or GT positions.
Run without --call to prepare/inspect; --call makes at most six Gemini requests.
"""
from pathlib import Path
import argparse, base64, hashlib, json, os, random, re, shlex, time
import urllib.request
from PIL import Image, ImageDraw, ImageFont, ImageSequence

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "data/results/evaluation/embodiment_cue_id/results_gt_olm_negatives_pilot20"
OUT = Path(__file__).resolve().parent / "candidate_selection_pilot6"
CUES = {
    "salute": "Salute: bring a hand to the brow in a conventional salute.",
    "highfive_invite": "Invite a high five: raise and present an open hand to another person.",
    "stop_palm_out": "Signal stop: present an outward-facing open palm and hold.",
    "cover_yawn_sleepy": "Cover the mouth while yawning sleepily.",
    "cover_ears_too_loud": "Cover both ears because the sound is too loud.",
    "shh_gesture": "Signal quiet: place a vertical finger or hand at the lips.",
}
MODEL = "gemini-2.5-pro"

def key():
    value = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if value:
        return value
    # Parse assignments only; do not execute the credential file or log its values.
    for line in (REPO / "APIKEY.sh").read_text().splitlines():
        parts = shlex.split(line, comments=True)
        for part in parts:
            if part.startswith(("GOOGLE_API_KEY=", "GEMINI_API_KEY=")):
                value = part.split("=", 1)[1]
    if not value:
        raise RuntimeError("Missing API credential")
    return value

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--call", action="store_true")
    args = ap.parse_args(); OUT.mkdir(exist_ok=True)
    rows = json.loads((SOURCE / "negatives.json").read_text())
    manifest = []
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    for ci, (cue, gloss) in enumerate(CUES.items()):
        row = next(r for r in rows if r["platform"] == "humanoid" and r["cue"] == cue)
        kinds = ["ori", "loc", "move"]
        random.Random(20260912 + ci).shuffle(kinds)
        kinds.insert(ci % 4, "gt")  # balanced as closely as six trials permit
        sheet = Image.new("RGB", (8 * 230 + 40, 4 * 252), "white")
        draw = ImageDraw.Draw(sheet); candidates = []
        for ri, kind in enumerate(kinds):
            name = cue if kind == "gt" else row["neg"][kind]["file_cue"]
            path = SOURCE / "render" / kind / "humanoid/gif" / f"emb_g{row['idx']:02d}_{name}.gif"
            frames = [im.convert("RGB") for im in ImageSequence.Iterator(Image.open(path))]
            indices = [round(j * (len(frames)-1) / 7) for j in range(8)]
            draw.text((6, ri * 252 + 90), chr(65+ri), font=font, fill="black")
            signature = hashlib.sha256()
            for j, fi in enumerate(indices):
                frame = frames[fi]
                frame = frame.crop((0, 0, frame.width, frame.height-14)).resize((230, 230))
                signature.update(frame.tobytes())
                sheet.paste(frame, (40+j*230, ri*252))
                draw.text((45+j*230, ri*252+231), f"{j+1}/8", font=font, fill="black")
            code = row["gt_code"] if kind == "gt" else row["neg"][kind]["code"]
            candidates.append(dict(label=chr(65+ri), kind=kind, code=code,
                                   media=str(path.relative_to(REPO)), frames=len(frames),
                                   sampled_indices=indices, pixel_sha256=signature.hexdigest()))
        if len({c["pixel_sha256"] for c in candidates}) != 4:
            raise RuntimeError(f"Pixel-identical candidates: {cue}")
        png = OUT / f"{cue}.png"; sheet.save(png)
        manifest.append(dict(cue=cue, gloss=gloss, candidates=candidates, image=png.name))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "gallery.html").write_text('<html><body style="font-family:sans-serif">' +
        '<h1>Six-cue selection pilot: chronological frames, left to right</h1>' +
        ''.join(f'<h2>{r["cue"]}</h2><img style="width:100%" src="{r["image"]}">' for r in manifest) + '</body></html>')
    for row in manifest:
        target = OUT / f'{row["cue"]}.response.json'
        if target.exists() or not args.call:
            continue
        prompt = (f'Target gesture: {row["gloss"]}\n'
            'The image contains four candidate robot motions, rows A-D. Each row shows eight '
            'chronologically ordered frames from left to right. Select the motion that best '
            'communicates the target. Consider orientation, body-relative placement, and movement. '
            'Do not assume the first candidate is correct. Return JSON only: '
            '{"choice":"A|B|C|D", "reason":"one short sentence"}.')
        body = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {
            "mime_type": "image/png", "data": base64.b64encode((OUT/row["image"]).read_bytes()).decode()}}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 2048,
                                  "thinkingConfig": {"thinkingBudget": 128},
                                  "responseMimeType": "application/json"}}
        # Save request metadata without image payload or credentials for reproducibility.
        (OUT/f'{row["cue"]}.request.json').write_text(json.dumps({"model":MODEL,
            "prompt":prompt, "generationConfig":body["generationConfig"]}, indent=2))
        req = urllib.request.Request(f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent',
            data=json.dumps(body).encode(), headers={"Content-Type":"application/json", "x-goog-api-key":key()})
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = json.load(response)
        target.write_text(json.dumps(raw, indent=2))
        print('Completed', row['cue'], flush=True)
    results = []
    for row in manifest:
        p = OUT/f'{row["cue"]}.response.json'
        if not p.exists(): continue
        raw = json.loads(p.read_text())
        text = ''.join(p.get('text','') for p in raw['candidates'][0]['content']['parts'] if not p.get('thought'))
        answer = json.loads(text)
        selected = next((c for c in row['candidates'] if c['label']==answer.get('choice')), None)
        results.append(dict(cue=row['cue'], answer=answer, selected_kind=selected['kind'] if selected else 'invalid',
                            correct=bool(selected and selected['kind']=='gt'), usage=raw.get('usageMetadata',{})))
    summary = dict(model=MODEL, n=len(results), correct=sum(r['correct'] for r in results),
                   chance=.25, max_requests=6, results=results,
                   limitations=['Convenience sample; one order per cue; no human rating.',
                     'Code-group perturbations may be coupled by IK; image differences are not proof of channel isolation.',
                     'Eight-frame PNGs may miss brief temporal events; not equivalent to continuous video.'])
    (OUT/'results.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
