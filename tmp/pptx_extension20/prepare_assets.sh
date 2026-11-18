#!/usr/bin/env bash
set -euo pipefail

repo="/Users/sb/Downloads/workspace/dev_robosuite"
build="$repo/tmp/pptx_extension20"
mkdir -p "$build/eval_gifs" "$build/human_gifs"

jq -r '.[] | [.platform, (.idx|tostring), .target, .gif] | @tsv' \
  "$repo/data/results/evaluation/embodiment_extension20_hardneg_trials.json" |
while IFS=$'\t' read -r platform idx cue source; do
  target="$build/eval_gifs/${platform}_$(printf '%02d' "$idx")_${cue}.gif"
  gifsicle -O3 --colors 64 --resize-fit 420x340 --output "$target" "$source"
done

jq -r '.[] | [.idx, .cue] | @tsv' \
  "$repo/data/seed/agent_fill/embodiment_short/cues_humanlike90.json" |
while IFS=$'\t' read -r idx cue; do
  selected=false
  if [ "$idx" -le 30 ]; then
    selected=true
  elif jq -e --argjson idx "$idx" '.[] | select(.idx == $idx)' \
    "$repo/data/seed/agent_fill/embodiment_extension20/cues.json" >/dev/null; then
    selected=true
  fi
  if [ "$selected" != true ]; then
    continue
  fi
  for platform in humanoid tiago; do
    if [ "$idx" -le 30 ]; then
      source="$repo/data/results/render/embodiment_motion_code_gt/$platform/gif/emb_g$(printf '%02d' "$idx")_${cue}.gif"
    elif [ "$platform" = humanoid ]; then
      source="$repo/data/results/render/embodiment_motion_code_synth60_motion60/humanoid/gif/emb_g$(printf '%02d' "$idx")_${cue}.gif"
    else
      source="$repo/data/results/render/embodiment_motion_code_synth60_tiago60/tiago/gif/emb_g$(printf '%02d' "$idx")_${cue}.gif"
    fi
    target="$build/human_gifs/${platform}_$(printf '%02d' "$idx")_${cue}.gif"
    if [ ! -f "$source" ]; then
      echo "missing: $source" >&2
      exit 1
    fi
    gifsicle -O3 --colors 64 --resize-fit 360x300 --output "$target" "$source"
  done
done

echo "eval=$(find "$build/eval_gifs" -name '*.gif' | wc -l | tr -d ' ')"
echo "human=$(find "$build/human_gifs" -name '*.gif' | wc -l | tr -d ' ')"
