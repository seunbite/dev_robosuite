#!/usr/bin/env bash
cd ~/sblee/dev_robosuite
conda init
conda activate m2m_caption32b

task=${1:-"all"}
model=${2:-"Qwen/Qwen2.5-VL-32B-Instruct"}
backend=${3:-"transformers"}

cmd="python adhoc/generation/robotarm/exp.py ${task} --backend ${backend} --model ${model}"
echo ${cmd}
eval ${cmd}