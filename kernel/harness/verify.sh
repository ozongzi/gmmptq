#!/bin/bash
cd ~/gmm-rebuild
source $HOME/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
for v in "$@"; do
  d=$PWD/h-$v/build/driver
  [ -f "$d" ] || { echo "$v: 没有 driver"; continue; }
  r1=$(bash harness/sweep.sh cases-b40da251a039 "$d" $PWD/gmmbench 1 2>&1 | tail -1)
  r2=$(bash harness/sweep.sh even "$d" $PWD/gmmbench 1 2>&1 | tail -1)
  echo "$v | 真实分组15点: $r1 | 合成75例: $r2"
done
