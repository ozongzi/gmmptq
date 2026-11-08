#!/bin/bash
# ABBA 交错对比两份内核。用法: abba.sh <reps> <iters> <out.tsv> [点号...]
cd ~/gmm-rebuild
source $HOME/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
REPS=${1:-2}; ITERS=${2:-20}; OUT=${3:-cmp.tsv}; shift 3
PTS=("$@"); [ ${#PTS[@]} -eq 0 ] && PTS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
: > $OUT
med() { bash harness/profile.sh "$1" "$2" "$ITERS" 2>/dev/null | awk "/median/{for(i=1;i<=NF;i++) if(\$i==\"median\"){print \$(i+1); exit}}"; }
for p in "${PTS[@]}"; do
  d=cases-b40da251a039/tp$(printf %02d $p)_even
  for r in $(seq 1 $REPS); do
    if [ $((r % 2)) -eq 1 ]; then o1=prod; o2=k9; else o1=k9; o2=prod; fi
    a=$(med $d $PWD/h-$o1/build/driver); b=$(med $d $PWD/h-$o2/build/driver)
    echo -e "$p\t$r\t$o1\t$a" >> $OUT
    echo -e "$p\t$r\t$o2\t$b" >> $OUT
  done
  echo "P$p done" >&2
done
