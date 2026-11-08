#!/bin/bash
# judgeall.sh <variant dir> [reps]  -- judge-equivalent measurement on all 15 points.
set -u
V=${1:?variant dir name under ~/gmm-rebuild}
REPS=${2:-2}
cd "$HOME/gmm-rebuild" || exit 1
C=$PWD/cases-b40da251a039
D=$PWD/$V/build/driver
OUT=$HOME/gmm-floor/judgeall-$V.tsv
printf "point\tjudge_local\n" > "$OUT"
for p in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15; do
  r=$(bash "$HOME/gmm-floor/judge.sh" "$C/tp${p}_even" "$D" x "$REPS" \
        | awk '{for(i=1;i<=NF;i++) if($i=="med"){print $(i+1); exit}}')
  if [ -z "$r" ]; then r="-"; fi
  printf "%s\t%s\n" "$p" "$r" | tee -a "$OUT"
done
echo "DONE $V"
