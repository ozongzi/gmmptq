#!/bin/bash
# judgefit.sh  -- run the judge-equivalent measurement over the 20-pair validation set
# and write a TSV. Four kernels with known judge-reported times x the five visible points.
set -u
cd "$HOME/gmm-rebuild" || exit 1
C=$PWD/cases-b40da251a039
OUT=$HOME/gmm-floor/judgefit.tsv
printf "point\tkernel\tjudge_local\n" > "$OUT"
for p in 01 02 03 04 05; do
  for v in h-prod h-merge h-m-p2pk h-m-p2b32; do
    r=$(bash "$HOME/gmm-floor/judge.sh" "$C/tp${p}_even" "$PWD/$v/build/driver" x 1 \
          | awk '{for(i=1;i<=NF;i++) if($i=="med"){print $(i+1); exit}}')
    if [ -z "$r" ]; then r="-"; fi
    printf "%s\t%s\t%s\n" "$p" "$v" "$r" | tee -a "$OUT"
  done
done
echo "DONE"
