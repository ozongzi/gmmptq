#!/bin/bash
# judgeab.sh <variantA> <variantB> [reps] -- judge-equivalent A/B across all 15 points,
# interleaved per point so box drift cannot bias the comparison.
set -u
A=${1:?variant A}; B=${2:?variant B}; REPS=${3:-2}
cd "$HOME/gmm-rebuild" || exit 1
C=$PWD/cases-b40da251a039
OUT=$HOME/gmm-floor/judgeab-$A-vs-$B.tsv
printf "point\t%s\t%s\tdelta_pct\n" "$A" "$B" > "$OUT"
for p in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15; do
  va=$(bash "$HOME/gmm-floor/judge.sh" "$C/tp${p}_even" "$PWD/$A/build/driver" x "$REPS" \
        | awk '{for(i=1;i<=NF;i++) if($i=="med"){print $(i+1); exit}}')
  vb=$(bash "$HOME/gmm-floor/judge.sh" "$C/tp${p}_even" "$PWD/$B/build/driver" x "$REPS" \
        | awk '{for(i=1;i<=NF;i++) if($i=="med"){print $(i+1); exit}}')
  if [ -z "$va" ]; then va="-"; fi
  if [ -z "$vb" ]; then vb="-"; fi
  d=$(awk -v a="$va" -v b="$vb" 'BEGIN{ if(a+0>0 && b+0>0) printf "%+.1f", 100*(b-a)/a; else printf "-" }')
  printf "%s\t%s\t%s\t%s\n" "$p" "$va" "$vb" "$d" | tee -a "$OUT"
done
echo "DONE"
