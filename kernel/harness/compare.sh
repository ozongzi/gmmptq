#!/bin/bash
# Profile every case and line the result up against tbest.
#
#   compare.sh <cases dir> <driver> [iters] [split]
#
# tbest is the judge's best recorded time for that point, measured on the
# judge's machine; the measurement here is this container's. Whether the two
# machines are comparable has never been established, so the ratio column is a
# bearing, not a verdict. What this script is good for is comparing two
# kernels on the same box. The TBEST array below is also stale -- see
# UPDATES.md for the current figures.
set -u
CASES=${1:?cases dir}
DRIVER=${2:?driver}
ITERS=${3:-20}
SPLIT=${4:-even}
HERE=$(cd "$(dirname "$0")" && pwd)

# tbest per case, from shape.md.
TBEST=(0 31.06 12.45 1.24 28.86 1.23 1.07 2.50 2.50 16.91 9.59 9.04 11.79 20.01 64.79 63.57)
FLOOR=(0 11.72 0.58 0.24 2.22 0.24 0.24 0.25 0.25 0.44 0.61 0.43 0.72 2.79 11.95 11.94)

printf "%-6s %10s %10s %10s %8s %8s  %s\n" case local_min tbest floor "x tbest" "x floor" counters
printf '%s\n' "---------------------------------------------------------------------------------"
total_ratio=0
for id in $(seq 1 15); do
  dir="$CASES/tp$(printf %02d "$id")_$SPLIT"
  [ -d "$dir" ] || continue
  out=$("$HERE/profile.sh" "$dir" "$DRIVER" "$ITERS" 2>&1)
  min=$(echo "$out" | awk 'NR==1{for(i=1;i<=NF;i++) if($i=="min"){print $(i+1); exit}}')
  ctr=$(echo "$out" | awk 'NR==2{printf "blocks %s cube %s vec %s", $2, $10, $12}')
  if [ -z "${min:-}" ]; then
    printf "%-6s %10s %10s %10s %8s %8s  %s\n" "tp$id" "-" "${TBEST[$id]}" "${FLOOR[$id]}" "-" "-" "no sample"
    continue
  fi
  rt=$(awk -v a="$min" -v b="${TBEST[$id]}" 'BEGIN{printf "%.2f", a/b}')
  rf=$(awk -v a="$min" -v b="${FLOOR[$id]}" 'BEGIN{printf "%.1f", a/b}')
  printf "%-6s %10s %10s %10s %8s %8s  %s\n" "tp$id" "$min" "${TBEST[$id]}" "${FLOOR[$id]}" "$rt" "$rf" "$ctr"
done
