#!/bin/bash
# judge.sh <case dir> <driver> [label] [reps]
#
# The judge-equivalent measurement: operator-level profiling (msprof op / msopprof)
# with a warm-up large enough to clear the first-touch ramp, which is 40/blocks
# launches long.
#
# Validated 2026-11-08 against four kernels x five visible points. The decisive
# comparison is point 1 on h-merge: `msprof --application` warm median 21.20,
# judge-reported 27.99, `msprof op` 26.8-28.0. So the judge measures operator-level,
# not application-level, and application warm medians understate 20-block points by
# about 26%. The default --warm-up of 5 is far too small for 1-2 block points, whose
# ramp is 20-40 launches.
set -u
DIR=${1:?case dir}; DRIVER=${2:?driver}; LABEL=${3:-$(basename "$DIR")}; REPS=${4:-1}
ITERS=300; WARMUP=60
vals=""
for r in $(seq 1 "$REPS"); do
  v=$(bash "$HOME/gmm-floor/opdur.sh" "$DIR" "$DRIVER" $ITERS x --warm-up=$WARMUP 2>/dev/null \
        | awk '/^TOTAL /{print $2}')
  if [ -n "$v" ] && [ "$v" != "-" ]; then vals="$vals $v"; fi
done
echo "$vals" | awk -v l="$LABEL" -v w="$WARMUP" -v it="$ITERS" '
{
  n = 0
  for (i = 1; i <= NF; i++) { n++; v[n] = $i + 0 }
  if (n == 0) { printf "%-28s JUDGE  no data\n", l; exit 1 }
  for (i = 2; i <= n; i++) { x = v[i]; j = i - 1
    while (j > 0 && v[j] > x) { v[j+1] = v[j]; j-- }
    v[j+1] = x }
  med = (n % 2) ? v[(n+1)/2] : (v[n/2] + v[n/2+1]) / 2
  printf "%-28s JUDGE med %8.3f  (n %d, iters %d, warmup %d) |", l, med, n, it, w
  for (i = 1; i <= n; i++) printf " %7.3f", v[i]
  printf "\n"
}'
