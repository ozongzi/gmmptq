#!/bin/bash
# Device-side timing for one case, via msprof.
#
#   profile.sh <case dir> <driver> [iters] [cores]
#
# Why msprof and not a host timer: at these sizes a launch plus synchronize is
# tens of microseconds of ACL overhead, so host wall time measures ACL, not the
# kernel. msprof reports each launch's own device duration. This path
# reproduced the official tuning guide's 218.1 us figure to within 3%, which is
# the only calibration we have that the numbers mean what we think.
#
# The driver launches `iters` times in one process, so one msprof run yields
# `iters` samples. min is the interesting one (least interference), median is
# the honest one.
set -u
DIR=${1:?case dir}
DRIVER=${2:?driver}
ITERS=${3:-20}
CORES=${4:-0}

PROF=$(mktemp -d)
trap 'rm -rf "$PROF"' EXIT

args=(--dir "$DIR" --iters "$ITERS")
[ "$CORES" -gt 0 ] && args+=(--cores "$CORES")

msprof --application="$DRIVER ${args[*]}" --output="$PROF" \
       --aic-metrics=ArithmeticUtilization --ai-core=on >"$PROF/msprof.log" 2>&1
rc=$?
SUM=$(find "$PROF" -name "op_summary*.csv" 2>/dev/null | head -1)
if [ -z "$SUM" ]; then
  echo "no op_summary produced (msprof rc=$rc); tail of its log:" >&2
  tail -5 "$PROF/msprof.log" >&2
  exit 1
fi

# Columns are named, not positional, so find them by header rather than
# trusting a fixed order across CANN versions.
awk -F',' -v dir="$(basename "$DIR")" '
NR==1 {
  for (i = 1; i <= NF; i++) {
    h = $i; gsub(/^"|"$|\r/, "", h)
    if (h == "Task Duration(us)") dur = i
    if (h == "aicore_time(us)")   aic = i
    if (h == "aiv_time(us)")      aiv = i
    if (h == "aic_mac_int8_ratio") mac = i
    if (h == "aiv_vec_fp32_ratio") vec = i
    if (h == "Block Num")         blk = i
    if (h == "Op Name")           nam = i
  }
  next
}
$dur != "" && $dur+0 > 0 {
  n++; d[n] = $dur+0
  sum += $dur+0
  if (min == 0 || $dur+0 < min) min = $dur+0
  if ($dur+0 > max) max = $dur+0
  amac += $mac+0; avec += $vec+0; ablk = $blk+0
  aaic += $aic+0; aaiv += $aiv+0
  name = $nam
}
END {
  if (n == 0) { print dir ": no timed launches"; exit 1 }
  asort_n = n
  # insertion sort; n is tens, not thousands
  for (i = 2; i <= n; i++) { v = d[i]; j = i - 1
    while (j > 0 && d[j] > v) { d[j+1] = d[j]; j-- } ; d[j+1] = v }
  med = (n % 2) ? d[(n+1)/2] : (d[n/2] + d[n/2+1]) / 2
  printf "%-16s n=%-3d min %8.3f us   median %8.3f us   max %8.3f us   spread %5.1f%%\n", \
         dir, n, min, med, max, (max-min)/min*100
  printf "%-16s blocks %-4d  aicore %7.2f us  aiv %7.2f us  cube_mac %5.1f%%  vec_f32 %5.1f%%\n", \
         "", ablk, aaic/n, aaiv/n, amac/n*100, avec/n*100
}' "$SUM"
