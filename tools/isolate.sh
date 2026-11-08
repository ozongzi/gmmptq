#!/bin/bash
# mkiso.sh <base.asc> <ref-with-probe.asc> <point 1..15> <out.asc>
# Splice the proven spin instrument (GmmProbeSpin / kGmmProbeTicks / GmmProbePoint)
# into a kernel that lacks it, isolating one test point: every OTHER point gets
# 50.00 ms of pure spin appended after its real kernel, so the judge's total score
# is (14 * a known constant + s_point) / 15 and s_point falls out exactly.
# Byte-exact on the base: head/tail only, never sed over the whole file -- the base
# has mixed CRLF/LF and must survive unchanged.  Seams are located on a CR-stripped
# index copy, which has the same line numbering.
set -eu
BASE=$1; REF=$2; P=$3; OUT=$4
case "$P" in ''|*[!0-9]*) echo "point must be 0..15 (0 = spin every point: the calibration run)"; exit 1;; esac
[ "$P" -ge 0 ] && [ "$P" -le 15 ] || { echo "point must be 0..15 (0 = spin every point: the calibration run)"; exit 1; }
IDX=$(mktemp); trap 'rm -f "$IDX"' EXIT
tr -d '\r' < "$BASE" > "$IDX"

one() {  # one <file> <ere> -- the single line number matching, or die
  local n; n=$(grep -cE "$2" "$1") || true
  [ "$n" = 1 ] || { echo "seam '$2' matched $n times in $1" >&2; exit 1; }
  grep -nE "$2" "$1" | cut -d: -f1
}
SIG=$(one "$IDX" '^extern "C" void run_kernel\($')
PB_A=$(one "$REF" '^#ifndef GMM_PROBE$')
PB_B=$(( $(one "$REF" '^static void GmmRunKernelBody\($') - 1 ))
WR_A=$(one "$REF" '^extern "C" void run_kernel\($')
WR_B=$(wc -l < "$REF")
sed -n "${PB_B}p" "$REF" | grep -qx '#endif' || { echo "probe block does not end on #endif"; exit 1; }

# the tick vector: 2 500 000 ticks x 20 ns = 50.00 ms appended to every other point
T=""
for i in $(seq 1 15); do
  [ -n "$T" ] && T="$T,"
  if [ "$i" = "$P" ]; then T="${T}0ull"; else T="${T}2500000ull"; fi
done

{
  printf '#define GMM_PROBE 1\n'
  printf '#define GMM_PROBE_TICKS %s\n' "$T"
  printf '#define GMM_PROBE_EMPTY 0u\n'
  printf '#define GMM_ASK 0u\n'
  head -n $((SIG-1)) "$BASE"
  sed -n "${PB_A},${PB_B}p" "$REF"
  printf 'static void GmmRunKernelBody(\n'
  tail -n +$((SIG+1)) "$BASE"
  sed -n "${WR_A},${WR_B}p" "$REF"
} > "$OUT"

echo "seams: base sig=$SIG   ref probe=$PB_A..$PB_B   ref wrapper=$WR_A..$WR_B"
echo "out:   $(wc -l < "$OUT") lines   ($(wc -l < "$BASE") base + $((PB_B-PB_A+1)) probe + $((WR_B-WR_A+1)) wrapper + 4 defines)"
