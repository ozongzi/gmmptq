#!/bin/bash
# A/B one kernel tweak across a set of cases: correctness first, then time.
#
#   ab.sh <cases dir> <gmmbench> <label> <sed expression> [case ids...]
#
# The sed expression is applied to kernel.asc; base.asc is restored afterwards,
# so a failed experiment leaves nothing behind. A variant that breaks
# correctness is reported as such and its timings are not shown -- a faster
# wrong kernel is worth nothing.
set -u
CASES=${1:?cases dir}
BENCH=${2:?gmmbench}
LABEL=${3:?label}
EXPR=${4:?sed expression}
shift 4
IDS=("$@")
[ ${#IDS[@]} -eq 0 ] && IDS=(1 2 4 9 12 14 15)
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

[ -f base.asc ] || cp kernel.asc base.asc
cp base.asc kernel.asc
sed -i "$EXPR" kernel.asc
if ! cmp -s base.asc kernel.asc; then :; else
  echo "$LABEL: the expression changed nothing -- check it" >&2
  cp base.asc kernel.asc
  exit 2
fi
if ! (cd build && make 2>&1 | grep -E "error:" | head -3); then :; fi
if ! (cd build && make >/dev/null 2>&1); then
  echo "$LABEL: build failed"
  cp base.asc kernel.asc
  (cd build && make >/dev/null 2>&1)
  exit 1
fi

echo "=== $LABEL ==="
bad=0
for id in "${IDS[@]}"; do
  d="$CASES/tp$(printf %02d "$id")_even"
  rm -f "$d/out.bin"
  if ! timeout 150 ./build/driver --dir "$d" --iters 2 >/dev/null 2>&1; then
    printf "  tp%-3s CRASH\n" "$id"; bad=1; continue
  fi
  if ! "$BENCH" verify "$CASES" "$d/out.bin" "$id" even >/dev/null 2>&1; then
    printf "  tp%-3s WRONG\n" "$id"; bad=1; continue
  fi
  out=$("$HERE/profile.sh" "$d" ./build/driver 20 2>&1)
  min=$(echo "$out" | awk 'NR==1{for(i=1;i<=NF;i++) if($i=="min"){print $(i+1); exit}}')
  ctr=$(echo "$out" | awk 'NR==2{printf "cube %s vec %s blocks %s", $10, $12, $2}')
  printf "  tp%-3s %10s us   %s\n" "$id" "${min:--}" "$ctr"
done
cp base.asc kernel.asc
(cd build && make >/dev/null 2>&1)
[ $bad -eq 0 ] || echo "  (correctness broke -- do not keep this)"
