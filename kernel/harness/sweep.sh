#!/bin/bash
# Run one kernel over every gmmbench case/split and report correctness.
#
#   sweep.sh <cases dir> <driver> <gmmbench> [iters] [cases...] 
#
# Correctness only -- timing comes from profile.sh, because host wall time
# around a launch is all ACL overhead at these kernel sizes.
set -u
CASES=${1:?cases dir}
DRIVER=${2:?driver}
BENCH=${3:?gmmbench}
ITERS=${4:-1}
# A kernel that deadlocks must fail the sweep, not wedge the whole run.
TIMEOUT=${GMM_TIMEOUT:-120}
shift 4 2>/dev/null || shift $# 
WANT=("$@")

pass=0; fail=0; crash=0
printf "%-6s %-7s %-9s %-12s %-12s %s\n" case split verdict max_rel max_abs note
printf '%s\n' "----------------------------------------------------------------------------"
for dir in "$CASES"/tp*/; do
  name=$(basename "$dir")
  id=$(echo "$name" | sed 's/^tp0*\([0-9]*\)_.*/\1/')
  split=$(echo "$name" | sed 's/^tp[0-9]*_//')
  if [ ${#WANT[@]} -gt 0 ]; then
    hit=0
    for w in "${WANT[@]}"; do [ "$w" = "$id" ] && hit=1; done
    [ $hit -eq 0 ] && continue
  fi
  rm -f "$dir/out.bin"
  if ! timeout "$TIMEOUT" "$DRIVER" --dir "$dir" --iters "$ITERS" >/dev/null 2>"$dir/driver.err"; then
    printf "%-6s %-7s %-9s %-12s %-12s %s\n" "tp$id" "$split" CRASH - - "$(head -1 "$dir/driver.err")"
    crash=$((crash+1)); continue
  fi
  out=$("$BENCH" verify "$CASES" "$dir/out.bin" "$id" "$split" 2>&1)
  verdict=$(echo "$out" | grep -o 'PASS\|FAIL' | head -1)
  rel=$(echo "$out" | awk '/max 相对误差/{print $3}')
  abs=$(echo "$out" | awk '/max 绝对误差/{print $3}')
  nfail=$(echo "$out" | awk '/失败\(宽松读法\)/{print $2}')
  strict=$(echo "$out" | awk '/失败\(严格读法\)/{print $2}')
  note=""
  [ "${verdict:-FAIL}" = "FAIL" ] && note="$nfail 个元素超容差"
  [ "${verdict:-FAIL}" = "PASS" ] && [ "${strict:-0}" != "0" ] && note="严格读法 $strict 个"
  printf "%-6s %-7s %-9s %-12s %-12s %s\n" "tp$id" "$split" "${verdict:-?}" "${rel:--}" "${abs:--}" "$note"
  if [ "${verdict:-FAIL}" = "PASS" ]; then pass=$((pass+1)); else fail=$((fail+1)); fi
done
printf '%s\n' "----------------------------------------------------------------------------"
echo "pass $pass  fail $fail  crash $crash"
[ $fail -eq 0 ] && [ $crash -eq 0 ]
