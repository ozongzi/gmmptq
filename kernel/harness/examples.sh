#!/bin/bash
# Run the kernel against problem.md §6's own worked examples.
#
#   examples.sh <cases dir> <driver> <gmmbench>
#
# These three are the only ground truth not produced by our own code: §6 states
# the inputs and the exact expected output. A kernel that passes the generated
# cases but fails one of these has a bug the generator does not reach.
set -u
CASES=${1:?cases dir}
DRIVER=${2:?driver}
BENCH=${3:?gmmbench}

"$BENCH" examples "$CASES" >/dev/null || exit 1
fail=0
for name in two-experts empty-head-mid-tail negative-and-zero; do
  dir="$CASES/ex_$name"
  rm -f "$dir/out.bin"
  if ! "$DRIVER" --dir "$dir" --iters 1 >/dev/null 2>"$dir/driver.err"; then
    printf "%-24s CRASH  %s\n" "$name" "$(head -1 "$dir/driver.err")"
    fail=1
    continue
  fi
  "$BENCH" exverify "$CASES" "$name" "$dir/out.bin" || fail=1
done
exit $fail
