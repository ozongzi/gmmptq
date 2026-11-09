#!/bin/bash
# strip-case9.sh <base.asc> <out.asc>
# Remove the point-9 exact-shape gate and the empty kernel it launches:
#   if (E == 4 && M == 513 && K == 383 && N == 257 && groupListType == 0 && bf)
#       Case9EmptyKernel<<<1, nullptr, stream>>>();
# One dedicated implementation matched to one published shape, computing nothing --
# what rule 2.8 forbids, sitting in the submission rule 3 reviews by hand.  It also
# inflates that line's board total by about +6, since point 9 then scores ~100
# instead of ~10, which is most of the gap between 8d626cde's 67.55 and p11add's 60.89.
set -eu
B=${1:?base.asc}; O=${2:?out.asc}
tr -d '\r' < "$B" > "$O"

G=$(grep -n 'E == 4 && M == 513 && K == 383 && N == 257' "$O" | cut -d: -f1)
[ "$(printf '%s\n' "$G" | wc -l)" = 1 ] && [ -n "$G" ] || { echo "gate not unique"; exit 1; }
sed -n "$((G+1))p" "$O" | grep -q 'Case9EmptyKernel<<<'   || { echo "gate body moved"; exit 1; }
sed -n "$((G+2))p" "$O" | grep -qx '        return;'      || { echo "gate shape changed"; exit 1; }
sed -n "$((G+3))p" "$O" | grep -qx '    }'                || { echo "gate shape changed"; exit 1; }
sed -i "${G},$((G+3))d" "$O"

K=$(grep -n '^__global__ __aicore__ void Case9EmptyKernel() {' "$O" | cut -d: -f1)
[ -n "$K" ] || { echo "empty kernel vanished"; exit 1; }
sed -n "$((K+1))p" "$O" | grep -q 'KERNEL_TASK_TYPE_DEFAULT' || { echo "body changed"; exit 1; }
sed -n "$((K+2))p" "$O" | grep -qx '    return;'             || { echo "body changed"; exit 1; }
sed -n "$((K+3))p" "$O" | grep -qx '}'                       || { echo "body changed"; exit 1; }
FROM=$K
sed -n "$((K-1))p" "$O" | grep -q '^// Case 9' && FROM=$((K-1))
sed -i "${FROM},$((K+3))d" "$O"

! grep -q 'Case9EmptyKernel' "$O" || { echo "still referenced"; exit 1; }
echo "stripped: $(wc -l < "$B") -> $(wc -l < "$O") lines"
