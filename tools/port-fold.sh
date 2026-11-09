#!/bin/bash
# port-fold.sh <base.asc> <out.asc>
# Replace point 11's two WholeReduceSum levels with the aligned halving Add tree
# (p11add).  Works on any line that still carries the original reduce site; the
# guards below are the whole safety net, so a base that drifted fails loudly.
# Split points round UP to 8 floats: f[50] at byte 200 is not 32-byte aligned and
# faults the vector unit (ACL 507035), which is what forced the few-and-wide shape.
set -eu
B=${1:?base.asc}; O=${2:?out.asc}
tr -d '\r' < "$B" > "$O.tmp"
L1=$(grep -n 'for(uint32_t col=0;col<realPairs;col+=64u)' "$O.tmp" | cut -d: -f1)
[ "$(printf '%s\n' "$L1" | wc -l)" = 1 ] && [ -n "$L1" ] || { echo "reduce site not unique"; exit 1; }
sed -n "$((L1+1))p" "$O.tmp" | grep -q 'WholeReduceSum(scratch\[col/64u\]' || { echo "L1 body moved"; exit 1; }
sed -n "$((L1+3))p" "$O.tmp" | grep -q 'WholeReduceSum(spaced,scratch' || { echo "L2 moved"; exit 1; }
sed -n "$((L1+2))p" "$O.tmp" | grep -q 'PipeBarrier<PIPE_V>();' || { echo "barrier moved"; exit 1; }
grep -q 'Gather(sums,spaced' "$O.tmp" || { echo "gather gone -- spaced is still needed"; exit 1; }
cat > "$O.fold" <<'FOLD'
                {   // Halving fold with 32-byte-aligned split points, then one
                    // reduce over the last <=64.  f[realPairs..pairs) is zero.
                    const BinaryRepeatParams tr{1,1,1,uint8_t(pairs/8u),uint8_t(pairs/8u),uint8_t(pairs/8u)};
                    uint32_t act=pairs;
                    while(act>64u){
                        const uint32_t half=((act+1u)/2u+7u)&~7u, tail=act-half;
                        for(uint32_t c=0;c<tail;c+=64u){
                            const uint64_t mask=tail-c<64u?tail-c:64u;
                            Add(f[c],f[c],f[half+c],mask,uint8_t(nr),tr);
                        }
                        PipeBarrier<PIPE_V>();
                        act=half;
                    }
                    WholeReduceSum(spaced,f,uint64_t(act),uint8_t(nr),8,1,uint8_t(pairs/8u));
                    PipeBarrier<PIPE_V>();
                }
FOLD
{ head -n $((L1-1)) "$O.tmp"; cat "$O.fold"; tail -n +$((L1+5)) "$O.tmp"; } > "$O"
rm -f "$O.tmp" "$O.fold"
echo "fold ported at line $L1: $(wc -l < "$B") -> $(wc -l < "$O") lines"
