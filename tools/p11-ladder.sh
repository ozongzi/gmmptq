#!/bin/bash
# mkp11lad.sh <rung 1..10> -- the CUMULATIVE stage-deletion ladder for point 11.
#
# Why cumulative and not the single deletions we already have: point 11's highest
# pipe is 42% busy, so every single deletion hides behind the 55% that nobody is
# using.  The seven single probes sum to 3.12 us of a 7.38 us duration, which is
# only an upper bound on what the stages cost *together*.  A cumulative ladder
# measures the actual floor.
#
# Safe against dead-code elimination: writes to a LocalTensor are stores to a
# fixed UB address, which the compiler keeps.  Checked empirically -- w11red
# deleted the reduce and vector busy fell 1.36 us, not the 2.6 it would have
# fallen if split/mul/cast/add had gone dead with it.
#
# Rungs (each includes all lower ones):
#   1  - fold + WholeReduceSum          (spaced left garbage, Gather still reads it)
#   2  - the fp32 Add
#   3  - the two int16->fp32 Casts
#   4  - the 8 strided Muls
#   5  - the 3 int8->int16 shifts
#   6  - the 4-op epilogue (Gather/Muls/Mul/Cast)   (out garbage, y store kept)
#   7  - the x and scales DMAs, speculative and in-loop, and their MTE2_V
#   8  - the whole weight path (DMA, 3 shifts, scalar weightScale)
#   9  - the groupList expert scan (first/last become begin/finish)
#  10  - the prologue: CreateVecIndex/Muls and the 5 DataCachePreload
#
# Rung 10 is TPipe + the UB declarations + one y store: the body's floor.
set -eu
cd ~/gmm-rebuild
L=${1:?rung 1..10}
N=lad$L
# base is p11add (the current best), not mkvar.sh's h-m-p11Q
rm -rf "$N"; mkdir -p "$N"
cp p11add/CMakeLists.txt p11add/driver.asc p11add/kernel.asc p11add/judge_shim.h "$N"/
cp p11add/profile.sh p11add/sweep.sh p11add/compare.sh "$N"/ 2>/dev/null || true
K=$N/kernel.asc

chk() { sed -n "$1p" $K | grep -qF "$2" || { echo "GUARD FAIL $1 wanted: $2"; sed -n "$1p" $K; exit 1; }; }
chk  944 'CreateVecIndex(index,int32_t(0),R);'
chk  954 'DataCachePreload(pf,int16_t(0));'
chk  963 'DataCopyPad(xl,xg[uint64_t(begin)*s.k]'
chk  967 'for(uint32_t e=0;e<s.e;++e){'
chk  968 'uint32_t entry=gg.GetValue(e)'
chk  969 'uint32_t first=begin>start?begin:start,last=finish<end?finish:end;'
chk  970 'if(first<last){'
chk  971 'DataCopyPad(wl,wg['
chk  974 'ShiftLeft(weven,wp,int16_t(8),pairs);'
chk  977 'float weightScale;'
chk  984 'DataCopyPad(xl,xg[uint64_t(m)*s.k]'
chk  987 'GvBarrier<HardEvent::MTE2_V>();'
chk  989 'ShiftLeft(even,xp,int16_t(8),nr*pairs);'
chk  995 'Mul(even[col],even[col],weven[col],mask,uint8_t(nr),row);'
chk  999 'Cast(f,even,RoundMode::CAST_NONE,nr*pairs);'
chk 1001 'Add(f,f,other,nr*pairs);'
chk 1002 'Halving fold with 32-byte-aligned'
chk 1015 'WholeReduceSum(spaced,f,uint64_t(act)'
chk 1018 'Gather(sums,spaced,'
chk 1021 'Cast(out,sums,RoundMode::CAST_RINT,nr);'
chk 1022 'DataCopyPad(yg[m],out,'
chk 1026 'start=end;'
echo "guards ok (23)"

# Deletions are applied BOTTOM UP so earlier line numbers stay valid.
d() { sed -i "$1d" $K; }   # d <range>

[ "$L" -ge 9  ] && { d 1026; }                       # start=end;  (scan gone)
[ "$L" -ge 6  ] && { d 1018,1021; }                  # epilogue
[ "$L" -ge 1  ] && { d 1002,1017; }                  # fold + reduce
[ "$L" -ge 2  ] && { d 1001; }                       # Add
[ "$L" -ge 3  ] && { d 999,1000; }                   # Casts
[ "$L" -ge 4  ] && { d 992,998; }                    # Muls
[ "$L" -ge 5  ] && { d 988,991; }                    # shifts
[ "$L" -ge 7  ] && { d 982,987; }                    # x/scales DMA + MTE2_V
[ "$L" -ge 8  ] && { d 971,979; }                    # weight path
[ "$L" -ge 9  ] && {                                 # expert scan -> straight range
  sed -i '967,969c\
{\
uint32_t first=begin,last=finish;' $K
}
[ "$L" -ge 7  ] && { d 961,966; sed -i 's/^bool haveSpec=false;$/bool haveSpec=false;(void)haveSpec;/' $K; }
[ "$L" -ge 10 ] && { d 952,959; d 944,945; }
true

printf "%s: lines=%s  reduce=%s add=%s cast=%s mul=%s split=%s epi=%s xdma=%s wpath=%s scan=%s pre=%s pvec=%s ystore=%s\n" \
  "$N" "$(wc -l < $K)" \
  "$(grep -c 'Halving fold' $K)" "$(grep -c 'Add(f,f,other' $K)" "$(grep -c 'Cast(f,even' $K)" \
  "$(grep -c 'Mul(even\[col\]' $K)" "$(grep -c 'ShiftLeft(even,xp' $K)" "$(grep -c 'Gather(sums,spaced' $K)" \
  "$(grep -c 'DataCopyPad(xl,xg' $K)" "$(grep -c 'DataCopyPad(wl,wg' $K)" "$(grep -c 'gg.GetValue(e)' $K)" \
  "$(grep -c 'DataCachePreload' $K)" "$(grep -c 'CreateVecIndex(index' $K)" "$(grep -c 'DataCopyPad(yg\[m\]' $K)"
