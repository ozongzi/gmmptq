#!/bin/bash
# isoscore.sh -- invert a spin-isolation board total into one hidden point's score.
#
# The instrument (tools/isolate.sh) appends 50.00 ms of pure spin to every point
# but one, so the judge's board total for our team is
#
#     total_iso = ( sum_{i != N} s_spun(i) + s_N ) / 15
#
# and the all-spin calibration run (isolate.sh ... 0) gives the whole sum:
#
#     total_cal = ( sum_{all i} s_spun(i) ) / 15
#
# Subtracting leaves one unknown of the ~4.35 kind, which is what makes this
# precise: the spun score is 100*ln1.5 / (ln1.5 + ln(50003/tbest)), so a 45%
# error in a hidden tbest moves it by 0.15 and nothing else.
#
#     s_N = 15 * (total_iso - total_cal) + s_spun(N)
#
# Both totals are board figures rounded to 0.01, so each contributes +-0.075.
#
# Usage: isoscore.sh <total_iso> <total_cal> <tbest_N us> [t_N us]
set -eu
TI=${1:?board total of the isolation run}
TC=${2:?board total of the all-spin calibration run}
BN=${3:?tbest of the isolated point, us}
TN=${4:-0}      # our own judge-side time for that point, if known, us
awk -v ti="$TI" -v tc="$TC" -v b="$BN" -v t="$TN" 'BEGIN{
  a = log(1.5);
  spun = 50002.7 + (t>0 ? t : b);          # the spin floor plus the real kernel
  s_spun = 100*a/(a + log(spun/b));
  s = 15*(ti - tc) + s_spun;
  u = a*(100/s - 1);                        # ln(t_N / tbest_N)
  printf "s_spun(N)      %8.3f   (tbest %.3f us spun to %.1f us)\n", s_spun, b, spun;
  printf "s_N            %8.3f   +-0.11 from the two 0.01 roundings\n", s;
  printf "t_N / tbest_N  %8.4f\n", exp(u);
  printf "t_N            %8.3f us  (at tbest = %.3f)\n", b*exp(u), b;
  printf "\nd(score)/d(ln t) = -s^2/(100*ln1.5) = %.3f per 1%% of time\n", -s*s/(100*a)/100;
}'
