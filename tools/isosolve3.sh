#!/bin/bash
# isosolve3.sh -- tbest of a hidden point, with no calibration constant at all.
#
# Two isolations of the SAME point with two different known spins on the point
# itself (isolate.sh's 5th argument, 20 ns ticks) both pay the same unknown
# spin-kernel launch cost h, so h cancels and tbest comes out in closed form:
#
#     u_i = a(100/s_i - 1) = ln( (t + h + D_i*0.02) / b )
#     (D2 - D1)*0.02 = b * ( exp(u2) - exp(u1) )
#  => b = (D2 - D1) * 0.02 / ( exp(u2) - exp(u1) )
#
# The D = 0 run (isolate.sh ... <point> <out> with no 5th argument) launches no
# spin kernel at all, so it pays no h and gives our own time directly:
#
#     t = b * exp(u0)        and then   h = b*exp(u1) - t - D1*0.02
#
# h is a property of the instrument, so one point measured this way calibrates it
# for all the others, which then need only one spun run each (isosolve.sh).
#
# Usage: isosolve3.sh <s1> <ticks1> <s2> <ticks2> [s0]
set -eu
S1=${1:?score of the first spun run}
D1=${2:?ticks of the first spun run}
S2=${3:?score of the second spun run}
D2=${4:?ticks of the second spun run}
S0=${5:-0}
awk -v s1="$S1" -v d1="$D1" -v s2="$S2" -v d2="$D2" -v s0="$S0" 'BEGIN{
  a = log(1.5);
  if (d2 == d1) { print "the two spun runs need different tick counts"; exit 1 }
  u1 = a*(100/s1 - 1); u2 = a*(100/s2 - 1);
  b  = (d2 - d1)*0.02 / (exp(u2) - exp(u1));
  if (b <= 0) { print "negative tbest: the scores are the wrong way round for these ticks"; exit 1 }
  printf "their tbest     %8.3f us      (from the two spun runs alone, no h)\n", b;
  printf "  our t + h at D1 %6.3f us, at D2 %6.3f us\n", b*exp(u1), b*exp(u2);
  if (s0 > 0) {
    t = b*exp(a*(100/s0 - 1));
    h = b*exp(u1) - t - d1*0.02;
    printf "our t           %8.3f us      (from the unspun isolation, s0 = %s)\n", t, s0;
    printf "t / tbest       %8.4f\n", t/b;
    printf "instrument h    %8.3f us      (reusable: one spun run per other point)\n", h;
  }
}'
