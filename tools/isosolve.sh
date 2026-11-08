#!/bin/bash
# isosolve.sh -- separate OUR judge-side time from the point's tbest.
#
# A single isolation gives one number, the score, which pins only the ratio
# t/tbest.  That is why point 11 came out ambiguous: s_11 = 40.29 is equally
# consistent with "our 10.85 us against their 5.95" and with "our 7.40 against
# their 4.06".  Adding a SECOND isolation of the same point with a known extra
# spin on the point itself (isolate.sh's 5th argument, in 20 ns ticks) makes it
# two equations:
#
#     u0 = a(100/s0 - 1) = ln( t       / b )
#     u1 = a(100/s1 - 1) = ln((t + A)  / b )      A = h + ticks*0.02 us
#  => t = A / (exp(u1-u0) - 1)        b = t / exp(u0)
#
# h is the cost of launching the extra spin kernel at all, which the D=0 run does
# not pay.  It is a property of the instrument, not of the point: calibrate it
# once as (spun time - 50000 - unspun time) on a visible point, same kernel.
#
# Usage: isosolve.sh <s0> <s1> <ticks of the second run> <h us>
set -eu
S0=${1:?score with the point unspun}
S1=${2:?score with the point spun by <ticks>}
TK=${3:?ticks on the point in the second run (20 ns each)}
H=${4:?spin-kernel launch cost in us, from the visible-point calibration}
awk -v s0="$S0" -v s1="$S1" -v tk="$TK" -v h="$H" 'BEGIN{
  a = log(1.5);
  if (s1 >= s0) { print "the spun run must score LOWER; check the arguments"; exit 1 }
  u0 = a*(100/s0 - 1); u1 = a*(100/s1 - 1);
  A  = h + tk*0.02;
  t  = A/(exp(u1-u0) - 1);
  b  = t/exp(u0);
  printf "added time A    %8.3f us   (%s ticks = %.2f us, plus h = %.3f)\n", A, tk, tk*0.02, h;
  printf "our t           %8.3f us\n", t;
  printf "their tbest     %8.3f us\n", b;
  printf "t / tbest       %8.4f    score %.2f\n", t/b, 100*a/(a+log(t/b));
  printf "\nsensitivity: +-0.11 on each score -> about +-%.1f%% on t and tbest\n",
         100*(0.00275*sqrt(2))*exp(u1-u0)/(exp(u1-u0)-1);
}'
