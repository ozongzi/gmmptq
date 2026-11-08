#!/bin/bash
# second calibration batch: judge-protocol regime, nop floor, eviction-size
# sensitivity, and a profiler-configuration control
set -u
cd ~/gmm-rebuild
source $HOME/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
C=cases-b40da251a039
D=~/gmm-cal/prod/build/cal_driver
R=~/gmm-cal
run() { echo "### $*"; python3 ~/gmm-cal/tools/cal.py sweep --cases $C --driver $D "$@"; }

run --modes cold0 --reps 3 --json $R/cold0.json
run --modes fresh,hot,cold --reps 2 --nop --points 6 --json $R/nop.json
for mib in 168 336 1024; do
  run --modes cold --reps 2 --points 1,4,12,14 --evict-mib $mib --json $R/evict$mib.json
done
run --modes fresh,hot,cold --reps 2 --points 3,5,12,14 --profiler harness --json $R/harness_prof.json
run --modes hot-long --reps 2 --points 3,5,6,10,12 --json $R/hotlong.json
echo "### BATCH2 COMPLETE"
