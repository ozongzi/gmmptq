#!/usr/bin/env python3
"""Cold/hot/fresh msprof timing for the 15 points, and ABBA A/B on the cold arm.

Runs on the device container. Three regimes, one driver (tools/cal_driver.asc):

  fresh  --prime-ms 0   --warmup 0  --samples 35  --cache-mode hot
         exactly what `harness/profile.sh <dir> <driver> 35` measures, which is
         how shape.md §4's 15 numbers were taken.
  hot    --prime-ms 600 --warmup 30 --samples 20  --cache-mode hot
         caches, ICache and clocks settled; what search/*_fast used.
  cold   --prime-ms 600 --warmup 30 --samples 20  --cache-mode evict
         same, plus a 512 MiB >L2 stream run to completion before every launch
         (the eviction kernel and size of experiments/oj_known_config).

Only the cold number is the one to rank candidates on; see shape.md §4/§5.

  cal.py build --kernel <kernel.asc> --out <dir> [--harness <dir>]
  cal.py sweep --cases <dir> --driver <path> [--points 1,2,...]
               [--modes fresh,hot,cold] [--reps 1] [--nop] [--json out.json]
  cal.py ab    --cases <dir> --a <driver> --b <driver> [--mode cold]
               [--reps 12] [--points ...] [--labels A,B] [--json out.json]

`ab` interleaves ABBA/BAAB by rep parity, 12 reps per arm by default, and
reports the paired per-rep ratio (that is the statistic shape.md §5 asks for).
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import statistics as st
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))

MODES = {
    # name: (prime_ms, warmup, samples, cache_mode)
    'fresh': (0, 0, 35, 'hot'),
    'hot': (600, 30, 20, 'hot'),
    'cold': (600, 30, 20, 'evict'),
    # diagnostics
    'cold0': (0, 0, 35, 'evict'),      # fresh process *and* evicted
    'hot-long': (2000, 50, 20, 'hot'),  # does 600 ms of priming suffice?
}

PROFILER_CLI = [
    # experiments/oj_known_config/profiler_cli.json -- the configuration whose
    # 50 inherited profiler fields were confirmed identical to the judge's.
    '--task-time=on', '--ai-core=on', '--aic-mode=task-based',
    '--aic-metrics=PipeUtilization', '--aic-freq=100', '--runtime-api=on',
    '--ascendcl=on', '--task-memory=off', '--ge-api=off', '--msproftx=off',
    '--export=on',
]

HARNESS_CLI = [
    # harness/profile.sh's configuration, kept so the two can be compared.
    '--aic-metrics=ArithmeticUtilization', '--ai-core=on',
]


def build(args):
    """Compile cal_driver.asc against one kernel.asc into its own build tree."""
    out = os.path.abspath(args.out)
    harness = os.path.abspath(args.harness)
    os.makedirs(out, exist_ok=True)
    for name in ('judge_shim.h',):
        shutil.copyfile(os.path.join(harness, name), os.path.join(out, name))
    shutil.copyfile(os.path.abspath(args.kernel), os.path.join(out, 'kernel.asc'))
    shutil.copyfile(os.path.join(HERE, 'cal_driver.asc'), os.path.join(out, 'cal_driver.asc'))
    with open(os.path.join(out, 'CMakeLists.txt'), 'w') as f:
        f.write('cmake_minimum_required(VERSION 3.16)\n'
                'find_package(ASC REQUIRED)\n'
                'project(gmm_cal LANGUAGES ASC CXX)\n'
                'add_executable(cal_driver cal_driver.asc)\n'
                'target_include_directories(cal_driver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})\n'
                'target_link_libraries(cal_driver PRIVATE tiling_api register platform m dl)\n'
                'target_compile_options(cal_driver PRIVATE\n'
                '    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)\n')
    bd = os.path.join(out, 'build')
    os.makedirs(bd, exist_ok=True)
    log = open(os.path.join(out, 'build.log'), 'w')
    for cmd in (['cmake', '..'], ['make', '-j8']):
        rc = subprocess.run(cmd, cwd=bd, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            log.close()
            sys.exit('build failed at %s; see %s/build.log' % (cmd[0], out))
    log.close()
    binary = os.path.join(bd, 'cal_driver')
    if not os.path.exists(binary):
        sys.exit('no cal_driver produced in ' + bd)
    print(binary)


def read_op_summary(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def measure(driver, case_dir, mode, cores, evict_mib, kernel, profiler):
    """One msprof capture. Returns the kept per-launch durations in order."""
    prime, warm, samples, cache = MODES[mode]
    prof = tempfile.mkdtemp(prefix='cal-')
    try:
        app = ('%s --dir %s --prime-ms %d --warmup %d --samples %d --cache-mode %s '
               '--evict-mib %d --kernel %s' %
               (driver, case_dir, prime, warm, samples, cache, evict_mib, kernel))
        if cores:
            app += ' --cores %d' % cores
        cmd = ['msprof', '--application=' + app, '--output=' + prof] + profiler
        log = os.path.join(prof, 'msprof.log')
        with open(log, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1800)
        files = glob.glob(os.path.join(prof, '**', 'op_summary*.csv'), recursive=True)
        if not files:
            tail = open(log).read()[-800:]
            raise RuntimeError('no op_summary for %s/%s:\n%s' % (case_dir, mode, tail))
        rows = read_op_summary(sorted(files)[0])
        # The driver tells us how many launches it actually made; without that
        # the kept samples cannot be located by position.
        launches = None
        for line in open(log):
            m = re.search(r'CAL .*launches=(\d+)', line)
            if m:
                launches = int(m.group(1))
        if launches is None:
            raise RuntimeError('driver printed no CAL line for %s/%s' % (case_dir, mode))
        dur = 'Task Duration(us)'
        target, evicts = [], 0
        for r in rows:
            name = (r.get('Op Name') or '').strip()
            if 'Evict' in name:
                evicts += 1
                continue
            if 'preload_stack' in name:
                continue
            if not (r.get(dur) or '').strip():
                continue
            target.append((name, float(r[dur])))
        if cache == 'evict' and evicts < launches:
            raise RuntimeError('%s/%s: %d evictions for %d launches' %
                               (case_dir, mode, evicts, launches))
        if len(target) % launches:
            raise RuntimeError('%s/%s: %d target tasks is not a multiple of %d launches' %
                               (case_dir, mode, len(target), launches))
        per = len(target) // launches          # kernels per call (usually 1)
        totals = [sum(d for _, d in target[i * per:(i + 1) * per])
                  for i in range(launches)]
        names = sorted({n for n, _ in target})
        return dict(mode=mode, launches=launches, per_call_tasks=per, names=names,
                    evictions=evicts, first=totals[0], samples=totals[-samples:],
                    median=st.median(totals[-samples:]),
                    mean=st.mean(totals[-samples:]),
                    min=min(totals[-samples:]), max=max(totals[-samples:]))
    finally:
        shutil.rmtree(prof, ignore_errors=True)


def sweep(args):
    points = [int(p) for p in args.points.split(',')] if args.points else list(range(1, 16))
    modes = args.modes.split(',')
    profiler = HARNESS_CLI if args.profiler == 'harness' else PROFILER_CLI
    out = []
    for rep in range(args.reps):
        for p in points:
            case = os.path.join(args.cases, 'tp%02d_even' % p)
            for mode in modes:
                t0 = time.time()
                r = measure(args.driver, case, mode, args.cores, args.evict_mib,
                            'nop' if args.nop else 'prod', profiler)
                r.update(point=p, rep=rep, wall_s=round(time.time() - t0, 1))
                out.append(r)
                print('P%-3d %-8s rep%-2d median %8.3f  min %8.3f  max %8.3f  '
                      'first %8.3f  (n=%d, %ds)' %
                      (p, mode, rep, r['median'], r['min'], r['max'], r['first'],
                       len(r['samples']), r['wall_s']), flush=True)
    print()
    print('%-5s %s' % ('point', '  '.join('%10s' % m for m in modes)))
    for p in points:
        cells = []
        for mode in modes:
            v = [r['median'] for r in out if r['point'] == p and r['mode'] == mode]
            cells.append('%10.3f' % st.median(v) if v else '%10s' % '-')
        print('%-5d %s' % (p, '  '.join(cells)))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=1)
        print('\nwrote', args.json)


def ab(args):
    points = [int(p) for p in args.points.split(',')] if args.points else [1, 2, 4, 9, 12, 14, 15]
    la, lb = args.labels.split(',')
    profiler = HARNESS_CLI if args.profiler == 'harness' else PROFILER_CLI
    rows = []
    for p in points:
        case = os.path.join(args.cases, 'tp%02d_even' % p)
        for rep in range(args.reps):
            # ABBA: odd reps run A first, even reps run B first.
            order = [(la, args.a), (lb, args.b)] if rep % 2 == 0 else [(lb, args.b), (la, args.a)]
            got = {}
            for label, drv in order:
                r = measure(drv, case, args.mode, args.cores, args.evict_mib, 'prod', profiler)
                got[label] = r['median']
                rows.append(dict(point=p, rep=rep, arm=label, mode=args.mode,
                                 median=r['median'], samples=r['samples']))
            print('P%-3d rep%-3d %s %8.3f   %s %8.3f   ratio %.4f' %
                  (p, rep, la, got[la], lb, got[lb], got[lb] / got[la]), flush=True)
    print()
    print('%-5s %10s %10s %8s %8s %6s' % ('point', la, lb, 'B/A', 'wins', 'n'))
    for p in points:
        a = [r['median'] for r in rows if r['point'] == p and r['arm'] == la]
        b = [r['median'] for r in rows if r['point'] == p and r['arm'] == lb]
        ratios = [y / x for x, y in zip(a, b)]
        wins = sum(1 for r in ratios if r < 1)
        print('%-5d %10.3f %10.3f %8.4f %4d/%-3d %6d' %
              (p, st.median(a), st.median(b), st.median(ratios), wins, len(ratios), len(a)))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(rows, f, indent=1)
        print('\nwrote', args.json)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build')
    b.add_argument('--kernel', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--harness', default=os.path.expanduser('~/gmm-rebuild/harness'))
    b.set_defaults(func=build)

    for name, fn in (('sweep', sweep), ('ab', ab)):
        s = sub.add_parser(name)
        s.add_argument('--cases', required=True)
        s.add_argument('--points', default='')
        s.add_argument('--cores', type=int, default=0)
        s.add_argument('--evict-mib', type=int, default=512)
        s.add_argument('--profiler', choices=['oj', 'harness'], default='oj')
        s.add_argument('--json', default='')
        if name == 'sweep':
            s.add_argument('--driver', required=True)
            s.add_argument('--modes', default='fresh,hot,cold')
            s.add_argument('--reps', type=int, default=1)
            s.add_argument('--nop', action='store_true')
        else:
            s.add_argument('--a', required=True)
            s.add_argument('--b', required=True)
            s.add_argument('--mode', default='cold', choices=sorted(MODES))
            s.add_argument('--reps', type=int, default=12)
            s.add_argument('--labels', default='A,B')
        s.set_defaults(func=fn)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
