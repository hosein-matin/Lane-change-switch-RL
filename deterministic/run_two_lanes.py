#!/usr/bin/env python3
"""Parallel, resumable runner for the revision experiments (commitment model, critic without leader
parameters, 15 seeds, model mismatch, conformal coverage).

    python run_all.py --cores 16            # run everything that is not done yet
    python run_all.py --status              # show progress and stop
    python run_all.py --smoke --cores 4     # 10-minute end-to-end test on tiny settings (separate folder)

Every job writes its log to <results>/logs/<job>.log and, on success, a marker <results>/.done/<job>.
Re-running the same command after an interruption (crash, reboot, Ctrl-C) skips finished jobs and restarts
unfinished ones; the evaluation scripts additionally resume from their last saved method. A progress summary
is rewritten every minute to <results>/status.txt.
"""
import argparse, os, subprocess, sys, time, json, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, 'code')
SEEDS_ALL = list(range(15))


def build_jobs(R, smoke=False, seeds=SEEDS_ALL):
    """Primary configuration of the revision: commitment model (one input certified over the whole lane
    change in both lanes) and a critic without the leaders' disturbance parameters. Name -> job."""
    it, ep, lab = ('3', '96', '64') if smoke else ('500', '6400', '4096')
    J = {}
    base_env = dict(DUAL_LANE='1', DUAL_MODE='hold', CRITIC_NODIST='1')
    cert = dict(ROUNDS='1', N_DAG='64', N_CAL='20', N_MIX='20') if smoke else dict(ROUNDS='2', N_DAG='2048', N_CAL='500', N_MIX='500')

    def job(name, cmd, deps=(), env=None, prio=0):
        e = dict(base_env); e.update(env or {})
        J[name] = dict(cmd=[sys.executable] + cmd, deps=list(deps), env=e, prio=prio)

    for s in seeds:
        S = str(s)
        job(f'ppo_s{s}', ['ppo.py', S, it, f'{R}/ppo_s{s}'], prio=100)
        act = f'{R}/ppo_s{s}/actor.pkl'
        job(f'dqn_s{s}', ['dqn.py', S, act, ep, f'{R}/dqn_s{s}'], [f'ppo_s{s}'], prio=90)
        for c0 in [5, 7, 10, 14, 20]:
            job(f'dqn_c0{c0}_s{s}', ['dqn.py', S, act, ep, f'{R}/dqn_c0{c0}_s{s}'], [f'ppo_s{s}'], {'C0_TRAIN': str(c0)}, prio=85)
        job(f'eval_s{s}', ['evaluate.py', S], [f'dqn_s{s}'], prio=80)
        job(f'eval2_s{s}', ['evaluate2.py', S], [f'eval_s{s}'], prio=80)
        job(f'eval3_s{s}', ['evaluate3.py', S, 'forecast'], [f'eval2_s{s}'], prio=50)
        job(f'eval6_s{s}', ['evaluate6.py', S], [f'ppo_s{s}'], prio=30)
        job(f'select_c0_s{s}', ['select_c0.py', S], [f'dqn_s{s}'] + [f'dqn_c0{c}_s{s}' for c in [5, 7, 10, 14, 20]], prio=75)
        for v in ['c05', 'c07', 'c010', 'c014', 'c020', 'best']:
            dep = f'select_c0_s{s}' if v == 'best' else f'dqn_{v}_s{s}'
            job(f'eval5_{v}_s{s}', ['evaluate5.py', S, v], [dep, f'eval2_s{s}'], prio=40)
        job(f'amort_best_s{s}', ['amortize.py', S, 'best'], [f'select_c0_s{s}'], {'N_LABEL_EP': lab}, prio=70)
        job(f'evalmis_s{s}', ['evaluate_mis.py', S, 'best'], [f'amort_best_s{s}', f'eval2_s{s}'], prio=60)
        # round C: on-policy relabelling of the student + episode-level conformal certificate (+ mismatch check)
        job(f'cert2_s{s}', ['cert2.py', S, 'best'], [f'amort_best_s{s}'], cert, prio=55)
        if s < 5:
            job(f'sweep_best_s{s}', ['amortize_sweep.py', S, 'best'], [f'amort_best_s{s}'], prio=20)
    job('timing', ['timing.py', str(seeds[0]), 'best'], [f'amort_best_s{seeds[0]}'], prio=10)
    for name in J:
        J[name]['env'].setdefault('RESULTS', R)
        if smoke:
            J[name]['env'].update(N_TEST='4', N_VAL='12')
    return J


def done_dir(R):
    d = os.path.join(R, '.done'); os.makedirs(d, exist_ok=True); return d


def is_done(R, name):
    return os.path.exists(os.path.join(done_dir(R), name))


def status(R, J, running=None, failed=None):
    n = len(J); d = sum(is_done(R, k) for k in J)
    lines = [time.strftime('%Y-%m-%d %H:%M:%S'), f'done {d}/{n}']
    if running:
        lines.append('running: ' + ', '.join(f'{k} ({(time.time()-t)/60:.0f} min)' for k, (_, t, _) in running.items()))
    if failed:
        lines.append('FAILED (see logs/<job>.log): ' + ', '.join(sorted(failed)))
    blocked = [k for k in J if not is_done(R, k) and failed and any(x in failed for x in J[k]['deps'])]
    if blocked:
        lines.append(f'blocked by failures: {len(blocked)} jobs')
    txt = '\n'.join(lines)
    open(os.path.join(R, 'status.txt'), 'w').write(txt + '\n')
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cores', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--results', default=os.path.join(HERE, 'results_cn'))
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--only', default=None, help='run only jobs whose name contains this text (and their dependencies)')
    a = ap.parse_args()
    R = os.path.abspath(a.results)
    seeds = SEEDS_ALL
    if a.smoke:
        R = os.path.join(HERE, 'smoke_cn'); seeds = [7]
        os.makedirs(R, exist_ok=True)
    J = build_jobs(R, a.smoke, seeds)
    if a.only:
        need = set(k for k in J if a.only in k)
        stack = list(need)
        while stack:
            for d in J[stack.pop()]['deps']:
                if d not in need:
                    need.add(d); stack.append(d)
        J = {k: v for k, v in J.items() if k in need}
    for d in [R, os.path.join(R, 'logs')]:
        os.makedirs(d, exist_ok=True)
    if a.status:
        print(status(R, J)); return
    print(f'{len(J)} jobs, {sum(is_done(R, k) for k in J)} already done, {a.cores} parallel workers; results in {R}', flush=True)
    running, failed = {}, set()
    last = 0
    try:
        while True:
            for k, (p, t0, lf) in list(running.items()):
                rc = p.poll()
                if rc is None:
                    continue
                lf.close(); del running[k]
                if rc == 0:
                    open(os.path.join(done_dir(R), k), 'w').write(time.strftime('%Y-%m-%d %H:%M:%S'))
                    print(f'[{time.strftime("%H:%M:%S")}] done   {k} ({(time.time()-t0)/60:.1f} min)', flush=True)
                else:
                    failed.add(k)
                    print(f'[{time.strftime("%H:%M:%S")}] FAILED {k} (exit {rc}), see {R}/logs/{k}.log', flush=True)
            todo = [k for k in J if not is_done(R, k) and k not in running and k not in failed]
            if not todo and not running:
                break
            ready = [k for k in todo if all(is_done(R, d) for d in J[k]['deps'])]
            ready.sort(key=lambda k: -J[k]['prio'])
            while ready and len(running) < a.cores:
                k = ready.pop(0)
                env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           PYTHONUNBUFFERED='1', **J[k]['env'])
                lf = open(os.path.join(R, 'logs', f'{k}.log'), 'a')
                lf.write(f'\n===== start {time.strftime("%Y-%m-%d %H:%M:%S")}\n'); lf.flush()
                p = subprocess.Popen(J[k]['cmd'], cwd=CODE, env=env, stdout=lf, stderr=subprocess.STDOUT)
                running[k] = (p, time.time(), lf)
                print(f'[{time.strftime("%H:%M:%S")}] start  {k}', flush=True)
            if not running and not ready:
                break   # everything left is blocked by a failure
            if time.time() - last > 60:
                status(R, J, running, failed); last = time.time()
            time.sleep(2)
    except KeyboardInterrupt:
        print('\nInterrupted: stopping running jobs (they will restart next time).', flush=True)
        for k, (p, _, lf) in running.items():
            p.terminate()
        for k, (p, _, lf) in running.items():
            try:
                p.wait(10)
            except Exception:
                p.kill()
            lf.close()
    print(status(R, J, {}, failed), flush=True)
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
