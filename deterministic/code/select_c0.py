"""Select the training-time switching cost per seed on the critic's checkpoint-validation score (scored with c0 = 2).
Usage: python select_c0.py SEED   -> copies the chosen critic to RESULTS/dqn_best_s{SEED}/ and writes
RESULTS/c0_select_s{SEED}.json with all validation scores.
"""
import sys, os, json, shutil
R0 = os.environ.get('RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results'))
TAGS = {'': 2, 'c05_': 5, 'c07_': 7, 'c010_': 10, 'c014_': 14, 'c020_': 20}


def main(seed):
    sc = {}
    for tag, c0 in TAGS.items():
        p = f'{R0}/dqn_{tag}s{seed}/dqn_log.json'
        sc[c0] = json.load(open(p))['best_val']
    best = max(sc, key=sc.get)
    tag = [t for t, v in TAGS.items() if v == best][0]
    dst = f'{R0}/dqn_best_s{seed}'
    if os.path.islink(dst) or os.path.isfile(dst):
        os.remove(dst)
    os.makedirs(dst, exist_ok=True)
    for fn in ['q.pkl', 'dqn_log.json']:
        shutil.copy(f'{R0}/dqn_{tag}s{seed}/{fn}', f'{dst}/{fn}')
    json.dump(dict(seed=seed, val=sc, best_c0=best), open(f'{R0}/c0_select_s{seed}.json', 'w'), indent=1)
    print('seed', seed, 'validation', sc, '-> c0 =', best, flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
