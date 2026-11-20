# coding: utf-8
"""
Best run (valid Recall@20) among the taps of one or more grid experiments, as --key=value overrides.
Reads taps/<Model>-<ds>-<exp>/meta.json or taps/<Model>-<ds>-<exp>/combo*/meta.json (written by --save_taps=True).

    python tools/taps_best.py LightGCNJEPA baby e4u                     # -> --mask_mode=uniform --mask_ratio=0.05 ...
    python tools/taps_best.py LightGCNJEPA baby e4u e4h e4t --key jepa_weight      # -> 0.1
    python tools/taps_best.py SGL baby e3_sgled --show                   # table of every run
Seeds are not printed (pass --seed yourself). Exit code 1 if no taps exist.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def runs(model, dataset, exps, taps_dir=os.path.join(ROOT, 'taps'), seed=999):
    out = []
    for e in exps:
        base = os.path.join(taps_dir, '{}-{}-{}'.format(model, dataset, e))
        for f in [os.path.join(base, 'meta.json')] + sorted(glob.glob(os.path.join(base, 'combo*', 'meta.json'))):
            if os.path.isfile(f):
                m = json.load(open(f))
                h = m.get('hyper_parameters', {})
                if seed is not None and h.get('seed') not in (None, seed):
                    continue
                out.append({'exp': e, 'dir': os.path.dirname(f), 'hyper': h, 'valid': m['valid'].get('recall@20', -1),
                            'test': m['test'].get('recall@20', -1), 'epoch': m.get('epoch')})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('dataset')
    ap.add_argument('exps', nargs='+')
    ap.add_argument('--key', default=None, help='print only this hyper-parameter')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--seed', type=int, default=999)
    a = ap.parse_args()
    rs = runs(a.model, a.dataset, a.exps, seed=a.seed)
    if not rs:
        sys.exit(1)
    if a.show:
        for r in sorted(rs, key=lambda r: -r['valid']):
            print('{:.4f} {:.4f} ep{:>4} {} {}'.format(r['valid'], r['test'], r['epoch'], r['exp'],
                                                       {k: v for k, v in r['hyper'].items() if k != 'seed'}))
        return
    best = max(rs, key=lambda r: r['valid'])
    h = {k: v for k, v in best['hyper'].items() if k != 'seed'}
    if a.key:
        print(h[a.key])
    else:
        print(' '.join('--{}={}'.format(k, v) for k, v in h.items()))


if __name__ == '__main__':
    main()
