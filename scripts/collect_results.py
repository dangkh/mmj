"""
Summarise the logs in src/log/ (one row per experiment; the latest log per name).

    python scripts/collect_results.py                 # baby
    python scripts/collect_results.py -d sports
    python scripts/collect_results.py --combos        # also every hyper-parameter combination
    python scripts/collect_results.py --group_seeds   # mean/std over names ending in _s<seed>

The reported combination is the one selected on VALID (mmj patch of quick_start).
Writes results_<dataset>.csv in the repo root.
"""
import argparse
import csv
import glob
import os
import re
from datetime import datetime

TIME_RE = r'[A-Z][a-z]{2}-\d{2}-\d{4}-\d{2}-\d{2}-\d{2}'
METRIC_RE = re.compile(r'([A-Za-z]+@\d+):\s*([0-9.]+)')
SHOW = ['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20']

# published Baby numbers (R@10, R@20, N@10, N@20), see scripts/experiments_baby_repro.txt
PAPER = {
    'baby': {
        'LightGCN': (0.0479, 0.0754, 0.0257, 0.0328),
        'LATTICE': (0.0547, 0.0850, 0.0292, 0.0370),
        'FREEDOM': (0.0627, 0.0992, 0.0330, 0.0424),
        'MGCN': (0.0620, 0.0964, 0.0339, 0.0427),
        'DRAGON': (0.0662, 0.1021, 0.0345, 0.0435),
    },
}


def metrics(s):
    return {k.lower(): float(v) for k, v in METRIC_RE.findall(s)}


def parse_log(path):
    text = open(path, encoding='utf-8', errors='ignore').read()
    row = {'status': 'running'}
    final = text.rfind('BEST ████')
    current = text.rfind('Current BEST')
    block = text[final:] if final >= 0 else (text[current:] if current >= 0 else None)
    if final >= 0:
        row['status'] = 'done'
    elif 'Traceback' in text or 'Training loss is nan' in text:
        row['status'] = 'failed'
    if block is not None:
        p = re.search(r'Parameters:\s*(.*?=\(.*?\))', block, re.S)
        v = re.search(r'Valid:\s*(.*)', block)
        t = re.search(r'Test:\s*(.*)', block)
        if p:
            row['params'] = ' '.join(p.group(1).split())
        if v:
            row.update({'valid_' + k: x for k, x in metrics(v.group(1)).items()})
        if t:
            row.update({'test_' + k: x for k, x in metrics(t.group(1)).items()})
    epochs = re.findall(r'epoch (\d+) training', text)
    row['epochs'] = int(epochs[-1]) + 1 if epochs else 0
    best_ep = re.findall(r'best eval result in epoch (\d+)', text)
    row['best_epoch'] = int(best_ep[-1]) if best_ep else ''
    combos = re.findall(r'Parameters: (\[.*?\]=\(.*?\)),\s*best valid: (.*?),\s*best test: (.*?)\n', text, re.S)
    row['_combos'] = [(c, metrics(v), metrics(t)) for c, v, t in combos]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='baby')
    ap.add_argument('--log_dir', default='src/log')
    ap.add_argument('--combos', action='store_true')
    ap.add_argument('--group_seeds', action='store_true')
    args = ap.parse_args()

    name_re = re.compile(r'^(?P<model>[^-]+)-{}-(?P<name>.+)-(?P<t>{})\.log$'.format(re.escape(args.dataset), TIME_RE))
    latest = {}
    for path in glob.glob(os.path.join(args.log_dir, '*-{}-*.log'.format(args.dataset))):
        m = name_re.match(os.path.basename(path))
        if not m:
            continue
        key = (m.group('model'), m.group('name'))
        t = datetime.strptime(m.group('t'), '%b-%d-%Y-%H-%M-%S')
        if key not in latest or t > latest[key]['time']:
            latest[key] = {'model': key[0], 'name': key[1], 'time': t, 'log': path}
    runs = []
    for r in latest.values():
        r.update(parse_log(r['log']))
        runs.append(r)
    if not runs:
        print('No logs with --exp_name in', args.log_dir)
        return
    runs.sort(key=lambda r: (r['model'], -r.get('test_recall@20', -1)))

    paper = PAPER.get(args.dataset, {})
    head = ['model', 'name', 'status', 'epochs', 'best_epoch', 'valid_recall@20'] + ['test_' + m for m in SHOW] + ['paper_R@20', 'gap_R@20']
    for r in runs:
        ref = paper.get(r['model'])
        if ref and 'test_recall@20' in r:
            r['paper_R@20'] = ref[1]
            r['gap_R@20'] = r['test_recall@20'] - ref[1]
    def fmt(x, h):
        if isinstance(x, float):
            return ('{:+.4f}' if h.startswith('gap') else '{:.4f}').format(x)
        return str(x)
    widths = [max(len(h), *(len(fmt(r.get(h, ''), h)) for r in runs)) for h in head]
    print('  '.join(h.ljust(w) for h, w in zip(head, widths)))
    print('  '.join('-' * w for w in widths))
    for r in runs:
        print('  '.join(fmt(r.get(h, ''), h).ljust(w) for h, w in zip(head, widths)))
        if r.get('params'):
            print('      selected on valid: ' + r['params'])
        if args.combos:
            for c, v, t in r['_combos']:
                print('      {:<60s} valid R@20={:.4f}  test R@20={:.4f}'.format(
                    c, v.get('recall@20', float('nan')), t.get('recall@20', float('nan'))))

    out = 'results_{}.csv'.format(args.dataset)
    keys = head + ['params'] + sorted({k for r in runs for k in r} - set(head) - {'params', 'time', '_combos'})
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k, '') for k in keys})
    print('\nSaved', out)

    if args.group_seeds:
        groups = {}
        for r in runs:
            if r['status'] == 'done' and re.search(r'_s\d+$', r['name']):
                groups.setdefault((r['model'], re.sub(r'_s\d+$', '', r['name'])), []).append(r)
        cols = ['valid_recall@20'] + ['test_' + m for m in SHOW]
        print('\nMean +/- std over seeds (finished runs named *_s<seed>)')
        print('  '.join(['model'.ljust(9), 'group'.ljust(16), 'n'] + [c.ljust(17) for c in cols]))
        for (model, g), rs in sorted(groups.items()):
            cells = []
            for c in cols:
                xs = [r[c] for r in rs if c in r]
                mu = sum(xs) / len(xs) if xs else float('nan')
                sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
                cells.append('{:.4f}+/-{:.4f}'.format(mu, sd).ljust(17))
            print('  '.join([model.ljust(9), g.ljust(16), str(len(rs))] + cells))


if __name__ == '__main__':
    main()
