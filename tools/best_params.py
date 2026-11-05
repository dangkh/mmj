# coding: utf-8
"""
Print the hyper-parameters selected on validation by a finished grid run, as --key=value overrides.

    python tools/best_params.py FREEDOM sports freedom_grid      # -> --dropout=0.8 --reg_weight=0.001
Exit code 1 (and nothing printed) if no finished log exists.
"""
import ast
import glob
import os
import re
import sys


def best_params(model, dataset, name, log_dir='src/log', skip=('seed',)):
    logs = sorted(glob.glob(os.path.join(log_dir, '{}-{}-{}-*.log'.format(model, dataset, name))), key=os.path.getmtime)
    for path in reversed(logs):
        text = open(path, encoding='utf-8', errors='ignore').read()
        i = text.rfind('BEST ████')
        if i < 0:
            continue
        m = re.search(r'Parameters:\s*(\[.*?\])=(\(.*?\))', text[i:], re.S)
        if not m:
            continue
        keys = ast.literal_eval(m.group(1))
        vals = ast.literal_eval(m.group(2))
        if not isinstance(vals, tuple):
            vals = (vals,)
        return {k: v for k, v in zip(keys, vals) if k not in skip}
    return None


if __name__ == '__main__':
    p = best_params(*sys.argv[1:4])
    if p is None:
        sys.exit(1)
    print(' '.join('--{}={}'.format(k, v) for k, v in p.items()))
