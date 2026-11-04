# coding: utf-8
# @email: enoche.chow@gmail.com
# mmj: command-line overrides (--key=value), --exp_name and --mg added.

"""
Main entry. Run from inside src/ (configs/ and ../data/ are relative to it):

    cd src
    python main.py -m FREEDOM -d baby
    python main.py -m DRAGON -d baby --exp_name=dragon_s999 --learning_rate=0.0001 --reg_weight=0.001 --seed=999
    python main.py -m MGCN -d baby --save_taps=True

Any extra --key=value overrides that config key (highest priority). Values are parsed
as YAML, so --seed=[999,1] or --cl_loss=0.01 work. A scalar given for a key listed in
hyper_parameters fixes that key to one value (no grid over it).
##########################
"""

import os
import argparse
import yaml
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


def parse_overrides(tokens):
    overrides, i = {}, 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith('--'):
            raise ValueError('Unexpected argument: {}'.format(tok))
        tok = tok.lstrip('-')
        if '=' in tok:
            key, raw = tok.split('=', 1)
        else:
            key = tok
            if i + 1 >= len(tokens) or tokens[i + 1].startswith('--'):
                raise ValueError('Missing value for --{}'.format(key))
            i += 1
            raw = tokens[i]
        overrides[key] = _numbers(yaml.safe_load(raw))
        i += 1
    return overrides


def _numbers(v):
    """PyYAML reads 1e-05 (no dot) as a string; turn such strings into floats, also inside lists."""
    if isinstance(v, list):
        return [_numbers(x) for x in v]
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--mg', action='store_true', help='use Mirror Gradient (configs/mg.yaml)')

    config_dict = {
        'gpu_id': 0,
    }

    args, unknown = parser.parse_known_args()
    overrides = parse_overrides(unknown)
    if overrides:
        print('Config overrides from command line:', overrides)
    config_dict.update(overrides)

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True, mg=args.mg)
