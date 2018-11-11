#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate some output.
"""
import json
import pdb
import csv
import sys

from collections import defaultdict

from tqdm import tqdm, trange

import numpy as np
from prosestats.schema import SentenceGraph, prune_graph
from prosestats.evaluation import Scorer

def score(_, dev, metrics=None):
    if not metrics:
        metrics = sorted(Scorer.METRICS.keys())

    ret = {}
    scorer = Scorer(metrics)
    for gold, guess in dev:
        scorer.update(gold, guess)

    for m in sorted(scorer.state):
        ret[m, "prec"] = scorer.state[m].prec()
        ret[m, "rec"] = scorer.state[m].rec()
        ret[m, "f1"] = scorer.state[m].f1()
    return ret

def cross_validate(dataset, fn, splits=10, **kwargs):
    block_size = len(dataset)//splits

    ret = defaultdict(list)
    for i in trange(splits, desc="Cross-validation"):
        train = dataset[:i*block_size] + dataset[(i+1)*block_size:]
        dev = dataset[i*block_size:(i+1)*block_size]

        ret_ = fn(train, dev, **kwargs)
        for k, v in ret_.items():
            ret[k].append(v)

    return ret

def do_command(args):
    if not args.metrics:
        args.metrics = sorted(Scorer.METRICS.keys())

    gold_graphs = [prune_graph(SentenceGraph.from_json(json.loads(line))) for line in args.gold]
    guess_graphs = [SentenceGraph.from_json(json.loads(line)) for line in args.guess]
    dataset = list(zip(gold_graphs, guess_graphs))
    ret = cross_validate(dataset, score, args.cross_validation_splits, metrics=args.metrics)

    # Write header.
    writer = csv.writer(args.output)
    writer.writerow(["metric", "p", "r", "f1"])
    for m in sorted(args.metrics):
        writer.writerow([
            m,
            np.mean(ret[m, "prec"]),
            np.mean(ret[m, "rec"]),
            np.mean(ret[m, "f1"]),
            ])

    #if args.verbose:
    #    for m in sorted(scorer.state):
    #        args.output.write("== {} ==\n".format(m))
    #        args.output.write(scorer.state[m].as_table())
    #        args.output.write("\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-g', '--guess', type=argparse.FileType('r'), help="")
    parser.add_argument('-G', '--gold', type=argparse.FileType('r'), help="")
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="")
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="")
    parser.add_argument('-cv', '--cross-validation-splits', type=int, default=1, help="Cross-validation splits to use")
    parser.add_argument('-m', '--metrics', type=str, nargs="*", help="Metrics to report")
    parser.set_defaults(func=do_command)

    #subparsers = parser.add_subparsers()
    #command_parser = subparsers.add_parser('command', help='' )
    #command_parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
