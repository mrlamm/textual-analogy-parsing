#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Applies the ILP on some graph output.
"""

import os
import sys
import json
import logging
from tqdm import tqdm
from prosestats.ilp_multiclass_analogy import solve_ilp
from prosestats.schema import SentenceGraph

def do_command(args):
    for line in tqdm(args.input):
        graph = SentenceGraph.from_json(json.loads(line))
        graph_ = solve_ilp(graph)
        json.dump(graph_.as_json(), args.output)
        args.output.write("\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-mp', '--model-path', default="out", help="Where to load/save models.")
    parser.add_argument('-i', '--input', type=argparse.FileType('r'), default=sys.stdin, help="A file containing sentence graphs for each line.")
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="")
    parser.set_defaults(func=do_command)

    #subparsers = parser.add_subparsers()
    #command_parser = subparsers.add_parser('command', help='' )
    #command_parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        os.makedirs(ARGS.model_path, exist_ok=True)
        root_logger = logging.getLogger()
        root_logger.addHandler(logging.FileHandler(os.path.join(ARGS.model_path, "log")))
        ARGS.func(ARGS)
