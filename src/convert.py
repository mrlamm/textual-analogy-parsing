#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert some graphs to frames (for easy visualization).
"""

import json
import sys
from prosestats.schema import load_graphs
from prosestats.data import StandoffWriter, parse_xml
from prosestats.features import Featurizer

def do_featurize(args):
    with Featurizer() as featurizer:
        for obj in (json.loads(line) for line in args.input):
            obj = featurizer.featurize_graph(obj)
            args.output.write(json.dumps(obj))
            args.output.write("\n")

def do_xml2graph(args):
    for frames in parse_xml(args.input):
        args.output.write(json.dumps(frames.as_json()))
        args.output.write("\n")

def do_graph2standoff(args):
    with open(args.output_prefix + ".txt", "w") as txt, open(args.output_prefix + ".ann", "w") as ann:
        writer = StandoffWriter(txt, ann)

        for graph in load_graphs(args.input):
            writer.write(graph)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('xml2graph', help='' )
    command_parser.add_argument('-i', '--input', type=argparse.FileType('r'), default=sys.stdin, help="Input xml")
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="Output frames json")
    command_parser.set_defaults(func=do_xml2graph)

    command_parser = subparsers.add_parser('graph2standoff', help='' )
    command_parser.add_argument('-i', '--input', type=argparse.FileType('r'), default=sys.stdin, help="Input sentence graphs json")
    command_parser.add_argument('-o', '--output-prefix', type=str, default='graph', help="Output prefix")
    command_parser.set_defaults(func=do_graph2standoff)

    command_parser = subparsers.add_parser('featurize', help='Featurizes either a graph or a frame' )
    command_parser.add_argument('-i', '--input', type=argparse.FileType('r'), default=sys.stdin, help="Input json")
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="Output json")
    command_parser.set_defaults(func=do_featurize)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
