#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute common stats.
"""

import pdb
import csv
import sys
from xml.etree import ElementTree
from collections import defaultdict, Counter
import logging
import logging.config

from tqdm import tqdm
from prosestats.data import compute_stats_xml
from prosestats.schema import load_graphs, SentenceGraph 
from prosestats.evaluation import _match_spans
from prosestats.stats import compute_alpha
from prosestats.util import cargmax

logger = logging.getLogger(__name__)

def do_frames(args):
    tree = ElementTree.parse(args.input)

    stats = Counter()

    for sentence_ix, sentence in enumerate(tree.findall("S")):
        if "skip" in sentence.attrib: continue
        n_frames, n_instances = compute_stats_xml(sentence)
        stats["frames"] += n_frames
        stats["instances"] += n_instances
        stats["n"] += 1

        stats["max_frames"] = max(n_frames, stats["max_frames"])
        stats["max_instances"] = max(n_instances, stats["max_instances"])

    writer = csv.writer(args.output, delimiter='\t')
    writer.writerow(["", "total", "average", "max",])
    writer.writerow(["frames"   , stats["frames"], stats["frames"]/stats["n"], stats["max_frames"],])
    writer.writerow(["instances"   , stats["instances"], stats["instances"]/stats["n"], stats["max_instances"],])

def do_iaa(args):
    graphs_a = list(tqdm(load_graphs(args.input_a), desc="loading graphs"))
    graphs_b = list(tqdm(load_graphs(args.input_b), desc="loading graphs"))
    assert len(graphs_a) == len(graphs_b), "# graphs in A don't match those in B?!"

    token_table = defaultdict(Counter)
    edge_table = defaultdict(Counter)

    for i, (graph_a, graph_b) in tqdm(enumerate(zip(graphs_a, graphs_b)), desc="measuring agreement"):
        assert graph_a.tokens == graph_b.tokens,\
                "Graph {} doesn't match between the two files:\nA = {},\nB = {}".format(
                        i, " ".join(graph_a.tokens), " ".join(graph_b.tokens))
        for j, lbl in enumerate(graph_a.per_token_labels):
            token_table["A"]["{}-{}".format(i,j)] = SentenceGraph.NODE_LABELS.index(lbl)+1 if lbl in SentenceGraph.NODE_LABELS else 0
        for j, lbl in enumerate(graph_b.per_token_labels):
            token_table["B"]["{}-{}".format(i,j)] = SentenceGraph.NODE_LABELS.index(lbl)+1 if lbl in SentenceGraph.NODE_LABELS else 0

        # Do matching:
        spans_a = {span: cargmax(attr) for span, attr, *_ in graph_a.nodes}
        spans_b = {span: cargmax(attr) for span, attr, *_ in graph_b.nodes}
        edges_a = {(span, span_): cargmax(attr) for span, span_, attr in graph_a.edges}
        edges_b = {(span, span_): cargmax(attr) for span, span_, attr in graph_b.edges}

        mapping, _ = _match_spans(sorted(spans_a.keys()), sorted(spans_b.keys()))
        edges_a    = {(mapping.get(span, span), mapping.get(span_, span_)): attr for (span, span_), attr in edges_a.items()}

        for (span, span_), attr in edges_a.items():
            edge_table["A"]["{}-{}:{}".format(i, span, span_)] = SentenceGraph.EDGE_LABELS.index(attr)+1 if attr else 0
        for (span, span_), attr in edges_b.items():
            edge_table["B"]["{}-{}:{}".format(i, span, span_)] = SentenceGraph.EDGE_LABELS.index(attr)+1 if attr else 0

        #pdb.set_trace()

    logger.info("Computing agreement")
    logger.info("Token alpha: %.3f", compute_alpha(token_table, "nominal"))
    logger.info("Edge alpha: %.3f", compute_alpha(edge_table, "nominal"))



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('frames', help='' )
    command_parser.add_argument('-i', '--input', type=argparse.FileType('r'), default=sys.stdin, help="")
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="")
    command_parser.set_defaults(func=do_frames)

    command_parser = subparsers.add_parser('iaa', help='Computes inter annotator agreement' )
    command_parser.add_argument('-ia', '--input-a', type=argparse.FileType('r'), help="")
    command_parser.add_argument('-ib', '--input-b', type=argparse.FileType('r'), help="")
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="")
    command_parser.set_defaults(func=do_iaa)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
