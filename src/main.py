#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prosestats identifies graphable groups from natural language text.
"""
import os
import gc
import pdb
import csv
import json
import sys
import logging
import logging.config
from io import StringIO

import numpy as np
import torch

from tqdm import tqdm, trange

from prosestats.data import StandoffWriter
from prosestats.models import Model
from prosestats.torch_utils import train_model, run_model, GraphDataset, run_example
from prosestats.helper import load_and_preprocess_data, ModelHelper
from prosestats.util import print_sentence, dictstr
from prosestats.util import vnormalize
from prosestats.schema import load_graphs, prune_graph, SentenceGraph
from prosestats import features

logger = logging.getLogger(__name__)

def split(dataset, train=0.8):
    """
    Splits the dataset into train and test.
    """
    pivot = int(len(dataset) * train)
    return dataset[:pivot], dataset[pivot:]

def cross_validate(dataset, fn, splits=5, iters=None, start=0, **kwargs):
    block_size = len(dataset)//splits

    if iters is None:
        iters = splits

    it = trange(start, min(splits, start+iters), desc="Cross-validation")

    train_stats, dev_stats, output, epochs = [], [], [], []
    for i in it:
        train = dataset[:i*block_size] + dataset[(i+1)*block_size:]
        dev = dataset[i*block_size:(i+1)*block_size]

        _, output_, epoch, train_stats_, dev_stats_ = fn(train, dev, **kwargs)
        output.extend(output_)
        train_stats.append(train_stats_)
        dev_stats.append(dev_stats_)
        epochs.append(epoch)

        it.set_postfix(stats=",".join("{:.3f}".format(x) for x in np.mean(dev_stats,0)))

    return output, np.array(train_stats), np.array(dev_stats), int(np.median(epochs))

def run_split(train_raw, dev_raw, helper, model_class, config, use_cuda=False, n_epochs=15, **train_args):
    gc.collect()
    train = GraphDataset(train_raw)
    dev = dev_raw and GraphDataset(dev_raw)

    # (d) Train model
    model = model_class(config, helper.embeddings)
    model, epoch, train_stats, dev_stats = train_model(model, train, dev, use_cuda=use_cuda, n_epochs=n_epochs, **train_args)
    output = run_model(model, dev, use_cuda) if dev else []

    return model, output, epoch, train_stats, dev_stats

def write_stats(stats, f):
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["split",] + sorted([
        "token_node",
        "span_node",
        "span_node_nomatch",
        "token_edge",
        "span_edge",
        "span_edge_nomatch",
        "decoded_span_node",
        "decoded_span_edge",
        "decoded_span_edge_nomatch",
        ]))
    for i, row in enumerate(stats):
        writer.writerow([i,] + row.tolist())
    writer.writerow(["mean",] + np.mean(stats,0).tolist())

def do_train(args):
    torch.manual_seed(args.seed)
    # (a) Load data and embeddings
    # (b) Normalize the data.
    helper, train_graphs = load_and_preprocess_data(args)
    dev_graphs = args.data_dev and list(tqdm(load_graphs(args.data_dev)))

    # (c) Get model and configure it.
    model_args = dict(args.model_args or ())
    train_args = dict(args.train_args or ())

    config = Model.make_config()
    config.update({
        "n_features_fixed": helper.n_features_fixed,
        "n_features_learn": helper.n_features_learn,
        "n_features_exact": helper.n_features_exact,
        "vocab_dim": helper.vocab_dim,
        "embed_dim": helper.embed_dim,
        "feat_dim" : helper.feat_dim,
        })
    if model_args.get("balance"):
        logger.info("Balancing labels")
        config.update({
            "node_weights": vnormalize([1./c if c > 0 else 0 for c in helper.node_counts]),
            "edge_weights": vnormalize([1./c if c > 0 else 0 for c in helper.edge_counts]),
            })
    config.update(model_args)

    with open(os.path.join(args.model_path, "helper.pkl"), "wb") as f:
        helper.save(f)
    with open(os.path.join(args.model_path, "model.config"), "w") as f:
        json.dump(config, f)

    # Cross-validation information.
    train = helper.vectorize(train_graphs)
    if args.cross_validation_iters > 0:
        dev_output, train_stats, dev_stats, args.n_epochs = cross_validate(
            train, run_split,
            splits=args.cross_validation_splits,
            iters=args.cross_validation_iters,
            start=args.cross_validation_start,
            model_class=Model, config=config, helper=helper, use_cuda=args.cuda,
            n_epochs=args.n_epochs,
            **train_args)
        logger.info("Final cross-validated train stats: %s", np.mean(train_stats,0))
        logger.info("Final cross-validated dev stats: %s", np.mean(dev_stats,0))

        with open(os.path.join(args.model_path, "train.scores"), "w") as f:
            write_stats(train_stats, f)

        with open(os.path.join(args.model_path, "dev.scores"), "w") as f:
            write_stats(dev_stats, f)

        with open(os.path.join(args.model_path, "predictions.json"), 'w') as f:
            for graph, (yhs, zhs) in zip(train_graphs, dev_output):
                graph_ = graph.with_predictions(yhs, zhs, collapse_spans=True)
                json.dump(graph_.as_json(), f)
                f.write("\n")

    # Run model and save model.
    if args.save:
        model, _, _, _, _ = run_split(train, None, helper, Model, config, use_cuda=args.cuda, n_epochs=args.n_epochs)

        with open(os.path.join(args.model_path, "model.pkl"), "wb") as f:
            torch.save(model.state_dict(), f)

        if dev_graphs:
            dev = helper.vectorize(dev_graphs)
            dev_output = run_model(model, GraphDataset(dev), args.cuda)

            with open(os.path.join(args.model_path, "dev_predictions.json"), 'w') as f:
                for graph, (yhs, zhs) in zip(dev_graphs, dev_output):
                    graph_ = graph.with_predictions(yhs, zhs, collapse_spans=True)
                    json.dump(graph_.as_json(), f)
                    f.write("\n")

def do_run(args):
    # Runs model on input text.
    logger.info("Loading data...")
    graphs = list(tqdm(load_graphs(args.data)))
    graphs = [prune_graph(graph) for graph in graphs]
    logger.info("Done. Read %d sentences", len(graphs))

    with open(os.path.join(args.model_path, "model.config"), "r") as f:
        config = json.load(f)
    with open(os.path.join(args.model_path, "helper.pkl"), "rb") as f:
        helper = ModelHelper.load(f)
        helper.add_embeddings(args.embeddings)

    model = Model(config, helper.embeddings)
    with open(os.path.join(args.model_path, "model.pkl"), "rb") as f:
        model.load_state_dict(torch.load(f))
    data = helper.vectorize(graphs)
    output = run_model(model, GraphDataset(data), args.cuda)

    for graph, (yhs, zhs) in zip(graphs, output):
        graph_ = graph.with_predictions(yhs, zhs, collapse_spans=True)
        json.dump(graph_.as_json(), args.output)
        args.output.write("\n")

def do_shell(args):
    with open(os.path.join(args.model_path, "model.config"), "r") as f:
        config = json.load(f)
    with open(os.path.join(args.model_path, "helper.pkl"), "rb") as f:
        helper = ModelHelper.load(f)
        helper.add_embeddings(args.embeddings)

    model = Model(config, helper.embeddings)
    with open(os.path.join(args.model_path, "model.pkl"), "rb") as f:
        model.load_state_dict(torch.load(f))

    with features.Featurizer(properties=features.CORENLP_PROPERTIES_TEST) as featurizer:
        logger.info("Warming up the featurizer...")
        featurizer.featurize_text("Warming up the featurizer")
        logger.info("Ready to interact!")


        txt_stream = StringIO()
        writer = StandoffWriter(txt_stream, args.output)
        while True:
            txt = input("> ")
            #txt = "Stocks rose 10% Monday to 1324 from 1224 last Friday."
            obj = featurizer.featurize_text(txt)
            # TODO: fix the use of labels.
            x = helper.vectorize_example(obj["tokens"], [""] * len(obj["tokens"]), obj["features"])

            yhs, zhs = run_example(model, x)
            graph = SentenceGraph(tokens=obj["tokens"]).with_predictions(yhs, zhs, collapse_spans=True)

            writer.write(graph)
            txt_stream.truncate(0)
            break


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')

    import argparse
    parser = argparse.ArgumentParser(description='Prose-stats: Graph your text.')
    parser.add_argument('-c', '--cuda', action="store_true", default=False, help="Use cuda")
    # NO LONGER NEEDED vv
    #parser.add_argument('-m', '--model', choices=["WindowModel", "BiLSTMModel", "BiLSTMCRFModel", "ConvolutionalSpansModel"], default="WindowModel", help="Which model to use?")
    parser.add_argument('-mp', '--model-path', default="out", help="Where to load/save models.")
    parser.add_argument('-eve', '--embeddings',   type=argparse.FileType('r'), default="glove/glove.6B.50d.txt",   help="Path to glove word vectors")
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('train', help='Cross-validate a model.')
    command_parser.add_argument('-s', '--seed', type=int, default=10, help='Random seed to use')
    command_parser.add_argument('-f', '--features', nargs="+", default=None, help='Features to use in the helper')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), required=True, help='Training dataset.')
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), required=False, default=None, help='Dev set to run on after train.')
    command_parser.add_argument('-dT', '--data-test', type=argparse.FileType('r'), required=False, help='Test dataset.')
    command_parser.add_argument('-cvs', '--cross-validation-splits', type=int, default=10, help="Cross-validation splits to use")
    command_parser.add_argument('-cvi', '--cross-validation-iters', type=int, default=1, help="Cross-validation splits to use")
    command_parser.add_argument('-cvx', '--cross-validation-start', type=int, default=0, help="Cross-validation splits to start with")
    command_parser.add_argument('-n', '--n_epochs', type=int, default=10, help='How many iterations to train')
    command_parser.add_argument('-x', '--model-args', nargs='+', type=dictstr, help='Any additional arguments to be passed into the model configuration')
    command_parser.add_argument('-t', '--train-args', nargs='+', type=dictstr, help='Any additional arguments to be passed into the model configuration')
    command_parser.add_argument('--save', action="store_true", help='Save model at the end.')
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('run', help='Run a trained model on something')
    command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), required=True, help='Data to run on.')
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), required=True, help='Where to save things')
    command_parser.set_defaults(func=do_run)

    command_parser = subparsers.add_parser('shell', help='Run an interactive shell')
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help='Where to save things')
    command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        # Set up logging directory.
        os.makedirs(ARGS.model_path, exist_ok=True)
        root_logger = logging.getLogger()
        root_logger.addHandler(logging.FileHandler(os.path.join(ARGS.model_path, "log")))
        ARGS.func(ARGS)
