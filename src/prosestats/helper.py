"""
Helpers to vectorize and process data data
"""
import pdb
import os
import json
import pickle
import logging

from collections import Counter
from itertools import product, chain

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer
from scipy import sparse

from .util import build_dict, invert_dict, cargmax
from .schema import SentenceGraph, overlaps, load_graphs, node_lbl, edge_lbl, prune_graph
from .wv import load_word_vector_mapping
from .ud_utils import construct_dependency_graph
from .candidates import generate_candidates
from .logistic_utils import span_feature_functions, edge_feature_functions

logger = logging.getLogger(__name__)

P_LEMMA = "LEMMA:"
P_POS = "POS:"
P_NER = "NER:"
P_CASE = "CASE:"
P_DEPREL = "REL:"
P_ORDERED_DEPREL = "OREL:"
P_REGEX = "REGEX:"
CASES = ["aa", "AA", "Aa", "aA"]
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK = "UNK"
NUM = "###"

def deprel(tag):
    return tag.split(":")[0]

def casing(word):
    if len(word) == 0: return word

    # all lowercase
    if word.islower(): return "aa"
    # all uppercase
    elif word.isupper(): return "AA"
    # starts with capital
    elif word[0].isupper(): return "Aa"
    # has non-initial capital
    else: return "aA"

def uncasing(word, case):
    if len(word) == 0: return word

    if case == "aa": return word.lower()
    elif case == "AA": return word.upper()
    elif case == "Aa": return word.title()
    else: return word

def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    try:
        float(word)
        return NUM
    except ValueError:
        return word.lower()

def test_vectorize():
    id2tok = ["a", "green", "cats", "ham", "like", UNK, P_CASE + "aa", P_CASE + "aA", P_CASE + "AA", P_CASE + "Aa", START_TOKEN, END_TOKEN]
    tok2id = {t: i for i, t in enumerate(id2tok)}

    helper = ModelHelper(tok2id, None, None, None, None, ["word"])
    assert helper.id2tok == id2tok
    # TODO: fix this test.
    #s = "Green cats like ham".split()
    #assert helper.unvectorize_example(helper.vectorize_example(s)) == s

# TODO: potentially handle label balances.
# node_label_counts = np.array([0. for _ in enumerate(SpanLabel.labels())], dtype=np.float32)
# edge_label_counts = np.array([0. for _ in enumerate(ArcLabel.labels()) ], dtype=np.float32)
# for graph in data:
#     for lbl in graph.nodes:
#         node_label_counts[lbl] += 1
#     for row in graph.edges:
#         for lbl in row:
#             edge_label_counts[lbl] += 1
# assert all(node_label_counts > 0)
# assert all(edge_label_counts > 0)
#
# node_class_weights = 1./np.sqrt(node_label_counts) # / node_label_counts.sum()
# edge_class_weights = 1./np.sqrt(edge_label_counts) # / edge_label_counts.sum()

class ModelHelper(object):
    """
    This helper takes care of preprocessing data, constructing embeddings, etc.
    """
    FEATURES = ["word", "casing", "lemma", "pos", "ner", "depparse", "depparse_copy", "depparse_dists", "regex",  "value", "depparse_path"]

    def __init__(self, tok2id, feat2id, lbl2id, node_counts, edge_counts, features):
        self.tok2id = tok2id
        self.feat2id = feat2id
        self.id2tok = invert_dict(tok2id)
        self.lbl2id = lbl2id
        self.node_counts = node_counts
        self.edge_counts = edge_counts
        self.embeddings = None
        self.features = features
        # added this for pathLSTM -- learned embeddings for pos tags and deprels
        self.learned_embeddings = None

    def save(self, f):
        pickle.dump([self.tok2id, self.feat2id, self.lbl2id, self.node_counts, self.edge_counts, self.features], f)

    @classmethod
    def load(cls, f):
        return cls(*pickle.load(f))

    @property
    def vocab_dim(self):
        return self.embeddings.shape[0]

    @property
    def feat_dim(self):
        return len(self.feat2id) + 1

    @property
    def embed_dim(self):
        return self.embeddings.shape[1]

    @property
    def n_features_fixed(self):
        ret = 0
        if "word" in self.features:
            ret += 1
        if "lemma" in self.features:
            ret += 1
        if "depparse_copy" in self.features:
            ret += 1
        return ret

    @property
    def n_features_learn(self):
        ret = 0
        if "casing" in self.features:
            ret += 1
        if "pos" in self.features:
            ret += 1
        if "ner" in self.features:
            ret += 1
        if "depparse_copy" in self.features:
            ret += 1
        if "depparse" in self.features:
            ret += 1
        if "regex" in self.features: # Always have values
            ret += 1
        if "value" in self.features: # Always have values
            ret += 0
        return ret

    @property
    def n_features_exact(self):
        ret = 0
        if "depparse_dists" in self.features:
            ret += 3
        if "value" in self.features:
            ret += 1
        return ret

    def vectorize_example(self, tokens, lbls, features):
        ret = []

        # Initializations
        dep_tags = [None for _ in tokens]
        dep_heads = [i for i, _ in enumerate(tokens)]
        dep_head_distances = [0 for _ in tokens]
        for head, tail, tag in features["depparse"]:
            dep_tags[tail] = deprel(tag)
            dep_heads[tail] = head
            dep_head_distances[tail] = tail - head if head > -1 else 0

        for i, (token, lbl) in enumerate(zip(tokens,lbls)):
            cell = []
            # The fixed features come first
            if "word" in self.features:
                cell.append(self.tok2id.get(normalize(token), self.tok2id[UNK]))
            if "lemma" in self.features:
                lemma = features["lemma"][i]
                cell.append(self.tok2id.get(normalize(lemma), self.tok2id[UNK]))
            if "depparse_copy" in self.features:
                # Use the vector of the head
                token_ = tokens[dep_heads[i]]
                cell.append(self.tok2id.get(normalize(token_), self.tok2id[UNK]))

            # The learned features come first
            if "casing" in self.features:
                cell.append(self.feat2id[P_CASE + casing(token)])
            if "pos" in self.features:
                pos_tag = features["pos"][i]
                cell.append(self.feat2id[P_POS + pos_tag])
            if "ner" in self.features:
                ner_tag = features["ner"][i]
                cell.append(self.feat2id[P_NER + ner_tag])
            if "depparse_copy" in self.features:
                # Use the vector of the head
                token_ = tokens[dep_heads[i]]
                cell.append(self.feat2id[P_CASE + casing(token_)])
            if "depparse" in self.features:
                dep_tag = dep_tags[i]
                cell.append(self.feat2id[P_DEPREL + dep_tag])
            if "regex" in self.features:
                tag = features["typed_values"][i]
                cell.append(self.feat2id[P_REGEX + tag])

            # IMPORTANT: The "exact" features come at the very end.
            if "depparse_dists" in self.features:
                cell.append(dep_head_distances[i])
                cell.append(features["dep_dist_to_next"][i])
                cell.append(features["dep_dist_from_prev"][i])
            if "value" in self.features: # Always have values
                cell.append(1 if lbl == "value" else 0)
            ret.append(cell)
        return np.array(ret, dtype=np.int64)

    def vectorize_nodes(self, nodes):
        return np.array([self.lbl2id[lbl] for lbl in nodes], dtype=np.int64)

    def vectorize_edges(self, edges):
        ret = np.zeros((len(edges), len(edges[0])), dtype=np.int64)
        for i, lbls in enumerate(edges):
            for j, lbl in enumerate(lbls):
                ret[i,j] = self.lbl2id[lbl]
        return ret

    def vectorize_dependency_paths(self, tokens, features):
        child_to_head = {
            int(i): {int(j):features["dep_child_to_head"][i][j] for j in features["dep_child_to_head"][i]}
            for i in features["dep_child_to_head"]
            }
        next_in_path = {
            int(i): features["dep_traceback"][i]
            for i in features["dep_traceback"]
            }

        ret = [[[] for v in range(len(tokens))] for u in range(len(tokens))]
        #ret_ = [[[] for v in range(len(tokens))] for u in range(len(tokens))]

        for u in range(0,len(tokens)):
            for v in range(0,len(tokens)):
                u_ = u
                if u != v: 
                    path = [u_]
                    while u_ != v:
                        u_ = next_in_path[u_][v]
                        path.append(u_)  

                    sequence = []
                    for i in range(0,len(path) - 1):
                        idx = path[i]

                        if path[i+1] in child_to_head[idx]:
                            rel = deprel(child_to_head[idx][path[i+1]]) + "<--"
                        else:
                            rel = deprel(child_to_head[path[i+1]][idx]) + "-->"

                        sequence.append((idx, rel))    

                    sequence.append((path[-1],None)) 

                    ## here is where things start to change
                    #word_vec = []
                    path_vec = []
                    for word, rel in sequence:
                        # fetch vectors, append stuff
                        #word_id = self.tok2id[normalize(word)]
                        #word_vec.append(word_id)

                        if rel != None:
                            # Escape hatch for no depparse.
                            rel_id = self.feat2id[P_ORDERED_DEPREL + rel] if "depparse" in self.features else 0
                            path_vec.append(rel_id)    

                    ret[u][v] = np.array(path_vec, dtype=np.int64)
                    #ret_[u][v] = np.array(word_vec, dtype=np.int64)

        return ret #, ret_

    def vectorize(self, data):
        """
        Returns vectorized sequences of the training data:
        Each returned element consists of an input, node label, lexicalized edge dependency path and edge label
        sequences (each is a numpy array).
        """
        ret = []
        for graph in tqdm(data, desc="vectorizing data"):
            x = self.vectorize_example(graph.tokens, graph.per_token_labels, graph.features)
            y = self.vectorize_nodes(graph.per_token_labels)
            q = self.vectorize_dependency_paths(graph.tokens, graph.features)
            z = self.vectorize_edges(graph.per_token_edge_labels)
            ret.append([x,y,q,z])
        return ret

    @classmethod
    def build(cls, data, features=None, test_data=None):
        """
        Use @data to construct a featurizer.
        """
        if not features:
            features = ModelHelper.FEATURES
        else:
            assert all(f in ModelHelper.FEATURES for f in features)

        # Preprocess data to construct an embedding
        # Reserve 0 for the special NIL token.
        tok2id, feat2id = {}, {}
        if "word" in features:
            tok2id.update(build_dict((normalize(word) for graph in chain(data, test_data) for word in graph.tokens), offset=len(tok2id)+1, max_words=10000))
        if "lemma" in features:
            pass
        if "casing" in features:
            feat2id.update(build_dict([P_CASE + c for c in CASES], offset=len(feat2id)+1))
        if "pos" in features:
            feat2id.update(build_dict((P_POS + word for graph in data for word in graph.features["pos"]), offset=len(feat2id)+1))
        if "ner" in features:
            feat2id.update(build_dict((P_NER + word for graph in data for word in graph.features["ner"]), offset=len(feat2id)+1))
        if "depparse" or "depparse_path" in features:
            feat2id.update(build_dict((P_DEPREL + deprel(tag) for graph in data for _, _, tag in graph.features["depparse"]), offset=len(feat2id)+1))
        if "depparse_path" in features:
            ## add tokens for ordered dependency relations
            feat2id.update(build_dict((P_ORDERED_DEPREL + deprel(tag) + "-->" for graph in data for _,_, tag in graph.features["depparse"]), offset=len(feat2id)+1))
            feat2id.update(build_dict((P_ORDERED_DEPREL + deprel(tag) + "<--" for graph in data for _,_, tag in graph.features["depparse"]), offset=len(feat2id)+1))
        if "regex" in features:
            feat2id.update(build_dict((P_REGEX + tag for graph in data for tag in graph.features["typed_values"]), offset=len(feat2id)+1))

        tok2id.update(build_dict([START_TOKEN, END_TOKEN, UNK], offset=len(tok2id)+1))
        logger.info("Built dictionary for %d tokens and %d features.", len(tok2id)+1, len(feat2id)+1)

        lbl2id = {None: 0}
        lbl2id.update({lbl: i+1 for i, lbl in enumerate(SentenceGraph.NODE_LABELS)})
        lbl2id.update({lbl: i+1 for i, lbl in enumerate(SentenceGraph.EDGE_LABELS)})

        node_counts = [0. for _ in range(1+len(SentenceGraph.NODE_LABELS))]
        edge_counts = [0. for _ in range(1+len(SentenceGraph.EDGE_LABELS))]
        for graph in tqdm(data, desc="Counting labels"):
            for lbl in graph.per_token_labels:
                node_counts[lbl2id[lbl]] += 1
            for lbl in chain.from_iterable(graph.per_edge_labels):
                edge_counts[lbl2id[lbl]] += 1
        logger.info("Node counts: %s", node_counts)
        logger.info("Edge counts: %s", edge_counts)

        return cls(tok2id, feat2id, lbl2id, node_counts, edge_counts, features)

    def load_embeddings(self, fstream):
        wvecs = load_word_vector_mapping(fstream)
        embed_size = len(next(iter(wvecs.values())))

        embeddings = np.array(np.random.randn(len(self.tok2id) + 1, embed_size), dtype=np.float32)
        embeddings[0,:] = 0. # (padding) zeros vector.
        for word, vec in wvecs.items():
            word = normalize(word)
            if "word" in self.features:
                if word in self.tok2id:
                    embeddings[self.tok2id[word]] = vec
        logger.info("Initialized embeddings.")

        return embeddings

    def add_embeddings(self, fstream):
        self.embeddings = self.load_embeddings(fstream)

def load_and_preprocess_data(args):
    """
    Loads training data from @args.data_train and uses this to build a
    ModelHelper that handles embedding and featurization of examples.
    """
    logger.info("Loading training data...")
    train_graphs = list(tqdm(load_graphs(args.data_train)))
    train_graphs = [prune_graph(graph) for graph in train_graphs]
    logger.info("Done. Read %d sentences", len(train_graphs))

    test_graphs = list(tqdm(load_graphs(args.data_test)))
    test_graphs = [prune_graph(graph) for graph in test_graphs]

    helper = ModelHelper.build(train_graphs, features=args.features, test_data=test_graphs)
    helper.add_embeddings(args.embeddings)

    return helper, train_graphs

def test_load_and_preprocess_data():
    class Object:
        data_train = open(os.path.join(os.path.dirname(__file__), "testdata", "test.json"))
        data_test = open(os.path.join(os.path.dirname(__file__), "testdata", "test.json"))
        embeddings = open(os.path.join(os.path.dirname(__file__), "testdata", "test.vectors"))
        features = ModelHelper.FEATURES
    args = Object()
    helper, train_graphs = load_and_preprocess_data(args)

    data = helper.vectorize(train_graphs)

    # make sure the lengths all match
    for x, y, q, z in data:
        assert x.shape[0] == y.shape[0]
        assert x.shape[0] == z.shape[0]
        assert z.shape[0] == z.shape[1]

## Matt Lamm extensions start here
def load_and_preprocess_featurized_graphs(fstream):
    """
    Loads featurized Frames from @args.data_train
    Employed by logistic regression
    """
    logger.info("Loading training data...")
    train_graphs = list(load_graphs(fstream))
    for graph in train_graphs:
        graph.features["dependency_graph"] = construct_dependency_graph(graph)
    logger.info("Done. Read %d sentences", len(train_graphs))
    return train_graphs
