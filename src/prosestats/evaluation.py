"""
Handles evaluation of nodes and edges.
"""
import pdb
from collections import defaultdict, Counter
import numpy as np

from munkres import Munkres

from .schema import SentenceGraph, Frames, Frame, Instance, convert_frames_to_graph, convert_graph_to_frames
from .schema import overlap, noverlap, chunk_tokens, node_lbl, edge_lbl
from .util import ConfusionMatrix, cargmax, PRF1
from .greedy_decode import cluster_graph

IGNORE_LABELS = ["value"]
#IGNORE_LABELS = []

## may not need this anymore
def _per_token_labels(edges, length):
    ret = np.zeros((length, length))
    for (start, end), (start_, end_), attr in edges:
        ret[start:end, start_:end_] = attr
    return ret

def _match_spans(gold_spans, guess_spans):
    if not gold_spans or not guess_spans:
        return {}, {}
    matrix = np.zeros((len(gold_spans), len(guess_spans)))
    for i, gold_span in enumerate(gold_spans):
        for j, guess_span in enumerate(guess_spans):
            matrix[i,j] = noverlap(gold_span, guess_span)
    # Convert to cost matrix.
    zero = matrix.max()
    matrix = zero - matrix
    indices = Munkres().compute(matrix.tolist())

    mapping = {gold_spans[i]: guess_spans[j] for i, j in indices if matrix[i,j] < zero}
    inv_mapping = {guess_spans[j]: gold_spans[i] for i, j in indices if matrix[i,j] < zero}

    return mapping, inv_mapping

class Scorer(object):
    """
    Computes the following evaluation metrics:
    - token-wise F1 scores for attribute.
    - (core attribute) span-wise F1 scores for attribute tagging.
    - edge-wise F1 scores (based on exact span-matches and overlap).
    - per-frame exact match on core/non-core attributes.
    """
    METRICS = {
        "token_node": lambda: ConfusionMatrix([None,] + sorted(SentenceGraph.NODE_LABELS), None, IGNORE_LABELS),
        "token_edge": lambda: ConfusionMatrix([None,] + sorted(SentenceGraph.EDGE_LABELS), None),
        "span_node": lambda: ConfusionMatrix([None,] + sorted(SentenceGraph.NODE_LABELS), None, IGNORE_LABELS),
        "span_edge": lambda: ConfusionMatrix([None,] + sorted(SentenceGraph.EDGE_LABELS), None),
        "span_node_nomatch": lambda: ConfusionMatrix([None,] + sorted(SentenceGraph.NODE_LABELS), None, IGNORE_LABELS),
        "span_edge_nomatch": lambda: ConfusionMatrix([None,] + sorted(SentenceGraph.EDGE_LABELS), None),
        "decoded_span_node": lambda: ConfusionMatrix([None,] + sorted(SentenceGraph.NODE_LABELS), None, IGNORE_LABELS),
        "decoded_span_edge": lambda: ConfusionMatrix([None,] + sorted(SentenceGraph.EDGE_LABELS), None),
        "decoded_span_edge_nomatch": lambda: ConfusionMatrix([None,] + sorted(SentenceGraph.EDGE_LABELS), None),
        }
    DEFAULT_METRICS = [
        "token_node",
        "token_edge",
        "span_node",
        "span_edge",
        "span_node_nomatch",
        "span_edge_nomatch",
        ]

    def __init__(self, metrics=None):
        if metrics is None:
            metrics = self.DEFAULT_METRICS

        assert all(m in Scorer.METRICS for m in metrics)
        self.metrics = metrics

        self.state = {k: Scorer.METRICS[k]() for k in self.metrics}

    def clear(self):
        for state in self.state.values():
            state.clear()

    def __str__(self):
        return " ".join("{}={:.2f}".format(key, state.f1() * 100) for key, state in self.state.items())

    def summary(self):
        return [self.state[k].f1() for k in self.metrics]

    def _update_token_node_metrics(self, gold, guess):
        """
        Expects a sequence of gold and guess token labels.
        """
        if not any(m in self.metrics for m in ["token_node",]): return

        for l, l_ in zip(gold, guess):
            l, l_ = node_lbl(l), node_lbl(l_)

            if "token_node" in self.metrics:
                self.state["token_node"].update(l, l_)

    def _update_token_edge_metrics(self, gold, guess):
        """
        @gold and @guess are 2d arrays with labels.
        """
        if not any(m in self.metrics for m in ["token_edge",]): return
        L = len(gold)

        assert L > 0
        assert len(guess) == L
        assert len(gold[0]) == L
        assert len(guess[0]) == L

        for i in range(L):
            for j in range(i+1, L):
                l = edge_lbl(gold[i][j])
                l_ = edge_lbl(guess[i][j])
                self.state["token_edge"].update(l, l_)

    def _update_span_node_metrics(self, gold, guess, with_matching=False):
        """
        Expects a sequence of (span, attr).
        """
        if not any(m in self.metrics for m in ["span_node","span_node_nomatch"]): return
        metric = "span_node" if with_matching else "span_node_nomatch"

        gold = {span: node_lbl(attr) for span, attr in gold}
        guess = {span: node_lbl(attr) for span, attr in guess}

        if guess and with_matching:
            mapping, _ = _match_spans(sorted(gold.keys()), sorted(guess.keys()))
            gold = {mapping.get(span, span): attr for span, attr in gold.items()}

        for span, attr in gold.items():
            attr_ = guess.get(span)
            # We don't care about None, None anyways...
            self.state[metric].update(attr, attr_)

        for span, attr_ in guess.items():
            self.state[metric].update(None, attr_)

    def _update_span_edge_metrics(self, gold_spans, gold_edges, guess_spans, guess_edges, with_matching=False):
        """
        @_spans are arrays of "span", "attr"
        @_edges spans in @_spans.
        """
        if not any(m in self.metrics for m in ["span_edge","span_edge_nomatch"]): return
        metric = "span_edge" if with_matching else "span_edge_nomatch"

        gold_spans = {span: node_lbl(attr) for span, attr in gold_spans}
        gold_edges = {(span, span_): edge_lbl(attr) for span, span_, attr in gold_edges}

        guess_spans = {span: node_lbl(attr) for span, attr in guess_spans}
        guess_edges = {(span, span_): edge_lbl(attr) for span, span_, attr in guess_edges}

        # Match spans ind gold and guess.
        if guess_spans and with_matching:
            mapping, _ = _match_spans(sorted(gold_spans.keys()), sorted(guess_spans.keys()))
            gold_edges = { (mapping.get(span, span), mapping.get(span_, span_)): gold_edges[(span,span_)] for span, span_ in gold_edges }

        for (span, span_), attr in gold_edges.items():
            attr_ = guess_edges.get((span,span_))
            # We don't care about None, None anyways...
            self.state[metric].update(attr, attr_)

        for (span, span_), attr_ in guess_edges.items():
            if (span, span_) not in gold_edges:
                self.state[metric].update(None, attr_)

    def _update_decode_span_and_edge_metrics(self, gold_spans, gold_edges, guess_spans, guess_edges):
        """
        ## apply greedy decode to guess spans/edges using cluster_graph, then update Confusion matrices
        """
        gold_spans = {span: node_lbl(attr) for span, attr in gold_spans}        
        gold_edges = {(span, span_): edge_lbl(attr) for span, span_, attr in gold_edges}
        
        guess_spans = {span: node_lbl(attr) for span, attr in guess_spans}
        guess_edges = {(span, span_): edge_lbl(attr) for span, span_, attr in guess_edges}

        # _, gold_instances = cluster_graph(gold_nodes, gold_edges)
        guess_spans, guess_edges = cluster_graph(guess_spans, guess_edges)
       
        ## need to perform matching for edges
        #matching = _match_frames(gold_instances, guess_instances)
        #cm = ConfusionMatrix([True, False], False)
        for span, attr in gold_spans.items():
            attr_ = guess_spans.get(span)
            # We don't care about None, None anyways...
            self.state["decoded_span_node"].update(attr, attr_)
        for span, attr_ in gold_spans.items():
            self.state["decoded_span_node"].update(None, attr_)

        # Match spans ind gold and guess.
        mapping, _ = _match_spans(sorted(gold_spans.keys()), sorted(guess_spans.keys()))
        gold_edges_ = {(mapping.get(span, span), mapping.get(span_, span_)): gold_edges[(span,span_)] for span, span_ in gold_edges }

        for (span, span_), attr in gold_edges.items():
            attr_ = guess_edges.get((span,span_))
            self.state["decoded_span_edge_nomatch"].update(attr, attr_)

        for (span, span_), attr in gold_edges_.items():
            attr_ = guess_edges.get((span,span_))
            self.state["decoded_span_edge"].update(attr, attr_)

        for (span, span_), attr_ in guess_edges.items():
            if (span, span_) not in gold_edges:
                self.state["decoded_span_edge_nomatch"].update(None, attr_)
            if (span, span_) not in gold_edges_:
                self.state["decoded_span_edge"].update(None, attr_)


    def update(self, gold, guess):
        """Update internal statistics for scores.
           Assumes @gold and @guess are SentenceGraphs"""
        assert isinstance(gold, SentenceGraph)
        assert isinstance(guess, SentenceGraph)
        assert gold.tokens == guess.tokens

        gold_spans = [(span, cargmax(attr)) for span, attr, _, _ in gold.nodes]
        guess_spans = [(span, cargmax(attr)) for span, attr, _, _ in guess.nodes]
        gold_edges = [(span, span_, cargmax(attr)) for span, span_, attr in gold.edges]
        guess_edges = [(span, span_, cargmax(attr)) for span, span_, attr in guess.edges]

        self._update_token_node_metrics(gold.per_token_labels, guess.per_token_labels)
        self._update_token_edge_metrics(gold.per_token_edge_labels, guess.per_token_edge_labels)

        self._update_span_node_metrics(gold_spans, guess_spans)
        self._update_span_node_metrics(gold_spans, guess_spans, with_matching=True)
        self._update_span_edge_metrics(gold_spans, gold_edges, guess_spans, guess_edges)
        self._update_span_edge_metrics(gold_spans, gold_edges, guess_spans, guess_edges, with_matching=True)
        self._update_decode_span_and_edge_metrics(gold_spans, gold_edges, guess_spans, guess_edges)

    def update_with_tokens(self, gold_ys, gold_zs, guess_ys, guess_zs, collapse_spans=False):
        """Update internal statistics for scores, where @gold, and
           @guess are (ys, zs) token sequences"""
        assert len(gold_ys.shape) == 1
        assert len(gold_zs.shape) == 2
        assert len(guess_ys.shape) == 2
        assert guess_zs == [] or len(guess_zs.shape) == 3

        L = len(gold_ys)
        assert len(guess_ys) == L

        L_ = len(gold_zs)
        assert len(gold_zs[0]) == L_
        if not collapse_spans:
            assert len(guess_zs) == L_
            assert len(guess_zs[0]) == L_
            assert L == L_

        if not collapse_spans:
            assert L == L_

        # chunk ys and zs
        gold_spans, gold_edges = chunk_tokens(gold_ys, gold_zs, with_argmax=False, collapse_spans=collapse_spans)
        guess_spans, guess_edges = chunk_tokens(guess_ys, guess_zs, with_argmax=True, collapse_spans=collapse_spans)
        guess_ys, guess_zs = guess_ys.argmax(-1), guess_zs.argmax(-1) if guess_zs != [] else []

        self._update_token_node_metrics(gold_ys, guess_ys)
        self._update_span_node_metrics(gold_spans, guess_spans)
        self._update_span_node_metrics(gold_spans, guess_spans, with_matching=True)

        self._update_token_edge_metrics(_per_token_labels(gold_edges, L), _per_token_labels(guess_edges, L))
        self._update_span_edge_metrics(gold_spans, gold_edges, guess_spans, guess_edges, with_matching=False)
        self._update_span_edge_metrics(gold_spans, gold_edges, guess_spans, guess_edges, with_matching=True)


def test_scorer_graph():
    np.random.seed(42)
    gold_ys = np.random.rand(5, len(SentenceGraph.NODE_LABELS)+1)
    gold_zs = np.random.rand(5, 5, len(SentenceGraph.EDGE_LABELS)+1)
    guess_ys = gold_ys + 0.1 * np.random.rand(5, len(SentenceGraph.NODE_LABELS)+1)
    guess_zs = gold_zs + 0.1 * np.random.rand(5, 5, len(SentenceGraph.EDGE_LABELS)+1)

    scorer = Scorer()
    scorer.update_with_tokens(gold_ys.argmax(-1), gold_zs.argmax(-1), gold_ys, gold_zs)
    assert scorer.state["token_node_all"].f1() == 1.0
    assert scorer.state["token_node_core"].f1() == 1.0
    assert scorer.state["span_node_all"].f1() == 1.0
    assert scorer.state["span_node_core"].f1() == 1.0
    assert scorer.state["token_edge"].f1() == 1.0
    assert scorer.state["span_bcubed_all"].f1() == 1.0
    assert scorer.state["span_bcubed_core"].f1() == 1.0

    scorer.clear()
    scorer.update_with_tokens(gold_ys.argmax(-1), gold_zs.argmax(-1), guess_ys, guess_zs)
    assert np.allclose(scorer.state["token_node_all"].f1(), 0.60, 5e-2)
    assert np.allclose(scorer.state["token_node_core"].f1(), 0.80, 5e-2)
    assert np.allclose(scorer.state["span_node_all"].f1(), 0.5, 5e-2)
    assert np.allclose(scorer.state["span_node_core"].f1(), 0.75, 5e-2)
    assert np.allclose(scorer.state["token_edge"].f1(), 0.90, 5e-2)
    assert np.allclose(scorer.state["span_bcubed_all"].f1(), 0.75, 5e-2)
    assert np.allclose(scorer.state["span_bcubed_core"].f1(), 0.55, 5e-2)
