"""
Some deprecated routines to parse data.
"""

import os
import logging
from enum import IntEnum
from xml.etree import ElementTree

import numpy as np
from nltk.tree import Tree
from .schema import SentenceGraph

logger = logging.getLogger(__name__)

_SPAN_LBLS = ["n", "iv", "il", "sm", "dvc", "dl", "du", "dv", "<s>", "</s>"]
class SpanLabel(IntEnum):
    NONE = 0 # n
    LABEL = 1 # iv
    TITLE = 2 # il
    DELTA = 3 # sm
    VALUE = 4 # dvc
    RELATION = 5 # dl

    _UNIT = 6 # du
    _VALUE = 7 # dv

    START = 8 # Special sequence tags.
    END = 9

    @classmethod
    def labels(cls):
        return ["-", "L", "T", "D", "V", "R"]

    @classmethod
    def parse(cls, l, convert_dvc=True):
        if convert_dvc and (l == "du" or l == "dv"):
            l = "dvc"
        if l not in _SPAN_LBLS:
            if l not in ["dr", "dm", "im", "dt", "gr", "dt"]:
                logger.warning("Could not find any label called: %s", l)
            return cls(0)
        else:
            return cls(_SPAN_LBLS.index(l))

_ARC_LBLS = ["o", "iv-iv", "iv-il", "iv-dvc", "iv-dl", "dvc-ind", "dvc-grp", "dvc-dl", "dvc-sm", "dl-dl"]
class ArcLabel(IntEnum):
    """
    Rules:
        * iv-iv: two iv spans if they have same group but different indexes.
        * iv-il: share a group (irrespective of index).
        * iv-dvc: same index and group
        * iv-dl: same group
        * dvc-dvc-ind: two dvcs with same index
        * dvc-dvc-grp: two dvcs with same group
        * dvc-sm: dvc and sm have the same index and group.
        * dl-dl: have different groups (because only one dl per group)
    """
    NONE = 0
    LABEL_LABEL = 1
    LABEL_TITLE = 2
    LABEL_VALUE = 3
    LABEL_RELATION = 4

    TITLE_LABEL = 5

    DELTA_VALUE = 6

    VALUE_LABEL = 7
    VALUE_DELTA = 8
    VALUE_VALUE_INDEX = 9
    VALUE_VALUE_GROUP = 10
    #VALUE_RELATION = 11

    RELATION_LABEL = 11
    #RELATION_VALUE = 13
    RELATION_RELATION = 12

    @classmethod
    def labels(cls):
        return ["-",
                "LL", "LT", "LV", "LR",
                "TL",
                "DV",
                "VL", "VD", "VVi", "VVg", #"VR",
                "RL", "RR",
               ]

    @classmethod
    def parse(cls, l):
        if l not in _ARC_LBLS:
            logger.warning("Could not find any arc label called: %s", l)
            return cls(0)
        else:
            return cls(_ARC_LBLS.index(l))


def parse_conll_to_graph(conll):
    """
    Consumes lines from fstream to parse a single graph.
    Each line has the following format
    token    node-label    index    group

    Edge rules:
        * iv-iv: two iv spans if they have same group but different indexes.
        * iv-il: share a group (irrespective of index).
        * iv-dvc: same index and group
        * iv-dl: same index and group
        * dvc-dvc-ind: two dvcs with same index
        * dvc-dvc-grp: two dvcs with same group
        * dvc-sm: dvc and sm have the same index and group.
        * dl-dl: have different groups (because only one dl per group)
    """
    _, tokens, lemmas, pos_tags, ner_tags, labels, groups, indices = conll

    nodes = np.array([SpanLabel.parse(l) for l in labels], dtype=np.int64)
    edges = np.zeros((len(nodes), len(nodes)), dtype=np.int64) # Note, 0 == ArcLabel.NONE
    for i, lbl in enumerate(nodes):
        if lbl == SpanLabel.NONE: continue
        for j, lbl_ in enumerate(nodes):
            if i == j or lbl_ == SpanLabel.NONE: continue
            # iv-iv: two iv spans if they have same group but different indexes.
            if lbl == SpanLabel.LABEL and lbl_ == SpanLabel.LABEL and groups[i] == groups[j] and indices[i] != indices[j]:
                edges[i, j] = ArcLabel.LABEL_LABEL
                edges[j, i] = ArcLabel.LABEL_LABEL
            # iv-il: share a group (irrespective of index).
            elif lbl == SpanLabel.LABEL and lbl_ == SpanLabel.TITLE and groups[i] == groups[j]:
                edges[i, j] = ArcLabel.LABEL_TITLE
                edges[j, i] = ArcLabel.TITLE_LABEL
            # iv-dvc: same index and group
            elif lbl == SpanLabel.LABEL and lbl_ == SpanLabel.VALUE and indices[i] == indices[j] and groups[i] == groups[j]:
                edges[i, j] = ArcLabel.LABEL_VALUE
                edges[j, i] = ArcLabel.VALUE_LABEL
            # iv-dl: same group
            elif lbl == SpanLabel.LABEL and lbl_ == SpanLabel.RELATION and groups[i] == groups[j]:
                edges[i, j] = ArcLabel.LABEL_RELATION
                edges[j, i] = ArcLabel.RELATION_LABEL
            # dvc-dvc-ind: two dvcs with same index
            elif lbl == SpanLabel.VALUE and lbl_ == SpanLabel.VALUE and indices[i] == indices[j]:
                edges[i, j] = ArcLabel.VALUE_VALUE_INDEX
                edges[j, i] = ArcLabel.VALUE_VALUE_INDEX
            # dvc-dvc-grp: two dvcs with same group (:-/ how to make
            # this mutually exclusive with dvc-dvc-ind?)
            elif lbl == SpanLabel.VALUE and lbl_ == SpanLabel.VALUE and groups[i] == groups[j]:
                edges[i, j] = ArcLabel.VALUE_VALUE_GROUP
                edges[j, i] = ArcLabel.VALUE_VALUE_GROUP
            # dvc-sm: dvc and sm have the same index and group.
            elif lbl == SpanLabel.VALUE and lbl_ == SpanLabel.DELTA and indices[i] == indices[j] and groups[i] == groups[j]:
                edges[i, j] = ArcLabel.VALUE_DELTA
                edges[j, i] = ArcLabel.DELTA_VALUE
            # dl-dl: have different groups (because only one dl per group)
            elif lbl == SpanLabel.RELATION and lbl_ == SpanLabel.RELATION and groups[i] != groups[j]:
                edges[i, j] = ArcLabel.RELATION_RELATION
                edges[j, i] = ArcLabel.RELATION_RELATION

    return SentenceGraph(tokens, nodes, edges, lemmas=lemmas, pos_tags=pos_tags, ner_tags=ner_tags)

def test_sentence_graph_parse_tsv():
    inp = """\
    New	iv	1	1
    England	iv	1	1
    Electric	iv	1	1
    ,	n	0	0
    based	n	0	0
    in	n	0	0
    Westborough	n	0	0
    ,	n	0	0
    Mass.	n	0	0
    ,	n	0	0
    had	n	0	0
    offered	dl	1	1
    $	du	1	1
    2	dv	1	1
    billion	dv	1	1
    to	n	0	0
    acquire	n	0	0
    PS	n	0	0
    of	n	0	0
    New	n	0	0
    Hampshire	n	0	0
    ,	n	0	0
    well	n	0	0
    below	n	0	0
    the	n	0	0
    $	du	1	2
    2.29	dv	1	2
    billion	dv	1	2
    value	n	0	0
    United	iv	1	2
    Illuminating	iv	1	2
    places	n	0	0
    on	n	0	0
    its	n	0	0
    bid	dl	1	2
    and	n	0	0
    the	n	0	0
    $	du	1	3
    2.25	dv	1	3
    billion	dv	1	3
    Northeast	iv	1	3
    says	n	0	0
    its	n	0	0
    bid	dl	1	3
    is	n	0	0
    worth	n	0	0
    .	n	0	0
    """
    tokens = "New England Electric , based in Westborough , Mass. , had offered $ 2 billion to acquire PS of New Hampshire , well below the $ 2.29 billion value United Illuminating places on its bid and the $ 2.25 billion Northeast says its bid is worth .".split()
    nodes  = np.array([SpanLabel.LABEL, SpanLabel.LABEL, SpanLabel.LABEL, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.RELATION, SpanLabel.VALUE, SpanLabel.VALUE, SpanLabel.VALUE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.VALUE, SpanLabel.VALUE, SpanLabel.VALUE, SpanLabel.NONE, SpanLabel.LABEL, SpanLabel.LABEL, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.RELATION, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.VALUE, SpanLabel.VALUE, SpanLabel.VALUE, SpanLabel.LABEL, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.RELATION, SpanLabel.NONE, SpanLabel.NONE, SpanLabel.NONE,])
    conll = read_conll(iter(inp.split("\n")))

    graph = parse_conll_to_graph(conll[0])


    assert graph.tokens == tokens
    assert np.allclose(graph.nodes, nodes)
    assert np.allclose(graph.edges[0].nonzero(), np.array([[11, 12, 13, 14, 29, 30, 34, 40, 43]])) # is a LABEL; 11 is 'offered', 12-14 = $2 billion, 29-30 is 'United Illuminating' and 40 is Northeast
    assert np.allclose(graph.edges[1].nonzero(), np.array([[11, 12, 13, 14, 29, 30, 34, 40, 43]])) # is a LABEL; 11 is 'offered', 12-14 = $2 billion, 29-30 is 'United Illuminating' and 40 is Northeast
    assert np.allclose(graph.edges[11].nonzero(), np.array([[0, 1, 2, 29, 30, 40]])) # The dl for this group.
    assert np.allclose(graph.edges[12].nonzero(), np.array([[0, 1, 2, 13, 14, 25, 26, 27, 37, 38, 39]])) # The iv for this group+index and other dvcs in the same group.
    assert np.allclose(graph.edges[4], np.zeros(len(tokens))) # 4 is a NONE

def _prune_null_subtrees(parse_tree):
    """
    Gets rid of any -NONE- tagged constituents and resulting empty subtrees
    """
    leaf_pos = set(parse_tree.treepositions(order="leaves"))
    nonterminals = [pos for pos in parse_tree.treepositions() if pos not in leaf_pos]
    null_pos = [pos for pos in nonterminals if parse_tree[pos].label() == "-NONE-" or len(parse_tree[pos]) == 0]
    while len(null_pos) != 0:
        del parse_tree[null_pos[0]]
        leaf_pos = set(parse_tree.treepositions(order="leaves"))
        nonterminals = [pos for pos in parse_tree.treepositions() if pos not in leaf_pos]
        null_pos = [pos for pos in nonterminals if parse_tree[pos].label() == "-NONE-" or len(parse_tree[pos]) == 0]
    return parse_tree

def test_prune_null_subtrees():
    parse_tree = Tree.fromstring('(S (NP (NNP New) (NNP England) (NNP Electric)) (, ,) (VP (VBN based) (NP (-NONE- *)) (PP-LOC-CLR (IN in) (NP (NP (NNP Westborough)) (, ,) (NP (NNP Mass.))))) (. .))')
    simple_tree = Tree.fromstring('(S (NP (NNP New) (NNP England) (NNP Electric)) (, ,) (VP (VBN based) (PP-LOC-CLR (IN in) (NP (NP (NNP Westborough)) (, ,) (NP (NNP Mass.))))) (. .))')

    simple_tree_ = _prune_null_subtrees(parse_tree)
    assert simple_tree == simple_tree_

def _process_attribute(attrib):
    """
    Converts xml attributes into their respective data types
    """
    for key in attrib:
        if key not in ["split", "range", "sign"]: # these are single value attributes.
            val = attrib[key].split(",") # integer value attributes.
            if len(val) == 1:
                val = [ int(val[0]) ]
            else:
                val[0] = val[0][1:]
                val[-1] = val[-1][:-1]
                val = sorted(set(int(v) for v in val)) # Is this even necessary?
            attrib[key] = val
    return attrib

def get_sequence_labels(sentence_node):
    """
    Extracts a sequence of tokens and labels from an XML sentence (which is whitespace tokenized) like:
    <S>
        <iv gr="[1,2]" ind="1">New England Electric</iv>
        <n>, based in Westborough , Mass. , had</n>
        <dl gr="1" ind="1">offered</dl>
        <du gr="1" ind="1">$</du>
        <dv gr="1" ind="1">2 billion</dv>
        <n>to acquire PS of New Hampshire.</n>
    </S>

    @returns: sequence of tokens, sequence of TYPE labels, GROUP labels and IND labels.
    """
    tokens, types, groups, indices = [], [], [], []

    for span in sentence_node:
        # Process the attributes.
        attrib = _process_attribute(span.attrib)

        label = span.tag
        ind = attrib.get("ind", [0])
        group = attrib.get("gr", [0])

        tokens_ = span.text.strip().split()
        tokens += tokens_
        types += [label] * len(tokens_)
        groups += [group] * len(tokens_)
        indices += [ind] * len(tokens_)

    # TODO: handle split, range, etc. annotations
    return tokens, types, groups, indices

def __parse_test_entry(i):
    i = int(i)
    if i < 10: return [i]
    else: return [int(i/10), i%10]

def test_get_sequence_labels():
    sentence = """
        <S>
        <iv gr="[1,2]" ind="1">New England Electric</iv>
        <n>, based in Westborough , Mass. , had</n>
        <dl gr="1" ind="1">offered</dl>
        <du gr="1" ind="1">$</du>
        <dv gr="1" ind="1">2 billion</dv>
        <n>to acquire PS of New Hampshire , well below the</n>
        <du gr="1" ind="2">$</du>
        <dv gr="1" ind="2">2.29 billion</dv>
        <n>value</n>
        <iv gr="[1,2]" ind="2">United Illuminating</iv>
        <n>places on its</n>
        <dl  gr="1" ind="2">bid</dl>
        <n>and the</n>
        <du gr="1" ind="3">$</du>
        <dv gr="1" ind="3">2.25 billion</dv>
        <iv gr="[1,2]" ind="3">Northeast</iv>
        <n>says its</n>
        <dl gr="1" ind="3">bid</dl>
        <n>is worth .</n>
        </S>
        """
    node = ElementTree.fromstring(sentence)
    tokens  = "New England Electric , based in Westborough , Mass. , had offered $  2  billion to acquire PS of New Hampshire , well below the $  2.29 billion value United Illuminating places on its bid and the $  2.25 billion Northeast says its bid is worth .".split()
    types   = "iv  iv      iv       n n     n  n           n n     n n   dl      du dv dv      n  n       n  n  n   n         n n    n     n   du dv   dv      n     iv     iv           n      n  n   dl  n   n   du dv   dv      iv        n    n   dl  n  n     n".split()
    groups  = "12  12      12       0 0     0  0           0 0     0 0   1       1  1  1       0  0       0  0  0   0         0 0    0     0   1  1    1       0     12     12           0      0  0   1   0   0   1  1    1       12        0    0   1   0  0     0".split()
    indices = "1   1       1        0 0     0  0           0 0     0 0   1       1  1  1       0  0       0  0  0   0         0 0    0     0   2  2    2       0     2      2            0      0  0   2   0   0   3  3    3       3         0    0   3   0  0     0".split()

    groups = [__parse_test_entry(i) for i in groups]
    indices = [__parse_test_entry(i) for i in indices]

    tokens_, types_, groups_, indices_ = get_sequence_labels(node)

    assert tokens == tokens_
    assert types == types_
    assert groups == groups_
    assert indices == indices_

def parse_xml_corpus(fstream):
    """
    @deprecated
    """
    sentences = list(ElementTree.parse(fstream).getroot())
    return [get_sequence_labels(sentence) for sentence in sentences if "skip" not in sentence.attrib and "question" not in sentence.attrib]

def __get_test_resource(path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'testdata', path)

def test_parse_corpus():
    TEST_XML_FILENAME = __get_test_resource('single_train_instance.xml')

    tokens = "Using the the NWA takeover as a benchmark , First Boston on Sept. 14 estimated that UAL was worth $ 250 to $ 344 a share based on UAL 's results for the 12 months ending last June 30 , but only $ 235 to $ 266 based on a management estimate of results for 1989 .".split()
    types = "n n n n n n n n n n n n n n n n n n dl du dv n du dv dv dv iv iv iv iv iv iv iv iv iv iv iv iv iv n n n du dv n du dv iv iv iv iv iv iv iv iv iv n".split()
    groups = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0".split()
    indices = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 2 2 0 2 2 2 2 2 2 2 2 2 2 2 0".split()
    groups = [__parse_test_entry(i) for i in groups]
    indices = [__parse_test_entry(i) for i in indices]

    with open(TEST_XML_FILENAME, 'r') as xml:
        sentences = parse_xml_corpus(xml)
    assert len(sentences) == 1
    assert len(sentences[0]) == 4
    tokens_, types_, groups_, indices_ = sentences[0]

    assert tokens == tokens_
    assert types == types_
    assert groups == groups_
    assert indices == indices_
