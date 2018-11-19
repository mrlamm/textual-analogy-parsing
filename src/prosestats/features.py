"""
Featurizers
"""

import pdb
import os
import re
import json
from collections import Iterable

from corenlp import CoreNLPClient
from .util import TokenSentence
from .schema import overlaps

def _dep_to_list(dep_graph):
    """
    Converts a protobuf dependency to a graph.
    """
    ret = [(edge.source-1, edge.target-1, edge.dep) for edge in dep_graph.edge]
    for root in reversed(dep_graph.root):
        ret.insert(0, (-1, root-1, "root"))
    return ret

CORENLP_ANNOTATORS = "tokenize lemma pos ner depparse"
CORENLP_PROPERTIES = {
    "tokenize.whitespace": True,
    "ssplit.isOneSentence": True,
    "timeout": '50000'
    }
CORENLP_PROPERTIES_TEST = {
    "ssplit.isOneSentence": True,
    "timeout": '50000'
    }

class Featurizer():
    def __init__(self, annotators=None, properties=None):
        with open(os.path.join(os.path.dirname(__file__), "assets", "regexes.json")) as f:
            self.regexer = RegexFeaturizer(json.load(f))
        self.annotators = annotators or CORENLP_ANNOTATORS 
        self.properties = properties or CORENLP_PROPERTIES
        self.client = CoreNLPClient(self.annotators, properties=self.properties, endpoint="http://localhost:9012")

    def __enter__(self):
        self.client.__enter__()
        return self

    def __exit__(self, *args):
        self.client.__exit__(*args)

    def _apply_features(self, obj, ann=None):
        """
        Adds features to a graph.
        """
        if "features" not in obj:
            obj["features"] = {}
        if ann:
            assert len(ann.sentence) == 1
            sentence = ann.sentence[0]
            assert len(sentence.token) == len(obj["tokens"])
            obj["features"]["lemma"] = [t.lemma for t in sentence.token]
            obj["features"]["pos"] = [t.pos for t in sentence.token]
            obj["features"]["ner"] = [t.ner for t in sentence.token]
            obj["features"]["depparse"] = _dep_to_list(sentence.enhancedPlusPlusDependencies)
            assert len({tail for _, tail, _ in obj["features"]["depparse"]}) == len(sentence.token)
            child_to_head, head_to_child, path_length, next_in_path, distance_to_next_token, distance_from_prev_token = compute_dependency_paths(obj)
            obj["features"]["dep_child_to_head"] = child_to_head
            obj["features"]["dep_head_to_child"] = head_to_child
            obj["features"]["dep_path_lengths"] = path_length
            obj["features"]["dep_traceback"] = next_in_path
            obj["features"]["dep_dist_to_next"] = distance_to_next_token
            obj["features"]["dep_dist_from_prev"] = distance_from_prev_token
        if self.regexer:
            obj["features"]["regexes"] = self.regexer.featurize(obj["tokens"])
            obj["features"]["typed_values"] = self.regexer.featurize_unit_spans(obj["tokens"])

    def featurize_graph(self, obj):
        ann = self.client.annotate(" ".join(obj["tokens"]))
        self._apply_features(obj, ann)
        return obj

    def featurize_text(self, text):
        ann = self.client.annotate(text)
        assert len(ann.sentence) == 1
        sentence = ann.sentence[0]

        obj = {
            "tokens": [t.word for t in sentence.token],
            }
        self._apply_features(obj, ann)
        return obj

class RegexFeaturizer(object):
    def __init__(self, patterns):
        self.patterns = []
        for pattern, name in patterns:
            pattern = pattern\
                .replace("#NUMBER_WORD", r"(One|one|Two|two|Three|three|Four|four|Five|five|Six|six|Seven|seven|Eight|eight|Nine|nine|Ten|ten|Eleven|eleven|Twelve|twelve|Thirteen|thirteen|Fourteen|fourteen|Fifteen|fifteen|Sixteen|sixteen|Seventeen|seventeen|Eighteen|eighteen|Nineteen|nineteen|Twenty|twenty|Twenty-?five|twenty-?five)")\
                .replace("#NUMBER",  r"([0-9]+)(,[0-9]{3})*(.[0-9]+)?")\
                .replace("#XILLION", r"(thousand|million|billion|trillion)s?")\
                .replace("#FRAC", r"(([0-9]+ )?([0-9]+\\/[0-9]+))")
            pattern = pattern

            self.patterns.append((re.compile(pattern), name))

    def featurize(self, tokens):
        ret = []
        txt = TokenSentence(tokens)
        for pattern, name in self.patterns:
            for m in pattern.finditer(txt.text):
                obj = {k: txt.to_tokens(m.span(k)) for k in m.groupdict()}
                # This was a false match.
                if any(x is None for x in obj.values()):
                    continue

                obj["_pattern"] = name
                ret.append(obj)
        return ret

    def featurize_unit_spans(self, tokens):
        ret = ["O"]*len(tokens)
        txt = TokenSentence(tokens)

        spans_found = {}
        for pattern, name in self.patterns:
            for m in pattern.finditer(txt.text):
                obj = {k: txt.to_tokens(m.span(k)) for k in m.groupdict()}
                # This was a false match.
                if any(x is None for x in obj.values()):
                    continue

                begin = min(obj[k][0] for k in obj)
                end = max(obj[k][1] for k in obj)

                spans_found[(begin,end)] = name
        
        # only keep non-overlapping spans  
        discarded = set()      
        for span in spans_found:
            for span_ in spans_found: 
                if span != span_ and overlaps(span,span_):           
                    if span_[0] < span[0] and span_[1] >= span[1]:
                        discarded.add(span)
                    elif span_[0] == span[0] and span_[1] > span[1]:
                        discarded.add(span)

        for span in spans_found:
            if span not in discarded:
                pattern = spans_found[span] 
                begin,end = span
                ret[begin:end] = [pattern] * (end - begin)

        return ret

def test_regex_featurizer():
    featurizer = RegexFeaturizer([(r"(?P<unit>\$) (?P<value>[0-9]+ ?((million|billion|trillion))?)", "$")])
    tokens = "Stocks rose $ 3 billion over the last year".split()
    feats = featurizer.featurize(tokens)
    assert feats == [{"unit": (2,3), "value":(3,5), "_pattern": "$"}]

def compute_dependency_paths(featurized_obj):
    if featurized_obj["features"]["depparse"] == None:
        return None,None,None,None
    else:
        head_to_child, child_to_head = {},{}
        
        for head_position, child_position, relation in featurized_obj["features"]["depparse"]:
            if child_position in child_to_head:
                child_to_head[child_position][head_position] = relation
            else:
                child_to_head[child_position] = { head_position:relation }

            if head_position != -1:
                if head_position in head_to_child:
                    head_to_child[head_position][child_position] = relation
                else:
                    head_to_child[head_position] = { child_position:relation }

        path_length = { i:[ 10000 ]*len(child_to_head) for i in range(len(child_to_head)) }
        next_in_path = { i:[None]*len(child_to_head) for i in range(len(child_to_head)) }

        ### path to itself is length zero
        for i in range(len(child_to_head)):
            path_length[i][i] = 0
            next_in_path[i][i] = i

        ### path to any elems that are connected via a deprel is length 1
        for child in child_to_head:
            for head in child_to_head[child]:
                if head != -1:
                    # symmetric; ignore dummy index
                    path_length[head][child] = 1    
                    path_length[child][head] = 1
                    next_in_path[head][child] = child
                    next_in_path[child][head] = head

        ### compute shortest paths for all pairs
        for k in range(len(child_to_head)):
            for i in range(len(child_to_head)):
                for j in range(len(child_to_head)):
                    if path_length[i][j] > path_length[i][k] + path_length[j][k]:
                        path_length[i][j] = path_length[i][k] + path_length[j][k]
                        next_in_path[i][j] = next_in_path[i][k]

        distance_to_next_token = [ path_length[i][i+1] for i in range(0,len(child_to_head)-1) ]
        distance_to_next_token = distance_to_next_token + [1]
        distance_from_prev_token = [ path_length[i][i-1] for i in range(1,len(child_to_head)) ]
        distance_from_prev_token = [1] + distance_from_prev_token

        return child_to_head, head_to_child, path_length, next_in_path , distance_to_next_token, distance_from_prev_token
