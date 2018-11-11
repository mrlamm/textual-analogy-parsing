"""
Data format
"""
import pdb
import json
import logging
from numbers import Number
from collections import defaultdict, Counter

from .util import cargmax, cmax, get_chunks, cnormalize

logger = logging.getLogger(__name__)

# TAGS
# 'AGENT': 213,
# 'CAUSE': 51,
# 'CONDITION': 13,
# 'CO_QUANT': 85,
# 'LOCATION': 224,
# 'MANNER': 548,
# 'QUANT': 1146,
# 'QUANT_MOD': 22,
# 'REFERENCE_TIME': 37,
# 'SOURCE': 103,
# 'THEME': 1072,
# 'THEME_MOD': 19,
# 'TIME': 1243,
# 'VALUE': 3256,
# 'WHOLE': 218,
# 'n': 6535

def _to_counter(dct):
    if dct is None or dct == "null": return Counter()
    ret = Counter()
    for k, v in dct.items():
        if v == 0.:
            v = 10e-7
        if k == "null":
            k = None
        ret[k] = v
    return ret

def span_lt(span, span_):
    return span[0] < span_[0] or (span[0] == span_[0] and span[1] < span_[1])

def ordered(span, span_):
    if span_lt(span, span_):
        return span, span_
    else:
        return span_, span

def overlaps(span, span_):
    return span[1] > span_[0] and span[0] < span_[1]

def overlap(span, span_):
    return max(0, min(span[1], span_[1]) - max(span[0], span_[0]))

def noverlap(span, span_):
    l = min(span[1], span_[1]) - max(span[0], span_[0]) # the overlapping region
    L =  max(span[1] - span[0], span_[1] - span_[0])
    return max(0, l)/L

def is_mod(attr):
    return attr in ["quant_mod", "theme_mod", "unit_restriction"]

def is_unmod(attr, attr_):
    if attr == "label_mod":
        return attr_ == "label"
    elif attr == "theme_mod":
        return attr_ == "theme"
    elif attr == "unit_restriction":
        return attr_ == "unit"
    else:
        return False

def is_core(attr):
    return attr in ["label", "value", "valuem", "theme", "unit", "possessor", "type"]

def graph_argmax(nodes, edges):
    nodes = [(span, cargmax(attr), cargmax(sign), cargmax(manner)) for span, attr, sign, manner in nodes]
    edges = [(span, span_, cargmax(attr)) for span, span_, attr in edges]

    return nodes, edges

def chunk_tokens(nodes, edges, with_argmax=False, collapse_spans=False):
    """
    If collapsed_spans, then the edges already use the spans that have been collapsed.
    """
    if with_argmax:
        nodes = nodes.argmax(axis=1)
    chunks = get_chunks(nodes)
    chunked_edges = []

    if collapse_spans:
        assert len(chunks) == len(edges)

    for i, (span, _) in enumerate(chunks):
        for j, (span_, _) in enumerate(chunks):
            if collapse_spans:
                attr = edges[i,j]
            else:
                attr = edges[span[0]:span[1], span_[0]:span_[1]].sum((0,1))
            if with_argmax:
                attr = attr.argmax()
            edge = (span, span_, attr)
            chunked_edges.append(edge)
    return chunks, chunked_edges

def chunk_tokens_with_distribution(nodes, edges, collapse_spans=False):
    chunks = get_chunks(nodes.argmax(axis=1))

    if collapse_spans:
        assert len(chunks) == len(edges)

    ret_nodes = []
    for span, _ in chunks:
        attr = nodes[span[0]:span[1]].mean(0)
        ret_nodes.append((span, attr))

    ret_edges = []
    for i, (span, _) in enumerate(chunks):
        for j, (span_, _) in enumerate(chunks):
            if collapse_spans:
                attr = edges[i, j]
            else:
                attr = edges[span[0]:span[1], span_[0]:span_[1]].mean((0,1))
            edge = (span, span_, attr)
            ret_edges.append(edge)
    return ret_nodes, ret_edges

def node_lbl(lbl):
    if isinstance(lbl, Number):
        return SentenceGraph.NODE_LABELS[int(lbl)-1] if lbl > 0 else None
    else:
        assert lbl is None or lbl in SentenceGraph.NODE_LABELS
        return lbl

def edge_lbl(lbl):
    if isinstance(lbl, Number):
        return SentenceGraph.EDGE_LABELS[int(lbl)-1] if lbl > 0 else None
    else:
        assert lbl is None or lbl in SentenceGraph.EDGE_LABELS
        return lbl

class Span(object):
    """
    A span holds a reference to a list of tokens and an offset within them.
    """
    def __init__(self, start, end, tokens=None):
        self.start = start
        self.end = end
        self.tokens = tokens

    def __getitem__(self, value):
        if value == 0:
            return self.start
        elif value == 1:
            return self.end
        else:
            raise IndexError("Spans only have 2 elements")

    def as_json(self):
        return (self.start, self.end)

    def __lt__(self, other):
        assert isinstance(other, Span)
        return self.start < other.start or (self.start == other.start and self.end < other.end)

    def __eq__(self, other):
        assert isinstance(other, Span)
        return self.start == other.start and self.end == other.end

    def __str__(self):
        if self.tokens:
            return " ".join(self.tokens[self.start:self.end])
        else:
            return "[{},{})".format(self.start,self.end)

    def __repr__(self):
        return "<Span: [{},{})>".format(self.start,self.end)

    def overlaps(self, other):
        assert isinstance(other, Span)
        return self.end > other.start and self.start < other.end

    def overlap(self, other):
        assert isinstance(other, Span)
        return max(0, min(self.end, other.end) - max(self.start, other.start))

def test_span():
    x = Span(2, 4, tokens="This is a test sentence .".split())
    assert x.start == 2
    assert x.end == 4
    assert x[0] == 2
    assert x[1] == 4
    assert str(x) == "a test"
    assert repr(x) == "<Span: [2,4)>"
    assert x.overlaps(Span(3,5))
    assert x.overlap(Span(3,5)) == 1

class Instance(object):
    SPANS = {"value", "valuem", "label", "label_mod"}
    VALUES = {"sign"}
    ATTRS = set.union(SPANS, VALUES)

    def __init__(self, obj=None, parent=None):
        self.obj = obj or {}
        self.parent = parent
        self.tokens = parent and parent.tokens

    def __repr__(self):
        return repr(self.obj)

    def as_json(self):
        ret = {attr: value.as_json() for attr, value in self.obj.items() if attr in self.SPANS}
        ret.update({attr: value for attr, value in self.obj.items() if attr in self.VALUES})
        return ret

    @classmethod
    def from_json(cls, obj, parent=None):
        ret = cls(parent=parent)
        for attr, value in obj.items():
            if attr in Instance.SPANS:
                ret.obj[attr] = Span(*value, tokens=ret.tokens)
            elif attr in Instance.VALUES:
                ret.obj[attr] = value
            else:
                logger.error("Ignoring invalid attribute %s in instance %s", attr, obj)
        return ret

    def __setattr__(self, key, value):
        if key in Instance.SPANS:
            self.obj[key] = Span(*value, tokens=self.tokens)
        elif key in Instance.VALUES:
            self.obj[key] = value
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key):
        if key in Instance.ATTRS:
            return self.obj.get(key)
        else:
            raise KeyError("Invalid attribute: {}".format(key))

    def insert(self, attribute, value):
        if attribute in self.ATTRS:
            setattr(self, attribute, value)
        else:
            raise ValueError("Invalid attribute for instance: {} (value:{})", attribute, value)

    def __len__(self):
        return len(self.obj)

    def ensure_valid(self):
        if not self: return # We're ok with absolutely empty instances.
        if not self.value:
            raise ValueError("VALUE must be set (instance: {})".format(self))
        if not self.label:
            raise ValueError("LABEL must be set (instance: {})".format(self))
        if self.sign and self.sign not in ["+", "-"]:
            raise ValueError("Invalid sign: {} (instance: {})".format(self.sign, self))

    def is_valid(self):
        try:
            self.ensure_valid()
            return True
        except ValueError:
            return False

    @property
    def spans(self):
        for key, span in self.obj.items():
            if key not in self.SPANS: continue
            yield (span.as_json(), key)

def test_instance():
    instance = Instance()
    instance.insert("value", [4,5])
    instance.insert("label", [6,8])
    assert instance.is_valid()
    assert instance.as_json() == {
        "label": (6,8),
        "value": (4,5),
        }

def test_instance_spans():
    instance = Instance()
    instance.insert("value", (4,5))
    instance.insert("label", (6,8))
    assert set(instance.spans) == {
        ((4,5), "value"),
        ((6,8), "label"),
    }

class Frame(object):
    SPANS = {"unit", "unit_restriction", "theme", "theme_mod", "possessor", "type"}
    VALUES = {"manner"}
    ATTRS = set.union(SPANS, VALUES)

    def __init__(self, obj=None, parent=None):
        raise ValueError("FRAMES ARE DEAD")
        self.parent = None
        self.tokens = parent and parent.tokens
        self.obj = obj or defaultdict(list)
        self.instances = []

    def __repr__(self):
        return repr(self.obj)

    @classmethod
    def from_json(cls, obj, parent=None):
        ret = cls(parent=parent)

        for key, value in obj.items():
            if key in Frame.SPANS:
                ret.obj[key].extend([Span(*v, tokens=ret.tokens) for v in value])
            elif key in Frame.VALUES:
                ret.obj[key] = value
            elif key == "instances":
                for obj_ in value:
                    ret.instances.append(Instance.from_json(obj_, parent=ret))
            else:
                logger.error("Ignoring invalid attribute %s in frame %s", key, obj)

        # Sort lists for comparability.
        for key in ret.obj:
            if key in Frame.SPANS:
                ret.obj[key].sort()
        return ret

    def as_json(self):
        ret = {attr: sorted([v.as_json() for v in value]) for attr, value in self.obj.items() if attr in self.SPANS}
        ret.update({attr: value for attr, value in self.obj.items() if attr in self.VALUES})
        ret["instances"] = [inst.as_json() for inst in self.instances]
        return ret

    def __setattr__(self, key, value):
        if key in Frame.SPANS:
            self.obj[key].append(Span(*value, tokens=self.tokens))
        elif key in Frame.VALUES:
            self.obj[key] = value
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key):
        if key in Frame.ATTRS:
            return self.obj.get(key)
        else:
            raise KeyError("Invalid attribute: {}".format(key))

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.instances[key]

    def __len__(self):
        return len(self.instances)

    def __iter__(self):
        return iter(self.instances)

    def insert(self, attribute, value, instances=None):
        """
        Inserts an attribute value into this frame.
        """
        if attribute in Frame.ATTRS:
            setattr(self, attribute, value)
        elif attribute in Instance.ATTRS:
            if not instances:
                raise ValueError("Expected some instances to insert attribute {} (value={})".format(attribute, value))

            while max(instances) >= len(self.instances):
                self.instances.append(Instance(parent=self))
            for i in instances:
                self.instances[i].insert(attribute, value)
        else:
            raise ValueError("{} is not a valid attribute".format(attribute))

    def ensure_valid(self):
        if len(self.instances) < 2:
            raise ValueError("Must have atleast 2 instances for a valid frame, have {} (frame: {})".format(len(self.instances), self))
        if not any(key in self.obj for key in Frame.ATTRS if key != "instances"):
            raise ValueError("Must have atleast one frame attribute (frame: {})".format(self))
        if self.theme_mod and not self.theme:
            raise ValueError("THEME must be set for THEME_MOD (frame: {})".format(self))
        if self.manner and self.manner not in ["equals", "changes", "delta", "count"]:
            raise ValueError("Invalid manner: {} (frame: {})".format(self.manner, self))
        for instance in self.instances:
            instance.ensure_valid()
            if self.manner and self.manner in ["changes"] and not instance.valuem:
                raise ValueError("When manner is {}, instances must have a VALUEM: {}".format(self.manner, instance))

    def is_valid(self):
        try:
            self.ensure_valid()
            return True
        except ValueError:
            return False

    @property
    def spans(self):
        for key, spans in self.obj.items():
            if key not in self.SPANS: continue
            for span in spans:
                yield (span.as_json(), key, None)

        for instance_ix, instance in enumerate(self.instances):
            for span, label in instance.spans:
                yield (span, label, instance_ix)

def test_frame():
    f = Frame()
    f.insert("unit",  (0,1))
    f.insert("theme", (1,2))
    f.insert("value", (2,3), instances=[0])
    f.insert("value", (3,4), instances=[1])
    f.insert("label", (4,5), instances=[0])
    f.insert("label", (5,6), instances=[1])
    assert f.is_valid()

def test_frame_spans():
    f = Frame()
    f.insert("unit",  (0,1))
    f.insert("theme", (1,2))
    f.insert("value", (2,3), instances=[0])
    f.insert("value", (3,4), instances=[1])
    f.insert("label", (4,5), instances=[0])
    f.insert("label", (5,6), instances=[1])
    assert set(f.spans) == {
        ((0,1), "unit",  None),
        ((1,2), "theme", None),
        ((2,3), "value", 0),
        ((3,4), "value", 1),
        ((4,5), "label", 0),
        ((5,6), "label", 1),
    }

class Frames(object):
    def __init__(self, tokens=None, frames=None, features=None):
        """
        """
        raise ValueError("FRAMES ARE DEAD")
        self.tokens = tokens or []
        self.frames = frames or []
        self.features = features or {}

    def as_json(self):
        return {
            "tokens": self.tokens,
            "frames": [f.as_json() for f in self.frames],
            "nFrames": self.n_frames,
            "nInstances": self.n_instances,
            "features": self.features,
            }

    def __repr__(self):
        return repr({
            "tokens": self.tokens,
            "nFrames": self.n_frames,
            "nInstances": self.n_instances,
            "frames": self.frames,
            "features": self.features,
            })

    def __str__(self):
        """A pretty print of the frame"""
        ret = ""
        for i, frame in enumerate(self.frames):
            ret += "Frame {}:\n".format(i)
            for key, spans in frame.obj.items():
                if key == "manner":
                    ret += " - {} = {}\n".format(key, spans)
                else:
                    ret += " - {} = {}\n".format(key, "; ".join(" ".join(self.tokens[slice(*span)]) for span in spans))
            if frame.instances:
                ret += " - Instances:\n"
                for j, instance in enumerate(frame.instances):
                    ret += " -- {}\n".format(j)
                    for key, span in instance.obj.items():
                        ret += " --- {} = {}\n".format(key, " ".join(self.tokens[slice(*span)]))
            ret += "\n"
        return ret

    @property
    def n_frames(self):
        return len(self.frames)

    @property
    def n_instances(self):
        return max(len(f.instances) for f in self.frames) if self.frames else 0

    @classmethod
    def from_json(cls, obj):
        assert "frames" in obj, "Invalid format"
        ret = Frames(obj["tokens"], [Frame.from_json(frame) for frame in obj["frames"]], obj.get("features"))
        return ret

    def add_frame(self, frame):
        self.frames.append(frame)

    def insert(self, attribute, value, frames, instances=None):
        """
        Insert an attribute-value into the frame.
        """
        assert frames

        while max(frames) >= len(self.frames):
            self.frames.append(Frame(parent=self))

        for frame_ix in frames:
            self.frames[frame_ix].insert(attribute, value, instances)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.frames[key]

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def ensure_valid(self):
        if not self.frames:
            raise ValueError("Must have atleast 1 frame (frames: {})".format(self))
        if self.tokens == 0:
            raise ValueError("Tokens must not be empty (frames: {})".format(self))
        for frame in self.frames:
            frame.ensure_valid()

    def is_valid(self):
        try:
            self.ensure_valid()
            return True
        except ValueError:
            return False

    @property
    def spans(self):
        span_map = defaultdict(list)

        for frame_ix, frame in enumerate(self.frames):
            for (span, label, instance_ix) in frame.spans:
                span_map[span].append((label, frame_ix, instance_ix))
        for span, tx in span_map.items():
            labels, frame_ixs, instance_ixs = zip(*tx)
            assert labels[0] == labels[-1]
            yield (span, labels[0], frame_ixs, instance_ixs)

def test_frames():
    f = Frames("Apples and bananas are the future of food .".split())
    f.insert("unit",  (1,2), [0])
    f.insert("theme", (2,3), [0])
    f.insert("value", (3,4), [0], instances=[0])
    f.insert("value", (4,5), [0], instances=[1])
    f.insert("value", (5,6), [1], instances=[0])
    f.insert("value", (6,7), [1], instances=[1])
    f.insert("unit",  (7,8), [1])
    f.insert("theme", (8,9), [1])
    f.insert("label", (9,10), [0,1], instances=[0])
    f.insert("label", (0,1), [0,1], instances=[1])

    assert f.is_valid()

def test_frames_spans():
    f = Frames("Apples and bananas are the future of food .".split())
    f.insert("unit",  (1,2), [0])
    f.insert("theme", (2,3), [0])
    f.insert("value", (3,4), [0], instances=[0])
    f.insert("value", (4,5), [0], instances=[1])
    f.insert("value", (5,6), [1], instances=[0])
    f.insert("value", (6,7), [1], instances=[1])
    f.insert("unit",  (7,8), [1])
    f.insert("theme", (8,9), [1])
    f.insert("label", (9,10), [0,1], instances=[0])
    f.insert("label", (0,1), [0,1], instances=[1])
    assert set(f.spans) == {
        ((1,2), "unit",  (0,),   (None,)),
        ((2,3), "theme", (0,),   (None,)),
        ((3,4), "value", (0,),   (0,)),
        ((4,5), "value", (0,),   (1,)),
        ((5,6), "value", (1,),   (0,)),
        ((6,7), "value", (1,),   (1,)),
        ((7,8), "unit",  (1,),   (None,)),
        ((8,9), "theme", (1,),   (None,)),
        ((9,10),"label", (0,1),  (0,0)),
        ((0,1), "label", (0,1),  (1,1)),
    }

class SentenceGraph(object):
    """
    Represents a sentence as a list of tokens, each with node labels and
    edges between them.

    @tokens: list of words.
    @nodes: array of node labels (non-zero entries correspond to labels).
    @edges: adjacency matrix (non-zero entries correspond to labels).
    """
    #NODE_LABELS = ['agent', 'cause', 'condition', 'co_quant', 'location', 'manner', 'quant', 'quant_mod', 'reference_time', 'source', 'theme', 'theme_mod', 'time', 'value', 'whole',]
    NODE_LABELS = ['agent', 'cause', 'co_quant', 'location', 'manner', 'quant', 'source', 'theme', 'time', 'value', 'whole',]
    CORE_LABELS = ["value",]
    EDGE_LABELS = ["analogy", "fact", "equivalence", "span"]


    def __init__(self, tokens=None, nodes=None, edges=None, features=None):
        self.tokens = tokens or []
        self.nodes = nodes or []
        self.edges = edges or []
        self.features = features or {} # Additional features

    @property
    def per_token_labels(self):
        ret = [None for _ in self.tokens]
        for span, attr, _, _ in self.nodes:
            attr = cargmax(attr)
            for idx in range(*span):
                ret[idx] = attr
        return ret

    @property
    def per_token_edge_labels(self):
        ret = [[None for _ in self.tokens] for _ in self.tokens]
        for span, _, _, _ in self.nodes:
            for i in range(*span):
                for j in range(*span):
                    ret[i][j] = ret[j][i] = "span"
        for span, span_, attr in self.edges:
            attr = cargmax(attr)
            for i in range(*span):
                for j in range(*span_):
                    ret[i][j] = ret[j][i] = attr
        return ret

    @property
    def per_edge_labels(self):
        span_ixs = {span: i for i, (span, *_) in enumerate(self.nodes)}
        ret = [[None for _ in self.nodes] for _ in self.nodes]
        for span, span_, attr in self.edges:
            ret[span_ixs[span]][span_ixs[span_]] = cargmax(attr)
        return ret

    @classmethod
    def from_json(cls, obj):
        tokens = obj["tokens"]

        nodes = [(tuple(span), _to_counter(attr), _to_counter(sign), _to_counter(manner)) for span, attr, sign, manner in obj["nodes"]  if max(attr.values()) > 0.]
        edges = [(tuple(span), tuple(span_), _to_counter(attr)) for span, span_, attr in obj["edges"] if max(attr.values()) > 0.]
        return cls(tokens, nodes, edges, obj.get("features"))

    def as_json(self):
        ret = {
            "tokens": self.tokens,
            "nodes": self.nodes,
            "edges": self.edges,
            }
        if self.features:
            ret["features"] = self.features

        return ret

    def binarize(self):
        """
        Constructs a new graph by binarizing this one. For each span, we
        choose the label with the highest score.
        Q: can I also enforce no overlapping spans?
        - basically, I choose every set of spans with a label that is
          overlapping and choose the highest scoring of this bunch?
        - I then remove any edges that don't match.
        """

        nodes = []
        for span, attr, sign, manner in self.nodes:
            (attr, attr_score), = attr.most_common(1)
            if attr is None:
                # Skip unlabeled nodes
                continue
            elif attr == "value":
                sign = cargmax(sign)
                manner = cargmax(manner)
            else:
                sign = None
                manner = None

            candidate = (span, attr, sign, manner, attr_score)
            for i, (span_, _, _, _, attr_score_) in enumerate(nodes):
                if overlaps(span, span_):
                    # Pick the highest scoring of the overlapping
                    # clusters
                    if attr_score > attr_score_:
                        nodes[i] = candidate
                    break
            else:
                # Create a new cluster
                nodes.append(candidate)

        # Post process nodes.
        node_spans = {span: attr for span, attr, _, _, _ in nodes}
        nodes = [(span, Counter({attr:1.0}), Counter({sign:1.0}), Counter({manner:1.0}),) for (span, attr, sign, manner, _) in nodes]

        edges = []
        for (span, span_, attr) in self.edges:
            # Ok, now drop all edges that are inconsistent with our
            # chosen spans
            if span not in node_spans or span_ not in node_spans:
                continue
            attr = cargmax(attr)
            # Only accept instance edges between possible instance spans
            if attr == "instance" and not (
                    node_spans[span] in Instance.SPANS and node_spans[span_] in Instance.SPANS):
                attr = None
            edges.append((span, span_, Counter({attr:1.0}),))

        ret = SentenceGraph(self.tokens, nodes, edges)
        return ret

    def as_clusters(self):
        return cluster_graph(*graph_argmax(self.nodes, self.edges))

    def with_predictions(self, yhs, zhs, collapse_spans=False):
        """
        Constructs a new graph with just the yhs and zhs predictions.
        """
        ret = SentenceGraph(self.tokens)
        nodes, edges = chunk_tokens_with_distribution(yhs, zhs, collapse_spans=collapse_spans)
        for span, attr in nodes:
            attr = Counter({node_lbl(i): float(v) for i, v in enumerate(attr)})
            ret.nodes.append((span, attr, Counter({None:.0}), Counter({None:.0})))
        for span, span_,attr in edges:
            if span < span_:
                attr = Counter({edge_lbl(i): float(v) for i, v in enumerate(attr)})
                # Get rid of the span tag.
                del attr["span"]
                cnormalize(attr)
                ret.edges.append((span, span_, attr))
        return ret


def test_graph_binarize():
    graph = SentenceGraph("This is a test".split())
    graph.nodes.append( ((0,1), Counter({"label": 0.4, "theme": 0.6}), Counter({None: 1.0}), Counter({None: 1.0}),) )
    graph.nodes.append( ((1,2), Counter({"label": 0.6, "theme": 0.4}), Counter({None: 1.0}), Counter({None: 1.0}),) )
    graph.nodes.append( ((1,3), Counter({"value": 0.9, "theme": 0.1}), Counter({None: 1.0}), Counter({None: 1.0}),) )

    # Add some edges now.
    graph.edges.append( ((0,1), (1,2), Counter({"fact": 0.6, "None": 0.4})) )
    graph.edges.append( ((0,1), (1,3), Counter({"fact": 0.6, "None": 0.4})) )
    graph.edges.append( ((1,2), (1,3), Counter({"fact": 0.6, "None": 0.4})) )

    graph_ = graph.binarize()
    assert graph_.tokens == graph.tokens
    assert len(graph_.nodes) == 2
    assert all(cmax(attr) == 1.0 for _, attr, _, _ in graph_.nodes)
    assert len(graph_.edges) == 1
    assert all(cmax(attr) == 1.0 for _, _, attr in graph_.edges)

def test_graph_labels():
    graph = SentenceGraph("This is a test".split())
    graph.nodes.append( ((0,1), Counter(["theme"]), Counter({None: 1.0}), Counter({None: 1.0}),) )
    graph.nodes.append( ((1,2), Counter(["theme"]), Counter({None: 1.0}), Counter({None: 1.0}),) )
    graph.nodes.append( ((1,3), Counter(["value"]), Counter({None: 1.0}), Counter({None: 1.0}),) )

    # Add some edges now.
    graph.edges.append( ((0,1), (0,1), Counter(["span"])))
    graph.edges.append( ((0,1), (1,2), Counter(["equiv"])))
    graph.edges.append( ((0,1), (1,3), Counter(["fact"])))

    lbls = graph.per_edge_labels
    assert len(lbls) == 3
    assert len(lbls[0]) == 3
    assert lbls[0][0] == "span"
    assert lbls[0][1] == "equiv"
    assert lbls[0][2] == "fact"
    assert lbls[1][0] is None
    assert lbls[1][1] is None
    assert lbls[1][2] is None


def one_hot(label):
    return Counter({label: 1.0})

def convert_frames_to_graph(frames):
    """
    Converts a frame into a graph.
    """
    ret = SentenceGraph(tokens=frames.tokens, features=frames.features)
    # Each span in the frame is a distinct node in the graph, with a
    # one_hot encoding of the label.
    nodes = {}
    for span, label, frame_ixs, instance_ixs in frames.spans:
        if label == "value":
            frame_ix, instance_ix = frame_ixs[0], instance_ixs[0]
            sign_label = frames[frame_ix][instance_ix].sign
            manner_label = frames[frame_ix].manner
        else:
            sign_label = None
            manner_label = None
        nodes[span] = one_hot(label), one_hot(sign_label), one_hot(manner_label)
    ret.nodes.extend([(span, *nodes[span]) for span in sorted(nodes.keys())])

    edges = {}
    for frame in frames:
        for span, attr, instance_ix in frame.spans:
            assert instance_ix is None or attr in Instance.SPANS
            for span_, attr_, instance_ix_ in frame.spans:
                assert instance_ix_ is None or attr_ in Instance.SPANS
                if span >= span_: continue

                if instance_ix is not None and instance_ix_ is not None:
                    if instance_ix == instance_ix_:
                        label = "instance"
                # (b) for instance nodes, only draw a frame arc if they
                # share attributes (no label-value arcs across frames).
                    elif instance_ix != instance_ix_ and attr == attr_:
                        label = "frame"
                    else:
                        continue
                else:
                    label = "frame"

                # Edge conditions:
                # (a) only draw a x_mod edge with x
                if is_mod(attr) or is_mod(attr_) and not (is_unmod(attr, attr_) or is_unmod(attr_, attr)):
                    continue
                edges[span, span_] = one_hot(label),

    ret.edges.extend([(span, span_, *edges[span, span_]) for span, span_ in sorted(edges.keys())])
    return ret

def test_convert_frames_to_graph():
    frames = Frames.from_json({
        'tokens': ['New', 'England', 'Electric', ',', 'based', 'in', 'Westborough', ',', 'Mass.', ',', 'had', 'offered', '$', '2', 'billion', 'to', 'acquire', 'PS', 'of', 'New', 'Hampshire', ',', 'well', 'below', 'the', '$', '2.29', 'billion', 'value', 'United', 'Illuminating', 'places', 'on', 'its', 'bid', 'and', 'the', '$', '2.25', 'billion', 'Northeast', 'says', 'its', 'bid', 'is', 'worth', '.'],
        'frames': [{
            'theme': [[11, 12], [34, 35], [43, 44]],
            'instances': [
                {'value': [13, 15], 'label': [0, 3]},
                {'value': [26, 28], 'label': [29, 31]},
                {'value': [38, 40], 'label': [40, 41]}],
            'theme_mod': [[15, 21]],
            'manner': 'equals',
            'unit': [[12, 13], [25, 26], [37, 38]]
            }],
        'nInstances': 3,
        'nFrames': 1
        })

    graph = convert_frames_to_graph(frames)
    assert graph.tokens == frames.tokens

    assert any(span == (11, 12) and attr["theme"] == 1.0 and sign[None] == 1.0 and manner[None] == 1.0 for (span, attr, sign, manner) in graph.nodes)
    assert any(span == (11, 12) and span_ == (13, 15) and attr["frame"] == 1.0 for (span, span_, attr) in graph.edges)
    assert any(span == (0, 3) and span_ == (13, 15) and attr["instance"] == 1.0 for (span, span_, attr) in graph.edges)
    assert len({(span, span_) for span, span_, _ in graph.edges}) == len(graph.edges)

    #assert len(graph.edges) == sum(frame_size * (frame_size-1)/2 for frame_size in [sum(1 for _ in frame.spans) for frame in frames])

def cluster_graph(nodes, edges):
    """
    Clusters a graphs edges in to frames and instances.
    """
    value_nodes = [span for span, attr, _, _ in nodes if attr == "value"]
    nodes = {span: attr for span, attr, _, _ in nodes if attr != "value"}
    frame_links = {(span, span_) for span, span_, attr in edges if attr == "frame"}
    instance_links = {(span, span_) for span, span_, attr in edges if attr == "instance"}

    # Create the frame and instance skeletons using the value nodes.
    frames = []
    while value_nodes:
        span = value_nodes.pop()
        for frame in frames:
            if any(ordered(span, span_) in frame_links for span_ in frame):
                frame.append(span)
                break
        else:
            # Add a new frame.
            frames.append([span])
    # Each of these values sits in a separate instance.
    instances = [[[span,] for span in frame] for frame in frames]

    # Ok, go through every node and assign them to frames.
    # The number of iterations required to ensure all connected
    # components are covered.
    N = len(nodes)
    for _ in range(N):
        for span in nodes:
            for frame_ix, frame in enumerate(frames):
                if span not in frame and any(ordered(span, span_) in frame_links for span_ in frame):
                    frame.append(span)
                if span not in frame and any(ordered(span, span_) in instance_links for span_ in frame):
                    frame.append(span)

                if nodes[span] not in Instance.SPANS: continue
                for instance in instances[frame_ix]:
                    if span not in instance and any(ordered(span, span_) in instance_links for span_ in instance):
                        instance.append(span)

    # Now, anything in nodes that isn't in a frame is thrown into a junk
    # pile.
    junk_frame = [span for span in nodes if not any(span in frame for frame in frames)]
    junk_instances = [span for span in junk_frame if nodes[span] in Instance.SPANS]

    if junk_frame:
        frames.append(junk_frame)
        instances.append([junk_instances,])

    frames = [sorted(frame) for frame in frames]
    instances = [[sorted(instance) for instance in frame] for frame in instances]
    return frames, instances

def _construct_frame(nodes, frame_spans, instances_spans):
    node_map = {span: (cargmax(attr), cargmax(sign), cargmax(manner)) for (span, attr, sign, manner) in nodes}
    ret = defaultdict(list)

    manners = Counter()
    for instance_spans in instances_spans:
        instance = {}
        for span in instance_spans:
            attr, sign, manner = node_map[span]
            instance[attr] = span
            if sign:
                instance["sign"] = sign
            if manner:
                manners[manner] += 1
        ret["instances"].append(instance)
    ret["manner"] = cargmax(manners) if manners else None

    for span in frame_spans:
        if any(span in instance_span for instance_span in instances_spans):
            continue
        attr, _, _ = node_map[span]
        ret[attr].append(span)
    return ret

def convert_graph_to_frames(graph):
    """
    Converts a graph into a collection of frames, choosing the maximal
    label for each node.
    NOTE: this conversion does NOT enforce consistency.
    """
    graph_ = graph.binarize()

    # Cluster the edges into frames.
    frames_spans, instances_spans = cluster_graph(*graph_argmax(graph_.nodes, graph_.edges))
    # For each frame, gather the attributes and instances.
    frames = [_construct_frame(graph_.nodes, f, i) for f, i in zip(frames_spans, instances_spans)]

    ret = Frames(graph_.tokens)
    for frame in frames:
        ret.add_frame(Frame.from_json(frame, parent=ret))

    return ret

def test_convert_graph_to_frames():
    frames = Frames.from_json({
        'tokens': ['New', 'England', 'Electric', ',', 'based', 'in', 'Westborough', ',', 'Mass.', ',', 'had', 'offered', '$', '2', 'billion', 'to', 'acquire', 'PS', 'of', 'New', 'Hampshire', ',', 'well', 'below', 'the', '$', '2.29', 'billion', 'value', 'United', 'Illuminating', 'places', 'on', 'its', 'bid', 'and', 'the', '$', '2.25', 'billion', 'Northeast', 'says', 'its', 'bid', 'is', 'worth', '.'],
        'frames': [{
            'theme': [[11, 12], [34, 35], [43, 44]],
            'instances': [
                {'value': [13, 15], 'label': [0, 3]},
                {'value': [26, 28], 'label': [29, 31]},
                {'value': [38, 40], 'label': [40, 41]}],
            'theme_mod': [[15, 21]],
            'manner': 'equals',
            'unit': [[12, 13], [25, 26], [37, 38]]
            }],
        'nInstances': 3,
        'nFrames': 1
        })
    graph = convert_frames_to_graph(frames)
    frames_ = convert_graph_to_frames(graph)

    assert frames.tokens == frames_.tokens
    assert len(frames.frames) == len(frames_.frames)
    assert frames[0].theme == frames_[0].theme

    # TODO: assert comparison with instance values, etc.

def load_frames(fstream):
    raise ValueError("FRAMES ARE DEAD")
    for line in fstream:
        frames = Frames.from_json(json.loads(line))
        yield frames

def load_graphs(fstream):
    for line in fstream:
        graph = SentenceGraph.from_json(json.loads(line))
        yield graph

def save_frames(frames, fstream):
    raise ValueError("FRAMES ARE DEAD")
    for f in frames:
        json.dump(f.as_json(), fstream)
        fstream.write("\n")

def save_graphs(graphs, fstream):
    for g in graphs:
        json.dump(g.as_json(), fstream)
        fstream.write("\n")

def prune_graph(graph):
    ret = SentenceGraph(tokens=graph.tokens, features=graph.features)
    ret.nodes.extend([node for node in graph.nodes if cargmax(node[1]) in SentenceGraph.NODE_LABELS])
    spans = [span for span, _, _, _ in ret.nodes]
    ret.edges.extend([edge for edge in graph.edges if edge[0] in spans and edge[1] in spans])
    return ret
