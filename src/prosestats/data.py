"""
Routines to read data
"""
import pdb
import logging
from xml.etree import ElementTree
from collections import Counter
from io import StringIO

from .schema import Frame, Frames, SentenceGraph, one_hot
from .util import cargmax, cmax

logger = logging.getLogger(__name__)

def parse_list_attribute(attr):
    """
    Parses list attributes like [1,2,3].
    """
    if attr[0] == "[":
        attr = [x.strip() for x in attr[1:-1].split(",")]
    else:
        attr = [attr]
    return attr

def parse_xml_to_text(fstream):
    """
    Parses the XML file (@fstream) to text.
    """
    tree = ElementTree.parse(fstream)

    for sentence in tree.findall("S"):
        if "skip" not in sentence.attrib:
            text = " ".join(child.text for child in sentence.getchildren())
            yield text

def count_members(mapping):
    # Technically converting to ints isn't required, but this is just
    # standardizing.
    all_values = sorted({int(v) for vs in mapping.values() for v in vs if v != "*"})
    assert all(i in all_values for i in range(1, len(all_values)+1)), "Missing indices: {}".format(all_values) 
    if not all_values:
        return 1
    else:
        return len(all_values)

def handle_stars(mapping):
    # Technically converting to ints isn't required, but this is just
    # standardizing.
    all_values = sorted({int(v) for vs in mapping.values() for v in vs if v != "*"})
    assert all(i in all_values for i in range(1, len(all_values)+1)), "Missing indices: {}".format(all_values) 
    if not all_values:
        all_values = [1]

    for span, vs in mapping.items():
        if vs == ["*"]:
            mapping[span] = all_values
        else:
            mapping[span] = [int(v) for v in vs]
    return mapping

def compute_stats_xml(sentence):
    """
    Parses a single <S> sentence, extracting all frames, instances and
    manner attributes.
    """
    # Get the number of frames from each xml's <gr>
    graph = SentenceGraph()
    tokens = []

    frames = {}
    instances = {}

    for child in sentence.getchildren():
        # Get the number of tokens from this element.
        tokens_ = child.text.split()
        span = (len(tokens), len(tokens) + len(tokens_))
        tokens += tokens_

        tag = child.tag.lower()
        if tag != "n":
            frames[span] = parse_list_attribute(child.attrib["gr"]) if child.attrib.get("gr") else ["*"]
            instances[span] = parse_list_attribute(child.attrib["ind"]) if child.attrib.get("ind") else ["*"]

            attr = tag
            manner = child.attrib.get("manner", None)
            sign = child.attrib.get("sign", None)

            graph.nodes.append([span, one_hot(attr), one_hot(manner), one_hot(sign)])
    # sort the nodes
    graph.nodes.sort()

    assert tokens

    # Replace stars.
    frames, instances = count_members(frames), count_members(instances)

    return frames, instances

def parse_xml_sentence(sentence):
    """
    Parses a single <S> sentence, extracting all frames, instances and
    manner attributes.
    """
    # Get the number of frames from each xml's <gr>
    graph = SentenceGraph()
    tokens = []

    frames = {}
    instances = {}

    for child in sentence.getchildren():
        # Get the number of tokens from this element.
        tokens_ = child.text.split()
        span = (len(tokens), len(tokens) + len(tokens_))
        tokens += tokens_

        tag = child.tag.lower()
        if tag != "n":
            frames[span] = parse_list_attribute(child.attrib["gr"]) if child.attrib.get("gr") else ["*"]
            instances[span] = parse_list_attribute(child.attrib["ind"]) if child.attrib.get("ind") else ["*"]

            attr = tag
            manner = child.attrib.get("manner", None)
            sign = child.attrib.get("sign", None)

            graph.nodes.append([span, one_hot(attr), one_hot(manner), one_hot(sign)])
    # sort the nodes
    graph.nodes.sort()

    assert tokens

    # Replace stars.
    frames, instances = handle_stars(frames), handle_stars(instances)

    def _in(xs, ys):
        return any(x in ys for x in xs)

    for span, attr, _, _ in graph.nodes:
        attr = cargmax(attr)
        frame = frames[span]
        index = instances[span]
        
        if attr == "value":
            assert len(frame) == 1 and len(index) == 1, "Value spans frames or indices: {} at {}".format(tokens[slice(*span)], span)

        for span_, attr_, _, _ in graph.nodes:
            if span >= span_: continue
            attr_ = cargmax(attr_)

            frame_ = frames[span_]
            index_ = instances[span_]

            if attr_ == "value":
                assert len(frame_) == 1 and len(index_) == 1, "Value spans frames or indices: {} at {}".format(tokens[slice(*span_)], span_)

            # Draw equivalence edges.
            if attr == attr_ and _in(frame, frame_) and _in(index, index_):
                assert attr != "value", "Value equivalence: {} = {} at {} = {}".format(tokens[slice(*span)], tokens[slice(*span_)], span, span_)
                graph.edges.append((span, span_, one_hot("equivalence")))
            # Manners in the same frame are always equivalent (important
            # for the ILP)
            elif attr == attr_ and _in(frame, frame_) and attr == "manner":
                assert attr != "value", "Value equivalence: {} = {} at {} = {}".format(tokens[slice(*span)], tokens[slice(*span_)], span, span_)
                graph.edges.append((span, span_, one_hot("equivalence")))
            # Draw analogy edges.
            elif attr == attr_ and _in(frame, frame_) and not _in(index, index_):
                graph.edges.append((span, span_, one_hot("analogy")))
            # Draw fact edges.
            elif (attr == "value" or attr_ == "value") and _in(frame, frame_) and _in(index, index_):
                graph.edges.append((span, span_, one_hot("fact")))

    graph.tokens.extend(tokens)
    # TODO: ensure graph is valid.
    # graph.ensure_valid()
    return graph

def parse_xml(fstream):
    """
    Parsers a sequence of frames from an XML.
    NOTE: returns a generator.
    """
    tree = ElementTree.parse(fstream)

    for sentence_ix, sentence in enumerate(tree.findall("S")):
        if "skip" in sentence.attrib: continue
        try:
            yield parse_xml_sentence(sentence)
        except ValueError as e:
            logger.error("Could not parse sentence %d (%s):\n %s", sentence_ix, e, ElementTree.tostring(sentence).decode("utf-8"))
        except AssertionError as e:
            logger.error("Could not parse sentence %d (%s):\n %s", sentence_ix, e, ElementTree.tostring(sentence).decode("utf-8"))

def test_parse_xml():
    xml = """
<DOC>
<S id='1'>
<AGENT gr="1" ind="1">New England Electric</AGENT>
<n>, based in Westborough , Mass. , had</n>
<QUANT gr="1">offered</QUANT>
<VALUE gr="1" ind="1">$ 2 billion</VALUE>
<n>to acquire</n> 
<THEME gr="1">PS of New Hampshire</THEME>
<n>, well below the</n>
<VALUE gr="1" ind="2">$ 2.29 billion</VALUE>
<n>value</n>
<AGENT gr="1" ind="2">United Illuminating</AGENT>
<n>places on its</n>
<QUANT gr="1">bid</QUANT>
<n>and the</n>
<VALUE gr="1" ind="3">$ 2.25 billion</VALUE>
<AGENT gr="1" ind="3">Northeast</AGENT>
<n>says its</n>
<QUANT gr="1">bid</QUANT>
<n>is worth</n>
<n>.</n>
</S>
</DOC>
    """
    tokens = "New England Electric , based in Westborough , Mass. , had offered $ 2 billion to acquire PS of New Hampshire , well below the $ 2.29 billion value United Illuminating places on its bid and the $ 2.25 billion Northeast says its bid is worth .".split()

    graph = next(parse_xml(StringIO(xml)))
    assert graph.tokens == tokens
    assert len(graph.nodes) == 10
    assert len([cargmax(lbl) == "equivalence" for _, _, lbl in graph.edges]) == 3 # quant equivalence
    assert len([cargmax(lbl) == "analogy" for _, _, lbl in graph.edges]) == 6 # value and agent analogy
    assert len([cargmax(lbl) == "fact" for _, _, lbl in graph.edges]) == 3 + 3 + 3*3 # value-agent, value-theme, value-quant

    print(graph.as_json())

def test_parse_xml_to_text():
    xml = """
<DOC>
<S id='1' manners="[equals]">
<LABEL gr="1" ind="1">New England Electric</LABEL>
<n>, based in Westborough , Mass. , had</n>
<THEME gr="1">offered</THEME>
<UNIT gr="1" ind="1">$</UNIT>
<VALUE gr="1" ind="1">2 billion</VALUE>
<THEME_MOD gr="1">to acquire PS of New Hampshire</THEME_MOD>
<n>, well below the</n>
<UNIT gr="1" ind="2">$</UNIT>
<VALUE gr="1" ind="2">2.29 billion</VALUE>
<n>value</n>
<LABEL gr="1" ind="2">United Illuminating</LABEL>
<n>places on its</n>
<THEME gr="1">bid</THEME>
<n>and the</n>
<UNIT gr="1" ind="3">$</UNIT>
<VALUE gr="1" ind="3">2.25 billion</VALUE>
<LABEL gr="1" ind="3">Northeast</LABEL>
<n>says its</n>
<THEME gr="1">bid</THEME>
<n>is worth</n>
<n>.</n>
</S>
</DOC>
    """
    txt = "New England Electric , based in Westborough , Mass. , had offered $ 2 billion to acquire PS of New Hampshire , well below the $ 2.29 billion value United Illuminating places on its bid and the $ 2.25 billion Northeast says its bid is worth ."
    txt_ = next(parse_xml_to_text(StringIO(xml)))

    assert txt_ == txt

class StandoffWriter(object):
    def __init__(self, txt_stream, ann_stream):
        self.txt_stream = txt_stream
        self.ann_stream = ann_stream

        self.entity_idx = 0
        self.relation_idx = 0
        self.char_offset = 0

    def write(self, graph):
        # First binarize the graph.
        graph = graph.binarize()

        token_offsets = []
        for token in graph.tokens:
            token_offsets.append((self.char_offset, self.char_offset + len(token)))
            self.txt_stream.write(token)
            self.txt_stream.write(" ")
            self.char_offset += len(token) + 1
        self.txt_stream.write("\n")
        self.char_offset += 1

        entity_ids = {}
        for span, attr, _, _ in sorted(graph.nodes, key=lambda x:[0]):
            if cmax(attr) == 0.0: continue
            attr = cargmax(attr)
            if attr is None or attr == "null": continue

            self.ann_stream.write("T{}\t{} {} {}\t{}\n".format(self.entity_idx, attr, token_offsets[span[0]][0], token_offsets[span[1]-1][1], " ".join(graph.tokens[span[0]:span[1]])))
            entity_ids[span] = self.entity_idx
            self.entity_idx += 1

        for span, span_, attr in graph.edges:
            if 'span' in attr:
                del attr['span']
            if not attr or cmax(attr) == 0.0: continue
            attr = cargmax(attr)
            if attr is None or attr == "null": continue

            self.ann_stream.write("R{}\t{} Arg1:T{} Arg2:T{}\n".format(self.relation_idx, attr, entity_ids[span], entity_ids[span_]))
            self.relation_idx += 1

def test_standoff_writer():
    graph = SentenceGraph("This is a test sentence .".split())
    graph.nodes.append(((0,1), Counter({"DET":0.9, "ART": 0.1}),  Counter(), Counter()))
    graph.nodes.append(((1,2), Counter({"null":0.9, "ART": 0.1}), Counter(), Counter()))
    graph.nodes.append(((3,5), Counter({"NN":0.9, "ART": 0.1}), Counter(), Counter()))

    graph.edges.append(((0,1), (1,2), Counter({None: 0.9})))
    graph.edges.append(((0,1), (3,5), Counter({"frame": 0.9})))

    txt, ann = StringIO(), StringIO()
    writer = StandoffWriter(txt, ann)
    writer.write(graph)

    assert txt.getvalue() == "This is a test sentence . \n"
    assert ann.getvalue() == "T0\tDET 0 4\tThis\nT1\tNN 10 23\ttest sentence\nR0\tframe Arg1:T0 Arg2:T1\n"
