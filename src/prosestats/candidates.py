from .schema import SentenceGraph, convert_frames_to_graph
from .assets.candidate_generation_vars import *
from .ud_utils import get_right_branching_indices, get_left_branching_indices , get_head_within_span
from itertools import product

def generate_candidates(sentence_graph):
    
    # something going on with dependencies, see if it goes away with new parses
    cands = generate_candidates_using_dependencies(sentence_graph)
    cands = cands.union(generate_candidates_using_regex(sentence_graph))
    cands = cands.union(generate_candidates_using_ner(sentence_graph))


    #reverse = sum([ 1.0 for cand in cands if cand[1] < cand[0]])

    return [ cand for cand in cands if cand[1] > cand[0] ]

def generate_candidates_using_dependencies(sentence_graph):
    """
    Generates candidate spans by expanding subtrees from promising heads
    Returns a set of (begin,end) span tuples 
    """

    dependency_graph=sentence_graph.features["dependency_graph"]
    nodes = dependency_graph.nodes
    
    # heuristic filter based on head relation
    # candidate_heads=[ address for address in nodes if nodes[address]["rel"] in promising_head_relations and address >= 0 ]
    candidate_heads = [ address for address in nodes if address >= 0 ] 
    span_candidates=set()
    for candidate_head in candidate_heads:
        #print(candidate_head)
        _,right_dependent_indices = get_right_branching_indices(dependency_graph,candidate_head)
        _,left_dependent_indices = get_left_branching_indices(dependency_graph,candidate_head)
        for pair in product(left_dependent_indices,right_dependent_indices):
            span_candidates.add(pair)

    return span_candidates

def generate_candidates_using_regex(sentence_graph):
    """
    use numerical regex to generate possible value/valuem spans
    """
    regex=sentence_graph.features.get("regexes")
    spans = []
    for elem in regex:
        these_spans = [ elem.get("value"), elem.get("value_split"), elem.get("unit"), elem.get("unit_split") ]
        these_spans = [ s for s in these_spans if s != None ]
        spans.extend(these_spans)


    return set([ tuple(s) for s in spans ])

def generate_candidates_using_ner(sentence_graph):
    """
    consider any contiguous named entity span a possible candidate
    """
    ner=sentence_graph.features.get("ner")

    contiguous_ner_spans = []
    if ner != None:
        prev = None
        for i in range(0,len(ner)):
            curr = ner[i]
            if curr != "O":
                if curr == prev:
                    span = contiguous_ner_spans[-1]
                    span[1] = i+1
                else:
                    contiguous_ner_spans.append([i,i+1])

            prev = curr

    return set([tuple(elem) for elem in contiguous_ner_spans])

def test_generate_candidates_using_dependencies():

    from .schema import Frames
    from .ud_utils import construct_dependency_graph
    import json

    sentence_graph = Frames.from_json(json.loads(open("../data/annotated_train.json").readlines()[0]))

    sentence_graph.features["dependency_graph"] = construct_dependency_graph(sentence_graph)

    sentence_graph = convert_frames_to_graph(sentence_graph)
    print(sentence_graph.nodes)

    candidate_spans = generate_candidates_using_dependencies(sentence_graph)
    spans = set([ span for span,label,_,_ in sentence_graph.nodes if None not in label or "value" not in label ])   

    for begin,end in candidate_spans:
        if begin != end:
            print( " ".join(sentence_graph.tokens[begin:end]) )


def compute_dependency_generation_recall():

    from .schema import Frames
    from .ud_utils import construct_dependency_graph
    import json

    n = 0.0
    d = 0.0 

    sentence_graphs = [ Frames.from_json(json.loads(line)) for line in open("../data/annotated_train.json").readlines() ] 
    for i,sentence_graph in enumerate(sentence_graphs):
        sentence_graph.features["dependency_graph"] = construct_dependency_graph(sentence_graph)
        candidate_spans = generate_candidates_using_dependencies(sentence_graph)
        sentence_graph = convert_frames_to_graph(sentence_graph)
                
        gold_spans = set([ span for span,label,_,_ in sentence_graph.nodes if None not in label and "value" not in label and "valuem" not in label ])   

        intersection = candidate_spans.intersection(gold_spans)
        n += len(intersection)
        d += len(gold_spans)

        if len(intersection) != len(gold_spans):
            print( "_____________________________________")
            print("missing some spans in sentence ", i)
            print(" ".join(sentence_graph.tokens ))
            for pair in gold_spans:
                if pair not in intersection:
                    b,e = pair
                    head_relation = get_head_within_span(sentence_graph.features["dependency_graph"],pair)
                    print("head relation",sentence_graph.features["dependency_graph"].nodes[head_relation[0]]["rel"])
                    print( " ".join(sentence_graph.tokens[b:e]))

    print( n * 1.0 / d )


def test_generate_candidates_using_ner():

    from .schema import Frames
    from .ud_utils import construct_dependency_graph
    import json

    sentence_graph = Frames.from_json(json.loads(open("../data/annotated_train.json").readlines()[0]))

    sentence_graph = convert_frames_to_graph(sentence_graph)

    candidate_spans = generate_candidates_using_ner(sentence_graph)

    print( "ner", [ pair for pair in enumerate(sentence_graph.features["ner"]) ])
    for begin,end in candidate_spans:
        print( begin,end )
        print( " ".join(sentence_graph.tokens[begin:end]) )


def test_generate_candidates_using_regex():

    from .schema import Frames
    from .ud_utils import construct_dependency_graph
    import json

    sentence_graph = Frames.from_json(json.loads(open("../data/annotated_train.json").readlines()[0]))

    sentence_graph = convert_frames_to_graph(sentence_graph)

    candidate_spans = generate_candidates_using_regex(sentence_graph)

    print( candidate_spans )

