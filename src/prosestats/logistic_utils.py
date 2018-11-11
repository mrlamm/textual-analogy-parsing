from collections import Counter
from .assets.candidate_generation_vars import tag2ctag
from nltk import bigrams
from .ud_utils import get_head_within_span, compute_shortest_path_bfs

P_LEMMA = "LEMMA:"
P_POS = "POS:"
P_CPOS = "CPOS:"
P_NER = "NER:"
P_CASE = "CASE:"
P_DEPREL = "DEPREL:"
P_BIGR = "BIGR:"
P_REGPAT = "REGEX_PATTERN:"
P_REG = "REGEX:"
CASES = ["aa", "AA", "Aa", "aA"]
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK = "UNK"
NUM = "###"

def check_span_matches_regex(span,match):
	rev = { tuple(match[key]):key for key in match if key != "_pattern" }
	if span in rev:
		return {rev[span]:1.0},{match.get("_pattern"):1.0} 
	else:
		return {},{}

"""
Span feature functions here
"""
def ner_features(sentence_graph,span):
	b,e=span
	ner = sentence_graph.features.get("ner",[])
	
	return dict(Counter( [ P_NER + word for word in ner[b:e] ] ))

def lemma_features(sentence_graph,span):
	b,e=span
	tokens = sentence_graph.tokens
	
	return dict(Counter( [ P_LEMMA + word.lower() for word in tokens[b:e] ] ))

def pos_features(sentence_graph,span):
	b,e=span
	pos = sentence_graph.features.get("pos",[])
	
	return dict(Counter( [ P_POS + word.lower() for word in pos[b:e] ] ))

def coarse_pos_features(sentence_graph,span):
	b,e=span
	pos = [ tag2ctag[tag] for tag in sentence_graph.features.get("pos",[]) ]

	return dict(Counter( [ P_CPOS + word.lower() for word in pos[b:e] ] ))

def bigram_features(sentence_graph,span):
	b,e=span
	tokens = sentence_graph.tokens
	bigs=[ word.lower() + "_" + word_.lower() for word,word_ in bigrams(tokens[b:e]) ]

	return dict(Counter( [ P_BIGR + word.lower() for word in bigs[b:e] ] ))

def deprel_features(sentence_graph,span):
	b,e=span
	dependency_graph = sentence_graph.features.get("dependency_graph")

	nodes = dependency_graph.nodes
	ret = []
	if dependency_graph != None:
		nodes = dependency_graph.nodes
		ret = [ dependency_graph.nodes[i]["rel"] for i in range(b,e) ]
	
	return dict(Counter( [ P_DEPREL + word for word in ret ] ) ) 

def regex_features(sentence_graph,span):

	regexes = sentence_graph.features.get("regexes")

	ret = Counter()
	for match in regexes:
		m,m_ = check_span_matches_regex(span,match)
		ret.update( Counter( {P_REG+key:m[key] for key in m }) )
		ret.update( Counter( {P_REGPAT+key:m_[key] for key in m_ }) )

	return ret

"""
Edge feature functions here 
"""

def compute_dice_coeff(set_1,set_2):
	return (2.0*len(set_1.intersection(set_2)))/(len(set_1) + len(set_2))

def compute_jaccard_coeff(set_1,set_2):
	return (1.0*len(set_1.intersection(set_2)))/(len(set_1.union(set_2)))


def ner_overlap(sentence_graph,edge):
	node,node_ = edge

	ner = sentence_graph.features.get("ner",[])

	ner_n = set(ner[node[0]:node[1]])
	ner_n_ = set(ner[node_[0]:node_[1]])
	
	ret = { "ner_JAC":compute_jaccard_coeff(ner_n,ner_n_), 
			"ner_DICE":compute_dice_coeff(ner_n,ner_n_) }

	return ret


def pos_overlap(sentence_graph,edge):
	node,node_ = edge

	pos = sentence_graph.features.get("pos",[])

	pos_n = set(pos[node[0]:node[1]])
	pos_n_ = set(pos[node_[0]:node_[1]])
	
	ret = { "pos_JAC":compute_jaccard_coeff(pos_n,pos_n_), 
			"pos_DICE":compute_dice_coeff(pos_n,pos_n_) }

	return ret

def deprel_overlap(sentence_graph,edge):
	node,node_ = edge

	dependency_graph = sentence_graph.features.get("dependency_graph",[])
	nodes = dependency_graph.nodes

	deprel_n = set([ nodes[i]["rel"] for i in range(node[0],node[1]) ])
	deprel_n_ = set([ nodes[i]["rel"] for i in range(node_[0],node_[1]) ])

	ret = { "pos_JAC":compute_jaccard_coeff(deprel_n,deprel_n_), 
			"pos_DICE":compute_dice_coeff(deprel_n,deprel_n_) }	

	return ret

def first_lemma_match(sentence_graph,edge):
	b,b_ = edge[0][0],edge[1][0]

	ret = {}
	if sentence_graph.tokens[b].lower() == sentence_graph.tokens[b_].lower():
		ret[ "first_lemma_match" ] = 1.0

	return ret

def dependency_path_features(sentence_graph,edge):
	node,node_ = edge

	dependency_graph = sentence_graph.features.get("dependency_graph")

	heads = get_head_within_span(dependency_graph,node)
	heads_ = get_head_within_span(dependency_graph,node_)

	head = None
	head_ = None

	ret = {}
	if len(heads) != 0 and len(heads_) != 0:
		head = heads[0]
		head_ = heads_[0]

	elif len(heads) != 0 and len(heads_) == 0:
		head = heads[0]
		head_ = node_[0]

	elif len(heads) == 0 and len(heads_) != 0:
		head = node[0]
		head_ = heads_[0]

	else: 
		head = node[0]
		head_ = node_[0]

	if dependency_graph.nodes[head]["rel"] == dependency_graph.nodes[head_]["rel"]:
	    ret["head_relation_match"] = 1.0 	

	path = compute_shortest_path_bfs(dependency_graph,head,head_)

	if len(path) != 0:
		## length of shortest dependency path
		ret["dep_path_path_length"] = len(path) 
		# updown counts
		ret = { **ret , **dict(Counter([ "ud_" + ud for _,_,ud in path ])) }	
		# relations on path
		ret = { **ret, **dict(Counter([ "path_rel_" + rel for _,rel,_ in path ])) }

	else:
		ret["no_dep_path!"] = 1.0

	return ret

def edge_regex_pattern_match(sentence_graph,edge):
	node,node_ = edge

	regexes = sentence_graph.features["regexes"]

	ret_gr,ret_pat = set(),set()
	ret_gr_,ret_pat_ = set(),set()


	for match in regexes:
		
		m_gr,m_pat = check_span_matches_regex(node,match) 
		m_gr_,m_pat_ = check_span_matches_regex(node_,match)

		ret_gr = ret_gr.union(set(m_gr.keys()))
		ret_pat = ret_pat.union(set(m_pat.keys()))
		ret_gr_ = ret_gr_.union(set(m_gr_.keys()))
		ret_pat_ = ret_pat_.union(set(m_pat_.keys()))

	gr = ret_gr.intersection(ret_gr_)
	pat = ret_pat.intersection(ret_pat_)

	ret = Counter()

	if len(gr) > 0 and len(pat) > 0:
		ret.update(Counter({"regex_pattern_and_gr_match":1.0}))
	elif len(gr) > 0:
		ret.update(Counter({"regex_gr_match":1.0}))
	elif len(pat) > 0: 
		ret.update(Counter({"regex_pattern_match":1.0}))	
	else:
		if len(ret_pat) > 0 and len(ret_pat_) > 0:
			ret.update(Counter({"distinct_patterns":1.0}))	
		elif len(ret_pat) > 0 or len(ret_pat_) > 0:
			ret.update(Counter({"only_one_endpoint_regex_match":1.0}))
		else:
			ret.update(Counter({"no_pattern_match":1.0}))

	return ret

span_feature_functions = [ ner_features, lemma_features, pos_features, coarse_pos_features, bigram_features, deprel_features, regex_features ]
edge_feature_functions = [ ner_overlap, pos_overlap, deprel_overlap, first_lemma_match , dependency_path_features , edge_regex_pattern_match , edge_regex_pattern_match ]


