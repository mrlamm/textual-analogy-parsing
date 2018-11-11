from nltk.parse.dependencygraph import DependencyGraph
from operator import itemgetter
from itertools import product
from collections import Counter, deque
from .schema import Frames, SentenceGraph, convert_frames_to_graph
from .assets.candidate_generation_vars import *
import time

def get_edges(dep_graph,address):
    '''
    # Get incoming and outgoing edges from node at address
    '''
    node=dep_graph.nodes[address]

    head=node["head"]
    rel=node["rel"]
    dependents=set([ (key,add,"down") for key in node["deps"] for add in node["deps"][key] ])

    return dependents.union(set([(rel,head,"up")]))

def compute_shortest_path_bfs(dep_graph,source,target):
    '''
    # Perform depth first search from the smaller of the addresses to the other 
    '''
    if source == target:
        return []
    elif dep_graph.contains_cycle:
        return []

    nodes=dep_graph.nodes

    queue=deque([])
    path_queue=deque([])
    visited=set()

    queue.append(source)
    path_queue.append([(source,"","")])
    visited.add(source)

    sought_path=None

    while queue:
        current_address=queue.popleft()
        current_path=path_queue.popleft()

        if current_address == target:
            sought_path = current_path
            break 

        edges=get_edges(dep_graph,current_address)

        for (rel,add,direction) in edges:
            if target not in visited:
                path=current_path[:]+[(add,rel,direction)]
                path_queue.append(path)
                queue.append(add)
                visited.add(add)

    return sought_path

def construct_dependency_graph(sentence_graph):
    """
    Given node addresses and arcs, construct dependency graph
    """
    tokens = sentence_graph.tokens
    pos = sentence_graph.features["pos"] 
    parse = sentence_graph.features["depparse"]

    dep_graph = DependencyGraph()
    
    dep_graph.remove_by_address(0)
    dep_graph.nodes[-1].update(
            {
                'tag': 'TOP',
                'address': -1,
            }
        )

    for head_address, address, relation in parse:
        node = {
                'tag': pos[address],
                'address': address,
                'head':head_address,
                'rel':relation,
                'word':tokens[address]
            }
        dep_graph.add_node(node)

    for head_address, address, _ in parse:
        dep_graph.add_arc(head_address,address)

    return dep_graph

def get_left_branching_indices(dependency_graph,head_address):
    '''
    # Get the indices to the left of and including head_address
    # These form the begin indices in candidate spans
    ''' 
    nodes = dependency_graph.nodes
    head_node=nodes[head_address]
    tag=head_node["tag"] 
    
    left_branching_tree_from_head=get_subtree_associated_with_head(dependency_graph,head_address,branch="l") # a dependency graph whose largest node address == candidate head
    left_branch_nodes=left_branching_tree_from_head.nodes

    # get the appropriate break relations for this head type
    head_type,break_relations,break_ctags = get_break_info("l",tag)

    # generate candidate indices
    if len(left_branch_nodes) == 2: # tree comprised of root node + head address 
        return "short tree", [ head_address ]
    
    else:
        # first element in the tree
        # took away minus ones
        break_indices = set([min([key for key in left_branch_nodes if key != -1 ]), head_address ])

        dependent_addresses = sorted([ (key,add) for key in left_branch_nodes[head_address]["deps"] for add in left_branch_nodes[head_address]["deps"][key] if add < head_address ],key=itemgetter(1))
        addresses_of_other_break_relations = [ add for key,add in dependent_addresses if key in break_relations or tag2ctag.get(left_branch_nodes[add]["tag"]) in break_ctags ]
        
        candidate_trees = { add:get_subtree_associated_with_head(dependency_graph,add,branch="r") for add in addresses_of_other_break_relations }

        for add in candidate_trees:
            # get the rightmost element associated with the subtree rooted at add e.g. *with* in  [ [ compared [ *with* ] ] [ [ the previous ] year ] ]  where break word is *compared*
            max_node = max(candidate_trees[add].nodes.keys())
            break_indices.add(max_node + 1)

        return head_type,break_indices

def get_right_branching_indices(dependency_graph,head_address):
    '''
    # Get the indices to the right of and including head_address
    # These form the end indices in candidate spans
    ''' 
    nodes=dependency_graph.nodes 
    head_node=nodes[head_address]
    tag=head_node["tag"]
    right_branching_tree_from_head=get_subtree_associated_with_head(dependency_graph,head_address,branch="r") # a dependency graph whose largest node address will == candidate head
    right_branch_nodes=right_branching_tree_from_head.nodes

    # get the appropriate break relations s
    head_type,break_relations,break_ctags =get_break_info("r",tag)

    # generate candidate indices
    if len(right_branch_nodes) == 2: # root node + head address 
        return "short tree",[ head_address + 1 ]
    else:
        # first element in the tree
        break_indices = set( [ max([key for key in right_branch_nodes if key != -1 and right_branch_nodes[key]["word"] not in {";",",","."}]) + 1 , head_address + 1 ])

        dependent_addresses = sorted([ (key,add) for key in right_branch_nodes[head_address]["deps"] for add in right_branch_nodes[head_address]["deps"][key] if add > head_address ],key=itemgetter(1))
        addresses_of_other_break_relations = [ add for key,add in dependent_addresses if key in break_relations or tag2ctag.get(right_branch_nodes[add]["ctag"]) in break_ctags ]

        candidate_trees = { add:get_subtree_associated_with_head(dependency_graph,add,branch="l") for add in addresses_of_other_break_relations }

        for add in candidate_trees:
            to_min = [ key for key in candidate_trees[add].nodes.keys() if key != -1 ]
            if len( to_min ) != 0:
                min_node = min(to_min)
                break_indices.add(min_node)

    return head_type,break_indices


def get_head_within_span(dependency_graph,span):
    '''
    # Given a span, find the heads 
    # i.e. find the nodes that aren't dependents of any other node in the span
    '''
    b,e = span  
    if e == b+1:
        return [b]

    nodes = [ dependency_graph.nodes[i] for i in range(b,e) ] 
    node_to_dependent_addresses = { n["address"]:get_dependent_addresses(n) for n in nodes }
    node_to_parent_addresses = {}

    ## reverse the mapping
    for child in nodes:
        child_add = child["address"]
        node_to_parent_addresses[child_add] = set()

        for parent in node_to_dependent_addresses:
            if child_add in node_to_dependent_addresses[parent]:
                node_to_parent_addresses[child_add].add(parent)

    orphans = [ n for n in node_to_parent_addresses if len(node_to_parent_addresses[n]) == 0 ]

    return orphans 


def get_dependent_addresses(node):
    '''
    # Get the addresses of dependents of a dependency graph node
    '''
    deps = node["deps"]
    return set([ add for key in deps for add in deps[key] ])

def get_subtree_associated_with_head(dependency_graph,address,branch="a"):
    '''
    # A BFS expansion of the subtree with head located at address
    '''
    nodes=dependency_graph.nodes

    visited = set()
    queue = deque([])

    queue.append(address)
    visited.add(address)

    subtree = DependencyGraph()
    subtree.remove_by_address(0)
    subtree.nodes[-1].update(
            {
                'tag': 'TOP',
                'address': -1,
            }
        )
    subtree.add_node(nodes[address])

    while queue:

        current = queue.popleft()
        deps = nodes[current]["deps"]

        all_adjacent_addresses = []

        if current == address:
            if branch == "r":
                all_adjacent_addresses = [ (rel,a) for rel in deps for a in deps[rel] if a not in visited and a > address ]
            elif branch == "l":
                all_adjacent_addresses = [ (rel,a) for rel in deps for a in deps[rel] if a not in visited and a < address ]
            else:
                all_adjacent_addresses = [ (rel,a) for rel in deps for a in deps[rel] if a not in visited  ]
        else:
            all_adjacent_addresses = [ (rel,a) for rel in deps for a in deps[rel] if a not in visited  ]

        for rel,a in all_adjacent_addresses:
            visited.add(a)
            queue.append(a)
            subtree.add_node(nodes[a])

    return subtree

def identify_break_relations(dependency_graph,span):
    """
    In candidate proposal using dependencies, we consider the left and right subtrees of 
    promising heads, and purturb them by removing subtrees at "break points"
    Break points are determined in terms of the head type (noun,verb,neither) and the 
    dependency relation by which they are connected to the head.
    """

    heads = get_head_within_span(dependency_graph,span)

    if len(heads) == 1:
        head = heads[0]
        print("head",head)
        head_type = tag2ctag.get(dependency_graph.nodes[head]["tag"])

        left_branch_nodes = get_subtree_associated_with_head(dependency_graph,head,branch="l").nodes
        right_branch_nodes = get_subtree_associated_with_head(dependency_graph,head,branch="r").nodes

        b,e = span

        left_break_relation = None
        right_break_relation = None
        head_dependent_addresses = get_dependent_addresses(dependency_graph.nodes[head])

        if head != b and b != min([ add for add in left_branch_nodes if add != -1 ]):

            if b-1 in dependency_graph.nodes[head]["deps"]:
                left_break_relation = tag2ctag.get(dependency_graph[b-1]["tag"])
            else:
                adds = [ add for add in head_dependent_addresses if add != -1 and add < b - 1 ]
                break_point = max(adds)
                
                left_break_relation = tag2ctag.get(dependency_graph.nodes[break_point]["tag"])

        if head != e - 1 and e != max([add for add in right_branch_nodes if right_branch_nodes[add]["word"] not in {";",",","."} ]) + 1:

            if e in dependency_graph.nodes[head]["deps"]:
                right_break_relation = tag2ctag.get(dependency_graph[e]["tag"])
            else:

                adds = [ add for add in head_dependent_addresses if add > e ]
                if len(adds) != 0:
                    break_point = min(adds)
                    right_break_relation = tag2ctag.get(dependency_graph.nodes[break_point]["tag"])
    
        return head_type,left_break_relation,right_break_relation

    else:
        return None,None,None


def get_break_info(branch,tag):
    """
    Return appropriate info candidate_generation_vars based on the type of head tag
    """
    ctag = tag2ctag[tag]

    if ctag == "VERB":
        if branch == "l":
            return "verbal_relation",verbal_head_left_branch_first_order_break_relations,[]
        elif branch == "r":
            return "verbal_relation",verbal_head_right_branch_first_order_break_relations,[]
    elif ctag == "NOUN" or ctag == "PROPN" or ctag == "NUM":
        if branch == "l":
            return "nominal_relation",nominal_head_left_branch_first_order_break_relations,nominal_head_left_branch_first_order_break_pos
        elif branch == "r":
            return "nominal_relation",nominal_head_right_branch_first_order_break_relations,[]
    else:
        if branch == "l":
            return "non_verbal_non_nominal_relation",non_verbal_non_nominal_head_left_branch_first_order_break_relations,[]
        else:
            return [],[],[]

    return "none",[],[]


def test_identify_break_relations():

    from .schema import Frames
    from .ud_utils import construct_dependency_graph
    import json

    sentence_graph = Frames.from_json(json.loads(open("../data/annotated_train.json").readlines()[5]))
    sentence_graph.features["dependency_graph"] = construct_dependency_graph(sentence_graph)

    sentence_graph = convert_frames_to_graph(sentence_graph)
    print(sentence_graph.nodes)
    print(" ".join(sentence_graph.tokens))
    print("\n")

    spans = set([ span for span,label,_,_ in sentence_graph.nodes if None not in label and "value" not in label and "valuem" not in label ])   

    for (begin,end) in spans:
        head_addresses = get_head_within_span(sentence_graph.features["dependency_graph"],(begin,end))
        if len(head_addresses) == 1:
            print((begin,end))
            head_address = head_addresses[0]
            print(head_address)
            print(" ".join(sentence_graph.tokens[begin:end]))
            print(sentence_graph.features["dependency_graph"].nodes[head_address]) 
            print(identify_break_relations(sentence_graph.features["dependency_graph"],(begin,end)))
            print("\n")




