## New cluster graph method
from itertools import product

def span_lt(span, span_):
    return span[0] < span_[0] or (span[0] == span_[0] and span[1] < span_[1])

def ordered(span, span_):
    if span_lt(span, span_):
        return span, span_
    else:
        return span_, span

def check_edge_typing_constraints(edge,attr,nodes):
    
    span,span_ = edge
    if attr == "fact":
        if nodes.get(span) == "value" and nodes.get(span_) != "value":
            return True
        elif nodes.get(span) != "value" and nodes.get(span_) == "value":
            return True
        else:
            return False
    elif attr == "analogy":
        if nodes.get(span) == nodes.get(span_):
            return True
        else: 
            return False
    else:
        if nodes.get(span) == nodes.get(span_) and nodes.get(span) != "value":
            return True
        else: 
            return False

def cluster_graph(nodes, edges):
    """
    ## applies greedy decoding to a graphs edges/nodes to make sure it satisfies 
        analogy constraints
    """
    
    ## assuming
    analogy = "analogy"
    equiv = "equivalence"
    fact = "fact"
    
    ##### first set of constraints starts here
    # first enforce edge typing constraints
    edges = { edge:edges[edge] for edge in edges if check_edge_typing_constraints(edge,edges[edge],nodes) }   
    # identify value nodes  that do not have alo fact edge and alo analogy edge 
    discard_values = { span for span in nodes if nodes[span] == "value" and not ( any(edges[edge] for edge in edges if span in edge and edges[edge] == "analogy" ) and any(edges[edge] for edge in edges if span in edge and edges[edge] == "fact" )) }   
    # discard invalid value nodes (i.e ones that will never make it into an analogy cluster) and other fully disconnected nodes
    nodes = { span:nodes[span] for span in nodes if any([ span in edge for edge in edges]) and span not in discard_values }
    edges = { (span,span_):edges[(span,span_)] for span,span_ in edges if span in nodes and span_ in nodes }

    # value_analogy_cliques = transitive_agglomerate(nodes, edges, "value", "analogy")
    # union of values in dictionary will partition the set of non-value spans
    # equivalence_cliques = { attr:transitive_agglomerate(nodes, edges, attr , "analogy") for attr in }

    ## Constraint 3: unique facts 
    value_spans = { span for span in nodes if nodes[span] == "value" }
    for span in value_spans:
        fact_edges_with_span = { edge for edge in edges if span in edge and edges[edge] == "fact" }
        for edge,edge_ in product(fact_edges_with_span,fact_edges_with_span):
            if edge != edge_:
                sibling = [ s for s in edge if s != span ][0]
                sibling_ = [ s for s in edge if s != span ][0]

                # confirm constraint isn't already satisfied
                if edges.get(ordered(sibling,sibling_)) != "equivalence":
                    if nodes[sibling] == nodes[sibling_]:
                        edges[ordered(sibling,sibling_)] = "equivalence"

    # reprune for disconnected nodes
    nodes = { span:nodes[span] for span in nodes if any([ span in edge for edge in edges]) }
        
    ## Constraint 4: enforce equivalence 
    ## first get equivalence clusters
    final_clusters = []
    all_node_clusters = [ {node} for node in nodes ]
    for i in range(0,len(all_node_clusters)):
        cluster = all_node_clusters[i]
        merged = False
 
        for j in range(i+1,len(all_node_clusters)):
            cluster_ = all_node_clusters[j]
            if any( edges.get(ordered(span,span_)) == "equivalence" for (span,span_) in product(cluster,cluster_) ):
                all_node_clusters[j] = cluster.union(cluster_)
                merged = True

        if not merged: 
            final_clusters.append(cluster)

    #print("clustered_equivalences",final_clusters)
        
    # now enforce definition of equivalence
    visited = set()
    for cluster in final_clusters:
        paired_spans = product(cluster,cluster)
        for span,span_ in paired_spans:
            if span != span_ and ordered(span,span_) not in visited: 

                edges[ordered(span,span_)] = "equivalence" 
                visited.add(ordered(span,span_))
    
                nodes_from_span = { node for node in nodes if ordered(span,node) in edges and node != span and node != span_ }
                nodes_from_span_ = { node for node  in nodes if ordered(span_,node) in edges and node != span and node != span_ }   
    
                for node in nodes_from_span:
                    edge_label = edges[ordered(node,span)]
                    if node not in nodes_from_span_:
                        edges[ordered(node,span_)] = edge_label
                    else:
                        edge_label_ = edges[ordered(node,span_)]   
                        if edge_label == "equivalence" and edge_label_ != "equivalence":
                            edges[ordered(node,span_)] = "equivalence"

                for node_ in nodes_from_span_:
                    edge_label_ = edges[ordered(node_,span_)]
                    if node_ not in nodes_from_span:
                        edges[ordered(node_,span)] = edge_label_
                    else:
                        edge_label = edges[ordered(node,span)] 
                        if edge_label != "equivalence" and edge_label_ == "equivalence":
                            edges[ordered(node,span)] = "equivalence"

    ## finally enforce analogy
    discarded_edges = set()
    v_analogy_edges = [ (span,span_) for (span,span_) in edges if edges[(span,span_)] == "analogy" and nodes[span] == "value" ]
    for v_span,v_span_ in v_analogy_edges: 
        facts = [ node for node in nodes if edges.get(ordered(node,v_span)) == "fact" ]
        facts_ = [ node for node in nodes if edges.get(ordered(node,v_span_)) == "fact" ]
        if not any( edges.get(ordered(n,n_)) == "analogy" for (n,n_) in product(facts,facts_) ):
            discarded_edges.add((v_span,v_span_))

    edges = { edge:edges[edge] for edge in edges if edge not in discarded_edges }

    discarded_edges = set()
    f_analogy_edges = [ (span,span_) for (span,span_) in edges if edges[(span,span_)] == "analogy" and nodes[span] != "value" ]
    for f_span,f_span_ in f_analogy_edges:
        values = [ node for node in nodes if edges.get(ordered(node,f_span)) == "fact" ]
        values_ = [ node for node in nodes if edges.get(ordered(node,f_span_)) == "fact" ]
        if not any( edges.get(ordered(n,n_)) == "analogy" for (n,n_) in product(values,values_) ):
            discarded_edges.add((f_span,f_span_))

    edges = { edge:edges[edge] for edge in edges if edge not in discarded_edges }

    return nodes,edges


def test_agglomerative_clustering():

    ## test 1 check typing constraints
    nodes = { (0,1):"value" ,
              (1,2):"value" ,  
              (2,3):"value" , 
              (3,4):"value" ,
              (4,5):"value" ,
              (5,6):"value" ,
              (6,7):"value" } 

    edges = { ((0,1),(1,2)):"analogy",        # 1 - 2
              ((1,2),(2,3)):"analogy",        # 2 - 3
              ((0,1),(3,5)):"analogy",        # 1 - 4
              ((4,5),(5,6)):"analogy"         # 5 - 6  
            }


    final_clusters = []
    value_spans = [ {span} for span in nodes if nodes[span] == "value" ]  
    
    for i in range(len(value_spans)-1):
        merged = False
        cluster = value_spans[i]
        for j in range(i+1,len(value_spans)):
            cluster_ = value_spans[j]
            if any( edges.get(ordered(span,span_)) == "analogy" for (span,span_) in product(cluster,cluster_) ):
                value_spans[j] = value_spans[j].union(cluster)
                merged = True

        if merged == False:
            final_clusters.append(cluster)

    print(final_clusters)


def test_typing_constraints():

    analogy = "analogy"
    equiv = "equivalence"
    fact = "fact"

    ## test 1 check typing constraints
    nodes = { (0,1):"value" ,
              (1,2):"value" ,  
              (2,3):"value" , 
              (3,4):"theme" ,
              (4,5):"quant" }

    edges = { ((0,1),(1,2)):"analogy",        # okay
              ((1,2),(2,3)):"fact",           # not okay
              ((0,1),(2,3)):"analogy",        # okay     
              ((3,4),(4,5)):"equiv",          # not okay
              ((0,1),(4,5)):"fact",           # okay
              ((1,2),(4,5)):"fact"            # okay
            }

    desired_edges = { ((0,1),(1,2)):"analogy",
                      ((0,1),(2,3)):"analogy",
                      ((0,1),(4,5)):"fact",
                      ((1,2),(4,5)):"fact" }   

    _,e = cluster_graph(nodes,edges,{})

    assert e == desired_edges
    print("desired", desired_edges)
    print("output", e)


def test_typing_plus_invalid_values_constraints():

    analogy = "analogy"
    equiv = "equivalence"
    fact = "fact"

    ## test 1 check typing constraints
    nodes = { (0,1):"value" ,
              (1,2):"value" ,  
              (2,3):"value" , 
              (3,4):"theme" ,
              (4,5):"quant" }

    edges = { ((0,1),(1,2)):"analogy",        # okay
              ((1,2),(2,3)):"fact",           # not okay
              ((0,1),(2,3)):"analogy",        # okay     
              ((3,4),(4,5)):"equivalence",          # not okay
              ((0,1),(4,5)):"fact"            # okay
            }

    # haven't pruned these yet
    desired_edges = { ((0,1),(4,5)):"fact" }

    desired_nodes = { (0,1):"value",
                      (4,5):"quant"
                    }

    n,e = cluster_graph(nodes,edges,{})

    assert e == desired_edges
    print("desired edges", desired_edges)
    print("output edges", e)

    print("desired nodes", desired_nodes)
    print("output nodes", n)

def test_equivalence_transitivity():

    nodes = { (0,1):"theme" ,
              (1,2):"theme" ,  
              (2,3):"theme" , 
              (3,4):"theme" }

    edges = { ((0,1),(1,2)):"equivalence",        
              ((1,2),(2,3)):"equivalence",       
              ((2,3),(3,4)):"equivalence"
            }
    print( "call 1 " )

    _,e = cluster_graph(nodes,edges,{})

    nodes = { (0,1):"theme" ,
              (1,2):"theme" ,  
              (2,3):"theme" , 
              (3,4):"theme" }

    edges = { ((0,1),(1,2)):"equivalence",        
              ((1,2),(2,3)):"equivalence",       
              ((0,1),(2,3)):"analogy"
            }

    print( "call 2 " )
    _,e = cluster_graph(nodes,edges,{})



def test_analogy_quadrangle():

    nodes = { (0,1):"value" ,
              (1,2):"value" ,  
              (2,3):"theme" , 
              (3,4):"theme" }

    edges = { ((0,1),(1,2)):"analogy",              
              ((0,1),(2,3)):"fact",
              ((1,2),(3,4)):"fact"
            }

    _,e = cluster_graph(nodes,edges,{})

    print(edges)
    print(e)

    ## first commented out discarded value nodes first pass, so that it's easier to define a test
    nodes = { (0,1):"value" ,
              (1,2):"value" ,  
              (2,3):"theme" , 
              (3,4):"theme" }

    edges = { ((0,1),(2,3)):"fact",
              ((1,2),(3,4)):"fact",
              ((2,3),(3,4)):"analogy"
            }

    print(edges)
    print(e)




# test_analogy_quadrangle()



