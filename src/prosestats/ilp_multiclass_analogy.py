"""
Exposes solve_ilp, which takes a sentence graph as input and output.
"""

import pdb
from collections import Counter
from itertools import product
import gurobipy as gr
import numpy as np

from .schema import SentenceGraph, ordered

null_label = "null"

#node_labels = ["value","manner","quant","co_quant","theme","agent","cause","source","condition","theme_mod","whole"]

node_labels = SentenceGraph.NODE_LABELS
node_labels.append(null_label)
# node_labels = [label.lower() for label in node_labels ]

edge_labels = ["equivalence","analogy","fact","null"]

'''
Helpers
'''
## this shouldn't occur
def get_overlapping_node_variables(node_variables):
    '''
    Return a [ (node variable,[conflicting node variables]) ] list covering all node overlaps but not repeating
    '''
    conflict_sets = []
    nodes = list(node_variables.keys())
    for i,node in enumerate(nodes):
        conflicting_nodes = [ node_variables[nodes[j]][l] for j in range(0,len(nodes)) for l in node_labels if check_node_overlap(node,nodes[j]) and j != i ]
        if conflicting_nodes:
            for label in node_labels:
                conflict_sets.append((node_variables[node][label],conflicting_nodes))

    return conflict_sets

def stringify_node(node):
    return "(" + str(node[0]) + "," + str(node[1]) + ")"

def stringify_edge(edge):
    source,target = edge
    return "((" + str(source[0]) + "," + str(source[1]) + "),(" + str(target[0]) + "," + str(target[1]) + "))"

def check_node_overlap(node_1,node_2):

    node_1 = set(range(node_1[0],node_1[1]))
    node_2 = set(range(node_2[0],node_2[1]))

    return len(node_1.intersection(node_2)) != 0

def get_other_node_in_edge(node,edge):
    assert edge[0] == node or edge[1] == node

    if edge[0] == node:
        return edge[1]
    else:
        return edge[0]

def access_clamped_edge_dictionaries(node,node_,clamped_endpoint_dict,reverse=False):

    if (node,node_) in clamped_endpoint_dict:
        if not reverse:
            return clamped_endpoint_dict[(node,node_)][0]
        else:
            return clamped_endpoint_dict[(node,node_)][1]
    else:
        if not reverse:
            return clamped_endpoint_dict[(node_,node)][1]
        else:
            return clamped_endpoint_dict[(node_,node)][0]

def get_edge_variable(node,node_,edge_variables,edge_type):
    var = None
    if (node,node_) in edge_variables:
        var = edge_variables[(node,node_)].get(edge_type)
    elif (node_,node) in edge_variables:
        var = edge_variables[(node_,node)].get(edge_type)
    return var

'''
# Variable creation methods
'''
def instantiate_variable(model,label,var_type,var,score):
    '''
    Instantiate positive and negative variable instances
        -ilp_model := gurobi model
        -label := classification label (e.g. "value" or "fact")
        -var_type := "node" or "edge"
        -index := unique integer identifier
        -score := 0 < s < 1
    '''
    variable = None
    if var_type == "node":
        node_name = stringify_node(var)
        if score > 0:
            variable = model.addVar(obj=-np.log(score), vtype=gr.GRB.BINARY, name=label + "_" + var_type + "_" + node_name )
        else:
            variable = model.addVar(obj=10000, vtype=gr.GRB.BINARY, name=label + "_" + var_type + "_" + node_name )
    else:
        edge_name = stringify_edge(var)
        if score > 0:
            variable = model.addVar(obj=-np.log(score), vtype=gr.GRB.BINARY, name=label + "_" + var_type + "_" + edge_name )
        else:
            variable = model.addVar(obj=10000, vtype=gr.GRB.BINARY, name=label + "_" + var_type + "_" + edge_name )

    model.update()
    return variable


def instantiate_same_role_conjunction_variables(model,node_variables,edge_variables):
    '''
    # for each edge (s,s'), for each role r, instantiate a conjunction r(s) AND r(s')   
    # these conditions are required by EQUIV or ANALOGY labels
    '''
    same_role_conjunctions = {}
    for edge in edge_variables:

        node_1,node_2=edge
        edge_name = stringify_edge(edge)

        same_role_conjunctions[edge] = {}

        for role in node_labels:
            if role != null_label:
                node_1_var = node_variables[node_1][role]
                node_2_var = node_variables[node_2][role]

                # print("adding " + edge_name+"_"+role+"_conjunction" )
                v = model.addVar(vtype=gr.GRB.BINARY,name=edge_name+"_"+role+"_conjunction")
                # doesn't need a name, just an internal constraint defining v
                model.addGenConstrAnd(v, [node_1_var,node_2_var])
                model.update()

                same_role_conjunctions[edge][role] = v

    return same_role_conjunctions

def instantiate_non_value_disjunction_variables(model,node_variables):

    non_value_disjunction_variables = {}

    for node in node_variables:
        node_string = stringify_node(node)

        var = model.addVar(vtype=gr.GRB.BINARY, name=node_string+"_"+"active_non-value")
        model.addGenConstrOr(var, [ node_variables[node][label] for label in node_labels if label != null_label and label != "value" ])

        model.update()

        non_value_disjunction_variables[node] = var

    return non_value_disjunction_variables

'''
# Add constraints
'''
def require_choose_one(model,variables,var_type):
    '''
    For any variable (edge or node) require that it is assigned exactly one label (including NULL)
    '''
    for elem in variables:
        model.addConstr( gr.quicksum([ variables[elem][label] for label in variables[elem]]) == 1)

    return

def require_same_role_endpoints(model,edge_variables,edge_label,same_role_conjunction_variables):
    '''
    For each edge, if edge_label, require same_role_conjunction
    '''
    #if edge_label == "equivalence":
    #    #pdb.set_trace()
    for edge in edge_variables:
        edge_var = edge_variables[edge][edge_label]

        if edge_label != "analogy":
            # require that equiv can never occur between value arcs 
            model.addGenConstrIndicator(edge_var, True, gr.quicksum([same_role_conjunction_variables[edge][role] for role in same_role_conjunction_variables[edge] if role != "value" ]) == 1)
        else:
            model.addGenConstrIndicator(edge_var, True, gr.quicksum([same_role_conjunction_variables[edge][role] for role in same_role_conjunction_variables[edge]]) == 1)
        model.update()


# disallow_arc_between_role(model,edge_variables,"analogy","manner",same_role_conjunction_variables)
def disallow_edge_between_roles(model,edge_variables,edge_label,role,same_role_conjunction_variables):
    '''
    For each if same_role_conjunction.role true of endpoints in edge, then edge.edge_label must be false
    '''
    #if edge_label == "equivalence":
    #    #pdb.set_trace()
    for edge in edge_variables:
        conjunction = same_role_conjunction_variables[edge][role] 
        edge_var = edge_variables[edge][edge_label]
        model.addGenConstrIndicator(conjunction, True, edge_var == 0)
        model.update()

def require_fact_arc_typing_constraints(model,edge_variables,node_variables,non_value_disjunction_variables):
    '''
    for each edge s,s', if edge_label == "fact", then either s == VALUE and s' != VALUE/NULL vice versa.
    '''

    for edge in edge_variables:

        fact_edge_var = edge_variables[edge]["fact"]

        node_1,node_2 = edge
        node_string_1,node_string_2 = stringify_node(node_1),stringify_node(node_2)

        node_1_value = node_variables[node_1]["value"]
        node_2_value = node_variables[node_2]["value"]

        node_1_active_nonvalue = non_value_disjunction_variables[node_1]
        node_2_active_nonvalue = non_value_disjunction_variables[node_2]
    
        c_1 = model.addVar(vtype=gr.GRB.BINARY, name=node_string_1 + "_VALUE_" + node_string_2  + "_NON_VALUE" )
        c_2 = model.addVar(vtype=gr.GRB.BINARY, name=node_string_1 + "_NON_VALUE_" + node_string_2 + "_VALUE" )

        model.addGenConstrAnd(c_1, [node_1_value, node_2_active_nonvalue])
        model.addGenConstrAnd(c_2, [node_1_active_nonvalue, node_2_value])

        model.addGenConstrIndicator(fact_edge_var, True, gr.quicksum([c_1,c_2]) == 1)

    model.update()

def require_transitive_equiv_triangles(model,edge_variables):
    '''
    if equiv(a,b) and equiv(b,c)) then equiv(a,c)
    '''

    edge_list = list(edge_variables.keys())

    edge_dyad_names = set()

    for i in range(len(edge_list)-2):
        
        edge_i = edge_list[i]
        i_endpoints = set(edge_i)

        for j in range(i+1,len(edge_list)-1):

            edge_j = edge_list[j]
            j_endpoints = set(edge_j)

            # check if share an endpoint
            if len(j_endpoints.intersection(i_endpoints)) == 1:

                i_j_non_overlap = i_endpoints.union(j_endpoints) - i_endpoints.intersection(j_endpoints)
                for k in range(j+1,len(edge_list)):
                    
                    edge_k = edge_list[k]
                    k_endpoints = set(edge_k)

                    # if holds then triangle
                    if len(i_j_non_overlap.intersection(k_endpoints)) == 2:

                        var_i = edge_variables[edge_i]["equivalence"]
                        var_j = edge_variables[edge_j]["equivalence"]
                        var_k = edge_variables[edge_k]["equivalence"]

                        ### avoid creating duplicate variables/constraints for edge dyads
                        dyad_name_i_j = "dyad_"+var_i.VarName+"_"+var_j.VarName
                        dyad_name_j_k = "dyad_"+var_j.VarName+"_"+var_k.VarName
                        dyad_name_i_k = "dyad_"+var_i.VarName+"_"+var_k.VarName

                        dyad_i_j = None
                        dyad_j_k = None
                        dyad_i_k = None

                        if dyad_name_i_j in edge_dyad_names:
                            dyad_i_j = model.getVarByName(dyad_name_i_j)
                        else:
                            dyad_i_j = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_i_j)
                            model.addGenConstrAnd(dyad_i_j,[var_i,var_j])
                            edge_dyad_names.add(dyad_name_i_j)                    

                        if dyad_name_j_k in edge_dyad_names:
                            dyad_j_k = model.getVarByName(dyad_name_j_k)
                        else:
                            dyad_j_k = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_j_k)
                            model.addGenConstrAnd(dyad_j_k,[var_j,var_k])
                            edge_dyad_names.add(dyad_name_j_k)

                        if dyad_name_i_k in edge_dyad_names:
                            dyad_i_k = model.getVarByName(dyad_name_i_k)
                        else:
                            dyad_i_k = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_i_k)
                            model.addGenConstrAnd(dyad_i_k,[var_i,var_k])
                            edge_dyad_names.add(dyad_name_i_k)

                        ### add all three possible triangle scenarios
                        model.addGenConstrIndicator(dyad_i_j, True, var_k == 1)
                        model.addGenConstrIndicator(dyad_j_k, True, var_i == 1)
                        model.addGenConstrIndicator(dyad_i_k, True, var_j == 1)

    model.update()


def require_transitive_analogy_triangles(model,node_variables,edge_variables):
    '''
    if equiv(a,b) and equiv(b,c)) and val(a),val(b) and val(c) then equiv(a,c)
    '''

    edge_list = list(edge_variables.keys())

    edge_dyad_names = set()

    for i in range(len(edge_list)-2):
        
        edge_i = edge_list[i]
        i_endpoints = set(edge_i)

        for j in range(i+1,len(edge_list)-1):

            edge_j = edge_list[j]
            j_endpoints = set(edge_j)

            # check if share an endpoint
            if len(j_endpoints.intersection(i_endpoints)) == 1:

                i_j_non_overlap = i_endpoints.union(j_endpoints) - i_endpoints.intersection(j_endpoints)
                for k in range(j+1,len(edge_list)):
                    
                    edge_k = edge_list[k]
                    k_endpoints = set(edge_k)

                    # if holds then triangle
                    if len(i_j_non_overlap.intersection(k_endpoints)) == 2:

                        endpoint_vars = [ node_variables[endpoint]["value"] for endpoint in i_endpoints.union(k_endpoints).union(j_endpoints) ]

                        var_i = edge_variables[edge_i]["analogy"]
                        var_j = edge_variables[edge_j]["analogy"]
                        var_k = edge_variables[edge_k]["analogy"]

                        ### avoid creating duplicate variables/constraints for edge dyads
                        dyad_name_i_j = "dyad_"+var_i.VarName+"_"+var_j.VarName
                        dyad_name_j_k = "dyad_"+var_j.VarName+"_"+var_k.VarName
                        dyad_name_i_k = "dyad_"+var_i.VarName+"_"+var_k.VarName

                        dyad_i_j = None
                        dyad_j_k = None
                        dyad_i_k = None

                        if dyad_name_i_j in edge_dyad_names:
                            dyad_i_j = model.getVarByName(dyad_name_i_j)
                        else:
                            dyad_i_j = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_i_j)
                            model.addGenConstrAnd(dyad_i_j,[var_i,var_j])
                            edge_dyad_names.add(dyad_name_i_j)                    

                        if dyad_name_j_k in edge_dyad_names:
                            dyad_j_k = model.getVarByName(dyad_name_j_k)
                        else:
                            dyad_j_k = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_j_k)
                            model.addGenConstrAnd(dyad_j_k,[var_j,var_k])
                            edge_dyad_names.add(dyad_name_j_k)

                        if dyad_name_i_k in edge_dyad_names:
                            dyad_i_k = model.getVarByName(dyad_name_i_k)
                        else:
                            dyad_i_k = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_i_k)
                            model.addGenConstrAnd(dyad_i_k,[var_i,var_k])
                            edge_dyad_names.add(dyad_name_i_k)

                        ## added requirement that analogy transitivity applies with VALUE roles only
                        dyad_i_j_and_value_roles = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_i_j + "_" + "VALUE_NODES")
                        dyad_j_k_and_value_roles = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_j_k + "_" + "VALUE_NODES")
                        dyad_i_k_and_value_roles = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_i_k + "_" + "VALUE_NODES")

                        model.addGenConstrAnd(dyad_i_j_and_value_roles,[dyad_i_j] + endpoint_vars)
                        model.addGenConstrAnd(dyad_j_k_and_value_roles,[dyad_j_k] + endpoint_vars)
                        model.addGenConstrAnd(dyad_i_k_and_value_roles,[dyad_i_k] + endpoint_vars)

                        ### add all three possible triangle scenarios
                        model.addGenConstrIndicator(dyad_i_j_and_value_roles, True, var_k == 1)
                        model.addGenConstrIndicator(dyad_j_k_and_value_roles, True, var_i == 1)
                        model.addGenConstrIndicator(dyad_i_k_and_value_roles, True, var_j == 1)

    model.update()

def require_fact_equiv_triangles_1(model,edge_variables):
    '''
    if fact(s,s') and equiv(s,s'') then fact(s,s'')
    '''

    edge_list = list(edge_variables.keys())

    edge_dyad_names = set()

    for i in range(len(edge_list)-2):
        
        edge_i = edge_list[i]
        i_endpoints = set(edge_i)

        for j in range(i+1,len(edge_list)-1):

            edge_j = edge_list[j]
            j_endpoints = set(edge_j)

            # check if share an endpoint
            if len(j_endpoints.intersection(i_endpoints)) == 1:

                i_j_non_overlap = i_endpoints.union(j_endpoints) - i_endpoints.intersection(j_endpoints)
                for k in range(j+1,len(edge_list)):
                    
                    edge_k = edge_list[k]
                    k_endpoints = set(edge_k)

                    # if holds then triangle
                    if len(i_j_non_overlap.intersection(k_endpoints)) == 2:

                        fact_var_i = edge_variables[edge_i]["fact"]
                        fact_var_j = edge_variables[edge_j]["fact"]
                        fact_var_k = edge_variables[edge_k]["fact"]

                        equiv_var_i = edge_variables[edge_i]["equivalence"]
                        equiv_var_j = edge_variables[edge_j]["equivalence"]
                        equiv_var_k = edge_variables[edge_k]["equivalence"]

                        ### avoid creating duplicate variables/constraints for edge dyads
                        dyad_name_fact_i_equiv_j = "dyad_"+fact_var_i.VarName+"_"+equiv_var_j.VarName
                        dyad_name_equiv_i_fact_j = "dyad_"+equiv_var_i.VarName+"_"+fact_var_j.VarName

                        dyad_name_fact_j_equiv_k = "dyad_"+fact_var_j.VarName+"_"+equiv_var_k.VarName
                        dyad_name_equiv_j_fact_k = "dyad_"+equiv_var_j.VarName+"_"+fact_var_k.VarName

                        dyad_name_fact_i_equiv_k = "dyad_"+fact_var_i.VarName+"_"+equiv_var_k.VarName
                        dyad_name_equiv_i_fact_k = "dyad_"+equiv_var_i.VarName+"_"+fact_var_k.VarName


                        if dyad_name_fact_i_equiv_j in edge_dyad_names:
                            dyad_fact_i_equiv_j = model.getVarByName(dyad_name_fact_i_equiv_j)
                            dyad_equiv_i_fact_j = model.getVarByName(dyad_name_equiv_i_fact_j)
                        else:
                            dyad_fact_i_equiv_j = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_fact_i_equiv_j)
                            dyad_equiv_i_fact_j = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_equiv_i_fact_j)
                            
                            model.addGenConstrAnd(dyad_fact_i_equiv_j,[fact_var_i,equiv_var_j])
                            model.addGenConstrAnd(dyad_equiv_i_fact_j,[equiv_var_i,fact_var_j])
                            
                            edge_dyad_names.add(dyad_name_fact_i_equiv_j)                    
                            edge_dyad_names.add(dyad_name_equiv_i_fact_j)

                        if dyad_name_fact_j_equiv_k in edge_dyad_names:
                            dyad_fact_j_equiv_k = model.getVarByName(dyad_name_fact_j_equiv_k)
                            dyad_equiv_j_fact_k = model.getVarByName(dyad_name_equiv_j_fact_k)
                        else:
                            dyad_fact_j_equiv_k = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_fact_j_equiv_k)
                            dyad_equiv_j_fact_k = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_equiv_j_fact_k)
                            
                            model.addGenConstrAnd(dyad_fact_j_equiv_k,[fact_var_j,equiv_var_k])
                            model.addGenConstrAnd(dyad_equiv_j_fact_k,[equiv_var_j,fact_var_k])
                            
                            edge_dyad_names.add(dyad_name_fact_j_equiv_k)                    
                            edge_dyad_names.add(dyad_name_equiv_j_fact_k)

                        if dyad_name_fact_i_equiv_k in edge_dyad_names:
                            dyad_fact_i_equiv_k = model.getVarByName(dyad_name_fact_i_equiv_k)
                            dyad_equiv_i_fact_k = model.getVarByName(dyad_name_equiv_i_fact_k)
                        else:
                            dyad_fact_i_equiv_k = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_fact_i_equiv_k)
                            dyad_equiv_i_fact_k = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_equiv_i_fact_k)
                            
                            model.addGenConstrAnd(dyad_fact_i_equiv_k,[fact_var_i,equiv_var_k])
                            model.addGenConstrAnd(dyad_equiv_i_fact_k,[equiv_var_i,fact_var_k])
                            
                            edge_dyad_names.add(dyad_name_fact_i_equiv_k)                    
                            edge_dyad_names.add(dyad_name_equiv_i_fact_k)

                        model.addGenConstrIndicator(dyad_fact_i_equiv_j, True, fact_var_k == 1) 
                        model.addGenConstrIndicator(dyad_equiv_i_fact_j, True, fact_var_k == 1) 

                        model.addGenConstrIndicator(dyad_fact_j_equiv_k, True, fact_var_i == 1) 
                        model.addGenConstrIndicator(dyad_equiv_j_fact_k, True, fact_var_i == 1) 

                        model.addGenConstrIndicator(dyad_fact_i_equiv_k, True, fact_var_j == 1) 
                        model.addGenConstrIndicator(dyad_equiv_i_fact_k, True, fact_var_j == 1) 

    model.update()

def require_fact_equiv_triangles_2(model,edge_variables,same_role_conjunction_variables):
    '''
    if fact(s,s') and fact(s,s'') and role(s') == role(s''), equiv(s',s'')
    '''
    edge_list = list(edge_variables.keys())

    edge_dyad_names = set()

    for i in range(len(edge_list)-2):
        
        edge_i = edge_list[i]
        i_endpoints = set(edge_i)

        for j in range(i+1,len(edge_list)-1):

            edge_j = edge_list[j]
            j_endpoints = set(edge_j)

            # check if share an endpoint
            if len(j_endpoints.intersection(i_endpoints)) == 1:

                i_j_non_overlap = i_endpoints.union(j_endpoints) - i_endpoints.intersection(j_endpoints)
                for k in range(j+1,len(edge_list)):
                    
                    edge_k = edge_list[k]
                    k_endpoints = set(edge_k)

                    # if holds then triangle
                    if len(i_j_non_overlap.intersection(k_endpoints)) == 2:

                        fact_var_i = edge_variables[edge_i]["fact"]
                        fact_var_j = edge_variables[edge_j]["fact"]
                        fact_var_k = edge_variables[edge_k]["fact"]

                        equiv_var_i = edge_variables[edge_i]["equivalence"]
                        equiv_var_j = edge_variables[edge_j]["equivalence"]
                        equiv_var_k = edge_variables[edge_k]["equivalence"]

                        ### avoid creating duplicate variables/constraints for edge dyads
                        dyad_name_fact_i_fact_j = "dyad_"+fact_var_i.VarName+"_"+fact_var_j.VarName
                        dyad_name_fact_j_fact_k = "dyad_"+fact_var_j.VarName+"_"+fact_var_k.VarName
                        dyad_name_fact_i_fact_k = "dyad_"+fact_var_i.VarName+"_"+fact_var_k.VarName

                        if dyad_name_fact_i_fact_j in edge_dyad_names:
                            dyad_fact_i_fact_j = model.getVarByName(dyad_name_fact_i_fact_j)
                        else:
                            dyad_fact_i_fact_j = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_fact_i_fact_j)
                            
                            model.addGenConstrAnd(dyad_fact_i_fact_j,[fact_var_i,fact_var_j])
                            
                            edge_dyad_names.add(dyad_name_fact_i_fact_j)

                        if dyad_name_fact_j_fact_k in edge_dyad_names:
                            dyad_fact_j_fact_k = model.getVarByName(dyad_name_fact_j_fact_k)
                        else:
                            dyad_fact_j_fact_k = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_fact_j_fact_k)
                            
                            model.addGenConstrAnd(dyad_fact_j_fact_k,[fact_var_j,fact_var_k])
                            
                            edge_dyad_names.add(dyad_name_fact_j_fact_k)   

                        if dyad_name_fact_i_fact_k in edge_dyad_names:
                            dyad_fact_i_fact_k = model.getVarByName(dyad_name_fact_i_fact_k)
                        else:
                            dyad_fact_i_fact_k = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_fact_i_fact_k)
                            
                            model.addGenConstrAnd(dyad_fact_i_fact_k,[fact_var_i,fact_var_k])
                            
                            edge_dyad_names.add(dyad_name_fact_i_fact_k)                         

                        for role in node_labels:
                            if role not in ["value", null_label]:

                                same_role_var_i = same_role_conjunction_variables[edge_i][role]
                                same_role_var_j = same_role_conjunction_variables[edge_j][role]
                                same_role_var_k = same_role_conjunction_variables[edge_k][role]

                                dyad_i_j_and_same_role_var = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_fact_i_fact_j+"_AND_"+same_role_var_k.VarName)
                                dyad_j_k_and_same_role_var = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_fact_j_fact_k+"_AND_"+same_role_var_i.VarName)
                                dyad_i_k_and_same_role_var = model.addVar(vtype=gr.GRB.BINARY, name=dyad_name_fact_i_fact_k+"_AND_"+same_role_var_j.VarName)

                                model.addGenConstrAnd(dyad_i_j_and_same_role_var,[dyad_fact_i_fact_j,same_role_var_k])
                                model.addGenConstrAnd(dyad_j_k_and_same_role_var,[dyad_fact_j_fact_k,same_role_var_i])
                                model.addGenConstrAnd(dyad_i_k_and_same_role_var,[dyad_fact_i_fact_k,same_role_var_j])

                                model.addGenConstrIndicator(dyad_i_j_and_same_role_var, True, equiv_var_k == 1)  
                                model.addGenConstrIndicator(dyad_j_k_and_same_role_var, True, equiv_var_i == 1)  
                                model.addGenConstrIndicator(dyad_i_k_and_same_role_var, True, equiv_var_j == 1)  
                                

    model.update()

def require_value_quadrangles(model,edge_variables):
   
    quad_counter = 0
    value_analogy_counter = 0
    for span,span_ in edge_variables:
        
        analogy_edge_var = edge_variables[(span,span_)]["analogy"]  

        value_analogy_counter += 1

        possible_quadrangles = []

        edges = [ e for e in edge_variables if span in e and span_ not in e ]
        edges_ = [ e for e in edge_variables if span_ in e and span not in e ]

        for e,e_ in product(edges,edges_):

                o_span = [ s for s in e if s != span ][0]
                o_span_ = [ s for s in e_ if s != span_ ][0]

                if o_span != o_span_:

                    fact_edge = edge_variables[e]["fact"]
                    fact_edge_ = edge_variables[e_]["fact"]   

                    quad_edge = edge_variables.get(ordered(o_span,o_span_))
                    if quad_edge == None:
                        break
                    else:
                        quad_edge = quad_edge.get("analogy")

                    quad_counter += 1

                    v = model.addVar(vtype=gr.GRB.BINARY, name="some_quadrangle_" + str(quad_counter))   
                    model.addGenConstrAnd(v, [ fact_edge, fact_edge_, quad_edge ])
                    possible_quadrangles.append(v)

        # val_analogy_conj = model.addVar(vtype=gr.GRB.BINARY, name="some_analogy_quadrangle_" + str(value_analogy_counter) )   
        model.addGenConstrIndicator(analogy_edge_var, True, gr.quicksum( possible_quadrangles ) >= 1 )

    model.update()


def require_alo_value_analogy_arc(model,node_variables,edge_variables):

    for span in node_variables:
        value_var = node_variables[span]["value"]
        model.addGenConstrIndicator(value_var, True, gr.quicksum( edge_variables[edge]["analogy"] for edge in edge_variables if span in edge ) >= 1 )

    model.update()

def require_alo_fact_arc_per_role(model,node_variables,edge_variables):

    for span in node_variables:
        for role in node_variables[span]:
            if role != "null":
                var = node_variables[span][role]
                model.addGenConstrIndicator(var, True, gr.quicksum( edge_variables[edge]["fact"] for edge in edge_variables if span in edge ) >= 1 )

    model.update()


def require_alo_value_node(model,node_variables):

    model.addConstr( gr.quicksum( node_variables[span]["value"] for span in node_variables ) >= 1 )

    model.update() 


'''
# ILP for numerical analogy parsing
'''
def construct_and_solve_sentence_ilp(sentence_datum):
    ### ILPs for decoding into analogy graphs
    ### currently returns converted model using old convert_solution method --- not sure if this needs redoing

    '''
    # Initialize gurobi model
    '''
    model = gr.Model()
    model.setParam(gr.GRB.Param.TimeLimit, 300) # (5 minutes == 5 * 60 seconds)

    '''
    # Unpack candidates
    '''
    nodes,node_scores = zip(*sentence_datum["nodes"])
    edges,edge_scores = zip(*sentence_datum["edges"])

    node_to_index = { node:i for i,node in enumerate(nodes) }
    edge_to_index = { edge:i for i,edge in enumerate(edges) }

    '''
    # Instantiate variables
    '''
    node_variables = {}
    for i,node in enumerate(nodes):
        node_variables[node] = { label:instantiate_variable(model,label,"node",node,node_scores[i].get(label)) for label in node_scores[i] }
    
    edge_variables = {}
    for i,edge in enumerate(edges):
        edge_variables[edge] = { label:instantiate_variable(model,label,"edge",edge,edge_scores[i].get(label)) for label in edge_scores[i] }
    

    ### ~~~ CHOOSE ONE CONSTRAINTS ~~~ ### 
    # unique roles
    require_choose_one(model,node_variables,"node")
    # unique edges (think this suffices for constraint (3) in writeup)
    require_choose_one(model,edge_variables,"edge")

    
    same_role_conjunction_variables = instantiate_same_role_conjunction_variables(model,node_variables,edge_variables)
    non_value_disjunction_variables = instantiate_non_value_disjunction_variables(model,node_variables)
    ### ~~~ TYPING CONSTRAINTS ~~~ ### 
    # if analogy arc then endpoint edges must have the same role
    ### apply ilp 2
    require_same_role_endpoints(model,edge_variables,"analogy",same_role_conjunction_variables)
    # if equiv arc then endpoint edges must have the same role
    ### apply ilp 3
    require_same_role_endpoints(model,edge_variables,"equivalence",same_role_conjunction_variables)
    # if frame arc then one endpoint must be a value and one must be a non-value
    ### apply ilp 4
    require_fact_arc_typing_constraints(model,edge_variables,node_variables,non_value_disjunction_variables)
    # if manner arcs then link must be equivalence 
    disallow_edge_between_roles(model,edge_variables,"analogy","manner",same_role_conjunction_variables)
    
    ### ~~~ TRIANGLE CONSTRAINTS ~~~ ###
    require_transitive_equiv_triangles(model,edge_variables)
    ### apply ilp 5
    require_transitive_analogy_triangles(model,node_variables,edge_variables)
    require_fact_equiv_triangles_1(model,edge_variables)
    require_fact_equiv_triangles_2(model,edge_variables,same_role_conjunction_variables)
    
    ### apply ilp 
    require_alo_fact_arc_per_role(model,node_variables,edge_variables)

    ### ~~~ QUADRANGLE CONSTRAINTS ~~~ ###
    require_value_quadrangles(model,edge_variables)

    ### ~~~ MINIMAL REQUIREMENT CONSTRAINTS ~~~ ###
    require_alo_value_analogy_arc(model,node_variables,edge_variables)
    require_alo_value_node(model,node_variables)

    
    model.optimize()

    '''
    for node in node_variables:
        for label in node_variables[node]:
            if node_variables[node][label].X == 1:
                print("**********")
                print(node_variables[node][label].VarName)
                print(node_variables[node][label].X)

    for edge in same_role_conjunction_variables:
        for role in same_role_conjunction_variables[edge]:
            if same_role_conjunction_variables[edge][role].X == 1:
                print("**********")
                print(same_role_conjunction_variables[edge][role].VarName)
                print(same_role_conjunction_variables[edge][role].X)

    for edge in edge_variables:
        for label in edge_variables[edge]:
            if edge_variables[edge][label].X == 1:
                print("**********")
                print(edge_variables[edge][label].VarName)
                print(edge_variables[edge][label].X)
    '''


    if model.Status == gr.GRB.INFEASIBLE or model.Status == gr.GRB.TIME_LIMIT:
        return {span: Counter({"null":1.}) for span in node_variables}, {(span, span_): Counter({"null":1.}) for span, span_ in edge_variables}
    else:
        return convert_solution(node_variables, edge_variables)


def convert_solution(node_variables, edge_variables):
    """
    Converts a solution with gurobi variables into a more familiar graph.
    """
    ret_nodes, ret_edges = {}, {}
    for span, attrs in node_variables.items():
        # Convert attrs into a Counter.
        ret_nodes[span] = Counter({attr: var.x for attr, var in attrs.items()})
        assert sum(ret_nodes[span].values()) > 0.
    for (span, span_), attrs in edge_variables.items():
        # Convert attrs into a Counter.
        ret_edges[(span, span_)] = Counter({attr: var.x for attr, var in attrs.items()})
        assert sum(ret_edges[(span, span_)].values()) > 0.
    return ret_nodes, ret_edges

def to_datum(graph):
    ret = {
        "nodes": [(span, {(key) or "null": value for key, value in attr.items()}) for span, attr, _, _ in graph.nodes],
        "edges": [((span, span_), {(key) or "null": value for key, value in attr.items()}) for span, span_, attr in graph.edges],
        }
    return ret

def solve_ilp(graph):
    """
    Wrapper around construct_and_solve_sentence_ilp that converts into/out of sentence graph.
    """
    # Ignore empty graphs.
    if not graph.nodes:
        return graph

    ret_nodes, ret_edges = construct_and_solve_sentence_ilp(to_datum(graph))

    assert len(ret_nodes) == len(graph.nodes)
    assert len(ret_edges) == len(graph.edges)

    # Update elements
    graph_ = SentenceGraph(graph.tokens)
    # TODO: properly support sign and manner attributes.
    for span, _, sign, manner in graph.nodes:
        attr = Counter({key.lower(): value for key, value in ret_nodes[span].items()})
        attr[None] = 1.0 - sum(attr.values())
        graph_.nodes.append((span, attr, sign, manner))
    for span, span_, _ in graph.edges:
        attr = Counter({key.lower(): value for key, value in ret_edges[span, span_].items()})
        attr[None] = 1.0 - sum(attr.values())
        graph_.edges.append((span, span_, attr))

    return graph_

'''
Tests
'''
def init_zero_datum():

    return { "nodes":[ ((4,5),{label:0 for label in node_labels}),
                       ((5,6),{label:0 for label in node_labels}),
                       ((6,7),{label:0 for label in node_labels})
                     ],
             
             "edges":[ (((4,5),(5,6)),{label:0 for label in edge_labels}),
                       (((4,5),(6,7)),{label:0 for label in edge_labels}),
                       (((5,6),(6,7)),{label:0 for label in edge_labels})
             ]
            }



def test_2():
    '''
    test equiv transitivity
    '''

    sentence_datum = init_zero_datum()

    print(" ________________ permutation_1")

    sentence_datum["nodes"][0][1]["theme"] = .9
    sentence_datum["nodes"][0][1][null_label] = .1

    sentence_datum["nodes"][1][1]["theme"] = .9
    sentence_datum["nodes"][1][1][null_label] = .1

    sentence_datum["nodes"][2][1]["theme"] = .9
    sentence_datum["nodes"][2][1][null_label] = .1


    sentence_datum["edges"][0][1]["equivalence"] = .9
    sentence_datum["edges"][0][1][null_label] = .1

    sentence_datum["edges"][1][1]["equivalence"] = .9
    sentence_datum["edges"][1][1][null_label] = .1

    sentence_datum["edges"][2][1]["equivalence"] = .45
    sentence_datum["edges"][2][1][null_label] = .55

    construct_and_solve_sentence_ilp(sentence_datum)

    print( " ________________ permutation_2")
    sentence_datum["nodes"][0][1]["theme"] = .9
    sentence_datum["nodes"][0][1][null_label] = .1

    sentence_datum["nodes"][1][1]["theme"] = .9
    sentence_datum["nodes"][1][1][null_label] = .1

    sentence_datum["nodes"][2][1]["theme"] = .9
    sentence_datum["nodes"][2][1][null_label] = .1

    sentence_datum["edges"][0][1]["equivalence"] = .9
    sentence_datum["edges"][0][1][null_label] = .1

    sentence_datum["edges"][1][1]["equivalence"] = .45
    sentence_datum["edges"][1][1][null_label] = .55

    sentence_datum["edges"][2][1]["equivalence"] = .9
    sentence_datum["edges"][2][1][null_label] = .1


    construct_and_solve_sentence_ilp(sentence_datum)
    
    print( " ________________  permutation_3")
    sentence_datum["nodes"][0][1]["theme"] = .9
    sentence_datum["nodes"][0][1][null_label] = .1

    sentence_datum["nodes"][1][1]["theme"] = .9
    sentence_datum["nodes"][1][1][null_label] = .1

    sentence_datum["nodes"][2][1]["theme"] = .9
    sentence_datum["nodes"][2][1][null_label] = .1

    sentence_datum["edges"][0][1]["equivalence"] = .45
    sentence_datum["edges"][0][1][null_label] = .55

    sentence_datum["edges"][1][1]["equivalence"] = .9
    sentence_datum["edges"][1][1][null_label] = .1

    sentence_datum["edges"][2][1]["equivalence"] = .9
    sentence_datum["edges"][2][1][null_label] = .1

    construct_and_solve_sentence_ilp(sentence_datum)


def test_3():
    '''
    test analogy value transitivity
    '''

    sentence_datum = init_zero_datum()

    print(" ________________ permutation_1")

    sentence_datum["nodes"][0][1]["value"] = .9
    sentence_datum["nodes"][0][1][null_label] = .1

    sentence_datum["nodes"][1][1]["value"] = .9
    sentence_datum["nodes"][1][1][null_label] = .1

    sentence_datum["nodes"][2][1]["value"] = .9
    sentence_datum["nodes"][2][1][null_label] = .1


    sentence_datum["edges"][0][1]["analogy"] = .9
    sentence_datum["edges"][0][1][null_label] = .1

    sentence_datum["edges"][1][1]["analogy"] = .9
    sentence_datum["edges"][1][1][null_label] = .1

    sentence_datum["edges"][2][1]["analogy"] = .45
    sentence_datum["edges"][2][1][null_label] = .55

    construct_and_solve_sentence_ilp(sentence_datum)

    print(" ________________ permutation_2")

    sentence_datum["nodes"][0][1]["value"] = .9
    sentence_datum["nodes"][0][1][null_label] = .1

    sentence_datum["nodes"][1][1]["value"] = .9
    sentence_datum["nodes"][1][1][null_label] = .1

    sentence_datum["nodes"][2][1]["value"] = .9
    sentence_datum["nodes"][2][1][null_label] = .1


    sentence_datum["edges"][0][1]["analogy"] = .9
    sentence_datum["edges"][0][1][null_label] = .1

    sentence_datum["edges"][1][1]["analogy"] = .45
    sentence_datum["edges"][1][1][null_label] = .55

    sentence_datum["edges"][2][1]["analogy"] = .9
    sentence_datum["edges"][2][1][null_label] = .1

    construct_and_solve_sentence_ilp(sentence_datum)

    print(" ________________ permutation_3")

    sentence_datum["nodes"][0][1]["value"] = .9
    sentence_datum["nodes"][0][1][null_label] = .1

    sentence_datum["nodes"][1][1]["value"] = .9
    sentence_datum["nodes"][1][1][null_label] = .1

    sentence_datum["nodes"][2][1]["value"] = .9
    sentence_datum["nodes"][2][1][null_label] = .1


    sentence_datum["edges"][0][1]["analogy"] = .45
    sentence_datum["edges"][0][1][null_label] = .55

    sentence_datum["edges"][1][1]["analogy"] = .9
    sentence_datum["edges"][1][1][null_label] = .1

    sentence_datum["edges"][2][1]["analogy"] = .9
    sentence_datum["edges"][2][1][null_label] = .1

    construct_and_solve_sentence_ilp(sentence_datum)

def test_4():
    '''
    test fact_equiv_triangle_1 --> fact(s,s') and equiv(s,s'') then fact(s,s'')
    '''
    sentence_datum = init_zero_datum()

    print(" ________________ permutation_1")

    sentence_datum["nodes"][0][1]["theme"] = .9
    sentence_datum["nodes"][0][1][null_label] = .1

    sentence_datum["nodes"][1][1]["value"] = .9
    sentence_datum["nodes"][1][1][null_label] = .1

    sentence_datum["nodes"][2][1]["theme"] = .9
    sentence_datum["nodes"][2][1][null_label] = .1

    sentence_datum["edges"][0][1]["fact"] = .9
    sentence_datum["edges"][0][1][null_label] = .1

    sentence_datum["edges"][1][1]["equivalence"] = .9
    sentence_datum["edges"][1][1][null_label] = .1

    sentence_datum["edges"][2][1]["fact"] = .45
    sentence_datum["edges"][2][1][null_label] = .55

    construct_and_solve_sentence_ilp(sentence_datum)

    print(" ________________ permutation_2")

    sentence_datum = init_zero_datum()

    sentence_datum["nodes"][0][1]["theme"] = .9
    sentence_datum["nodes"][0][1][null_label] = .1

    sentence_datum["nodes"][1][1]["theme"] = .9
    sentence_datum["nodes"][1][1][null_label] = .1

    sentence_datum["nodes"][2][1]["value"] = .9
    sentence_datum["nodes"][2][1][null_label] = .1

    #((4,5),(5,6))                         
    sentence_datum["edges"][0][1]["equivalence"] = .9
    sentence_datum["edges"][0][1][null_label] = .1

    # ((4,5),(6,7))
    sentence_datum["edges"][1][1]["fact"] = .4
    sentence_datum["edges"][1][1][null_label] = .6

    # ((5,6),(6,7))
    sentence_datum["edges"][2][1]["fact"] = .9
    sentence_datum["edges"][2][1][null_label] = .1

    construct_and_solve_sentence_ilp(sentence_datum)

    print(" ________________ permutation_3")

    sentence_datum = init_zero_datum()

    sentence_datum["nodes"][0][1]["value"] = .9
    sentence_datum["nodes"][0][1][null_label] = .1

    sentence_datum["nodes"][1][1]["theme"] = .9
    sentence_datum["nodes"][1][1][null_label] = .1

    sentence_datum["nodes"][2][1]["theme"] = .9
    sentence_datum["nodes"][2][1][null_label] = .1

    #((4,5),(5,6))                         
    sentence_datum["edges"][0][1]["fact"] = .4
    sentence_datum["edges"][0][1][null_label] = .6

    # ((4,5),(6,7))
    sentence_datum["edges"][1][1]["fact"] = .9
    sentence_datum["edges"][1][1][null_label] = .1

    # ((5,6),(6,7))
    sentence_datum["edges"][2][1]["equivalence"] = .9
    sentence_datum["edges"][2][1][null_label] = .1

    construct_and_solve_sentence_ilp(sentence_datum)

def test_5():
    '''
    test fact_equiv_triangle_2 --> if fact(s,s') and fact(s,s'') and role(s') == role(s''), equiv(s',s'')
    '''
    sentence_datum = init_zero_datum()

    print(" ________________ permutation_1")

    # (4,5)
    sentence_datum["nodes"][0][1]["theme"] = .9
    sentence_datum["nodes"][0][1][null_label] = .1

    # (5,6)
    sentence_datum["nodes"][1][1]["value"] = .9
    sentence_datum["nodes"][1][1][null_label] = .1

    # (6,7)
    sentence_datum["nodes"][2][1]["theme"] = .9
    sentence_datum["nodes"][2][1][null_label] = .1

    #((4,5),(5,6)) 
    sentence_datum["edges"][0][1]["fact"] = .9
    sentence_datum["edges"][0][1][null_label] = .1

    # ((4,5),(6,7))
    sentence_datum["edges"][1][1]["equivalence"] = .45
    sentence_datum["edges"][1][1][null_label] = .55

    # ((5,6),(6,7))
    sentence_datum["edges"][2][1]["fact"] = .9
    sentence_datum["edges"][2][1][null_label] = .1

    construct_and_solve_sentence_ilp(sentence_datum)


    print(" ________________ permutation_2")

    # (4,5)
    sentence_datum["nodes"][0][1]["value"] = .9
    sentence_datum["nodes"][0][1][null_label] = .1

    # (5,6)
    sentence_datum["nodes"][1][1]["theme"] = .9
    sentence_datum["nodes"][1][1][null_label] = .1

    # (6,7)
    sentence_datum["nodes"][2][1]["theme"] = .9
    sentence_datum["nodes"][2][1][null_label] = .1

    #((4,5),(5,6)) 
    sentence_datum["edges"][0][1]["fact"] = .9
    sentence_datum["edges"][0][1][null_label] = .1

    # ((4,5),(6,7))
    sentence_datum["edges"][1][1]["fact"] = .9
    sentence_datum["edges"][1][1][null_label] = .1

    # ((5,6),(6,7))
    sentence_datum["edges"][2][1]["equivalence"] = .55
    sentence_datum["edges"][2][1][null_label] = .45

    construct_and_solve_sentence_ilp(sentence_datum)


    print(" ________________ permutation_3")

    sentence_datum = init_zero_datum()

    sentence_datum["nodes"][0][1]["theme"] = .9
    sentence_datum["nodes"][0][1][null_label] = .1

    sentence_datum["nodes"][1][1]["theme"] = .9
    sentence_datum["nodes"][1][1][null_label] = .1

    sentence_datum["nodes"][2][1]["value"] = .9
    sentence_datum["nodes"][2][1][null_label] = .1

    #((4,5),(5,6))                         
    sentence_datum["edges"][0][1]["equivalence"] = .55
    sentence_datum["edges"][0][1][null_label] = .45

    # ((4,5),(6,7))
    sentence_datum["edges"][1][1]["fact"] = .9
    sentence_datum["edges"][1][1][null_label] = .1

    # ((5,6),(6,7))
    sentence_datum["edges"][2][1]["fact"] = .9
    sentence_datum["edges"][2][1][null_label] = .1

    construct_and_solve_sentence_ilp(sentence_datum)



