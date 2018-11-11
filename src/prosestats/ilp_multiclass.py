"""
Exposes solve_ilp, which takes a sentence graph as input and output.
"""

import pdb
from collections import Counter

import gurobipy as gr
import numpy as np

from .schema import SentenceGraph

node_labels=["VALUE", "VALUEM", "LABEL", "LABEL_MOD", "THEME", "THEME_MOD", "POSSESSOR", "UNIT", "UNIT_RESTRICTION", "TYPE"]
instance_labels=["VALUE","VALUEM","LABEL","LABEL_MOD"]
edge_labels=["FRAME", "INSTANCE"]
core_frame_labels=["THEME","POSSESSOR","TYPE","UNIT"]
noncore_frame_labels=["THEME_MOD","UNIT_RESTRICTION"]
core_instance_labels=["VALUE","VALUEM","LABEL"]

mod_label_pairs=[("THEME","THEME_MOD"),("UNIT","UNIT_RESTRICTION"),("LABEL","LABEL_MOD"),("VALUE","VALUEM")]
null_label = "NULL"
all_node_labels = node_labels + [null_label]
all_edge_labels = edge_labels + [null_label]


endpoint_clamp_tuples = [   ("LABEL","LABEL"),
                            ("VALUE","VALUE"),
                            ("VALUEM","VALUEM"),
                            ("CORE_FRAME","CORE_FRAME"),
                            ("CORE_FRAME","CORE_INSTANCE") ]

clamped_edge_tuples = [ ("LABEL","FRAME"),
                        ("VALUE","INSTANCE"),
                        ("LABEL","INSTANCE"),
                        ("VALUE","FRAME"),
                        ("THEME","FRAME"),
                        ("UNIT","FRAME") ]

'''
Helpers
'''
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
def add_disjunction_over_variable_set(model,elem,variable_dictionary,label_set):
    v = model.addVar(vtype=gr.GRB.BINARY)
    model.addConstr( v == gr.quicksum([ variable_dictionary[elem][label] for label in label_set ]) )
    model.update()
    return v

def add_conjunction_between_two_variables(model,var_1,var_2):
    v = model.addVar(vtype=gr.GRB.BINARY)
    model.addGenConstrAnd(v, [var_1,var_2])
    model.update()
    return v

def instantiate_variable(model,label,var_type,var,score):
    '''
    Instantiate positive and negative variable instances
        -ilp_model := gurobi model
        -label := classification label (e.g. "VALUE" or "FRAME")
        -var_type := "node" or "edge"
        -index := unique integer identifier
        -score := 0 < s < 1
    '''
    positive_variable = None
    if var_type == "node":
        node_name = stringify_node(var)
        if score > 0:
            positive_variable = model.addVar(obj=-np.log(score), vtype=gr.GRB.BINARY, name=label + "_" + var_type + "_positive_" + node_name )
        else:
            positive_variable = model.addVar(obj=10000, vtype=gr.GRB.BINARY, name=label + "_" + var_type + "_positive_" + node_name )
    else:
        edge_name = stringify_edge(var)
        if score > 0:
            positive_variable = model.addVar(obj=-np.log(score), vtype=gr.GRB.BINARY, name=label + "_" + var_type + "_positive_" + edge_name )
        else:
            positive_variable = model.addVar(obj=10000, vtype=gr.GRB.BINARY, name=label + "_" + var_type + "_positive_" + edge_name )

    return positive_variable

def instantiate_clamped_endpoint_edge_constraints(model,attr,attr_,edge_variables,node_variables,core_frame_attr_indicators,core_instance_attr_indicators):

    assert (attr in node_labels or attr in ["CORE_INSTANCE","CORE_FRAME"]) and ( attr_ in node_labels or  attr_ in ["CORE_INSTANCE","CORE_FRAME"] )

    clamped_endpoint_dict = {}

    '''
    print( "in clamped endpoint edge constraints ")
    print( "attr:", attr)
    print( "attr_:", attr_)
    '''

    for edge in edge_variables:
        source,target = edge

        source_attr = None
        source_attr_ = None
        target_attr = None
        target_attr_ = None

        if attr == "CORE_INSTANCE":
            source_attr = core_instance_attr_indicators[source]
            target_attr = core_instance_attr_indicators[target]
        elif attr == "CORE_FRAME":
            source_attr = core_frame_attr_indicators[source]
            target_attr = core_frame_attr_indicators[target]
        else:
            source_attr = node_variables[source][attr]
            target_attr = node_variables[target][attr]

        if attr_ == "CORE_INSTANCE":
            source_attr_ = core_instance_attr_indicators[source]
            target_attr_ = core_instance_attr_indicators[target]
        elif attr_ == "CORE_FRAME":
            source_attr_ = core_frame_attr_indicators[source]
            target_attr_ = core_frame_attr_indicators[target]
        else:
            source_attr_ = node_variables[source][attr_]
            target_attr_ = node_variables[target][attr_]

        conj = add_conjunction_between_two_variables(model,source_attr,target_attr_)
        conj.VarName = stringify_edge(edge) + "_" + attr + "_" + attr_

        '''
        print( "edge", edge )
        print( "source" , source )
        print( "attr", attr, source_attr )
        print( "attr_" , attr_, source_attr_ )
        print( "target" , target )
        print( "attr", attr, target_attr )
        print( "attr_", attr_, target_attr_ )

        print("\n")
        '''

        if attr == attr_:
            clamped_endpoint_dict[edge] = (conj,conj)
        else:
            conj_ = add_conjunction_between_two_variables(model,source_attr_,target_attr)
            conj_.VarName = stringify_edge(edge) + "_" + attr_ + "_" + attr

            clamped_endpoint_dict[edge] = (conj,conj_)

        #print("\n\n")


    return (attr+"_"+attr_),clamped_endpoint_dict

def instantiate_partial_clamped_endpoint_edge_constraints(model,attr,edge_type,edge_variables,node_variables,core_frame_attr_indicators,core_instance_attr_indicators):

    assert (attr in node_labels or attr in ["CORE_INSTANCE","CORE_FRAME"])

    clamped_endpoint_dict = {}
    for node in node_variables:

        clamped_endpoint_dict[node] = {}
        ## get edges from node
        edges_from_node = [ edge for edge in edge_variables if node in edge ]

        for edge in edges_from_node:

            edge_var = edge_variables[edge][edge_type]
            node_ = get_other_node_in_edge(node,edge)

            if attr == "CORE_INSTANCE":
                target_var = core_instance_attr_indicators[node_]
            elif attr == "CORE_FRAME":
                target_var = core_frame_attr_indicators[node_]
            else:
                target_var = node_variables[node_][attr]

            clamped_endpoint_dict[node][edge] = add_conjunction_between_two_variables(model,edge_var,target_var)

    return (attr+"_"+edge_type,clamped_endpoint_dict)

'''
# Add constraints
'''
def require_choose_one(model,variables,var_type):
    for elem in variables:
        model.addConstr( gr.quicksum([ variables[elem][label] for label in variables[elem]]) == 1)


    return

def require_node_span_non_overlap(model,node_variables):
    overlaps = get_overlapping_node_variables(node_variables)
    non_overlap_constraints = [ model.addGenConstrIndicator(var, True, gr.quicksum(conflicting_vars) == 0, name=var.getAttr("VarName") + "_nonoverlap") for var,conflicting_vars in overlaps ]


    return non_overlap_constraints

def require_distinct_instance_arc_endpoints(model,node_variables,edge_variables):

    for edge in edge_variables:
        edge_var = edge_variables[edge]["INSTANCE"]
        source,target = edge
        for label in instance_labels:
            model.addConstr( edge_var + node_variables[source][label] + node_variables[target][label] <= 2 )


def require_instance_labels_for_instance_arcs(model,node_variables,edge_variables,core_instance_attr_indicators):

    for edge in edge_variables:
        instance_edge_var = edge_variables[edge]["INSTANCE"]
        source,target = edge

        core_instance_endpoints = model.addVar(vtype=gr.GRB.BINARY)
        ## constraints involving "core_X_attribute_indicators" (which are disjunctions), are seriously inefficient!
        model.addGenConstrAnd(core_instance_endpoints,[core_instance_attr_indicators[source],core_instance_attr_indicators[target]])

        source_label_target_label_mod = model.addVar(vtype=gr.GRB.BINARY)
        model.addGenConstrAnd(source_label_target_label_mod,[node_variables[source]["LABEL"],node_variables[target]["LABEL_MOD"]])

        source_label_mod_target_label = model.addVar(vtype=gr.GRB.BINARY)
        model.addGenConstrAnd(source_label_mod_target_label,[node_variables[source]["LABEL_MOD"],node_variables[target]["LABEL"]])

        model.addGenConstrIndicator(instance_edge_var , True, core_instance_endpoints + source_label_mod_target_label + source_label_target_label_mod == 1)

    return

def require_endpoints_for_frame_arcs(model,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,edge_endpoint_conjunctions):

    for edge in edge_variables:


        frame_edge_var = edge_variables[edge]["FRAME"]
        source,target = edge

        core_frame_endpoints = edge_endpoint_conjunctions["CORE_FRAME_CORE_FRAME"][edge][0]
        source_cf_target_ci = edge_endpoint_conjunctions["CORE_FRAME_CORE_INSTANCE"][edge][0]
        source_ci_target_cf = edge_endpoint_conjunctions["CORE_FRAME_CORE_INSTANCE"][edge][1]

        source_theme_target_theme_mod = add_conjunction_between_two_variables(model,node_variables[source]["THEME"],node_variables[target]["THEME_MOD"])
        source_theme_mod_target_theme = add_conjunction_between_two_variables(model,node_variables[source]["THEME_MOD"],node_variables[target]["THEME"])
        source_unit_target_unit_restriction = add_conjunction_between_two_variables(model,node_variables[source]["UNIT"],node_variables[target]["UNIT_RESTRICTION"])
        source_unit_restriction_target_unit = add_conjunction_between_two_variables(model,node_variables[source]["UNIT_RESTRICTION"],node_variables[target]["UNIT"])

        label_endpoints = edge_endpoint_conjunctions["LABEL_LABEL"][edge][0]
        value_endpoints = edge_endpoint_conjunctions["VALUE_VALUE"][edge][0]
        valuem_endpoints = edge_endpoint_conjunctions["VALUEM_VALUEM"][edge][0]

        all_conjs = [ core_frame_endpoints ,
                      source_cf_target_ci ,
                      source_ci_target_cf,
                      source_theme_target_theme_mod ,
                      source_theme_mod_target_theme ,
                      source_unit_target_unit_restriction,
                      source_unit_restriction_target_unit ,
                      label_endpoints ,
                      value_endpoints ,
                      valuem_endpoints ]


        model.addGenConstrIndicator(frame_edge_var, True,  gr.quicksum(all_conjs) == 1)

    return


def require_alo_label_label_frame_edge(model,edge_variables,edge_endpoint_conjunctions):
    endpoint_conjunctions = edge_endpoint_conjunctions["LABEL_LABEL"]

    z = model.addVar(vtype=gr.GRB.BINARY)

    label_label_frame_edge_variables = []
    for edge in endpoint_conjunctions:

        this_edge = edge_variables[edge]["FRAME"]
        endpoint_conjunction = endpoint_conjunctions[edge][0]

        this_var = model.addVar(vtype=gr.GRB.BINARY)
        model.addGenConstrAnd(this_var,[this_edge,endpoint_conjunction])

        label_label_frame_edge_variables.append(this_var)

    model.addConstr( gr.quicksum(label_label_frame_edge_variables) >= 1 )

    return


def require_alo_y_for_attr_x(model,anchor,dependent,edge_type,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,target_clamped_edge_conjunctions):

    conjunction_string = dependent + "_" + edge_type
    target_clamped_conjunctions = target_clamped_edge_conjunctions[conjunction_string]

    for source in node_variables:
        source_var = None
        if anchor == "CORE_INSTANCE":
            source_var = core_instance_attr_indicators[source]
        elif anchor == "CORE_FRAME":
            source_var = core_frame_attr_indicators[source]
        else:
            source_var = node_variables[source][anchor]

        these_target_clamped_conjs = [ target_clamped_conjunctions[source][edge] for edge in target_clamped_conjunctions[source] ]

        model.addGenConstrIndicator(source_var, True, gr.quicksum(these_target_clamped_conjs) >= 1)

    return


def require_exactly_one_y_for_attr_x(model,anchor,dependent,edge_type,node_variables,edge_variables,target_clamped_edge_conjunctions ):

    conjunction_string = dependent + "_" + edge_type
    target_clamped_conjunctions = target_clamped_edge_conjunctions[conjunction_string]

    for source in node_variables:
        source_var = None
        if anchor == "CORE_INSTANCE":
            source_var = core_instance_attr_indicators[source]
        elif anchor == "CORE_FRAME":
            source_var = core_frame_attr_indicators[source]
        else:
            source_var = node_variables[source][anchor]

        these_target_clamped_conjs = [ target_clamped_conjunctions[source][edge] for edge in target_clamped_conjunctions[source] ]

        model.addGenConstrIndicator(source_var, True, gr.quicksum(these_target_clamped_conjs) == 1)

    return

def require_frame_triangle_constraints(model,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,value_valuem_attr_indicators,edge_endpoint_conjunctions):
    # unpack stuff
    cf_cf_endpoints = edge_endpoint_conjunctions["CORE_FRAME_CORE_FRAME"]
    cf_ci_endpoints = edge_endpoint_conjunctions["CORE_FRAME_CORE_INSTANCE"]
    label_label_endpoints = edge_endpoint_conjunctions["LABEL_LABEL"]
    value_value_endpoints = edge_endpoint_conjunctions["VALUE_VALUE"]
    valuem_valuem_endpoints = edge_endpoint_conjunctions["VALUEM_VALUEM"]

    # for keeping track of cases where there is no proposed arc, avoid duplicate variables
    dummy_variables = {}

    for node in node_variables:

        frame_edges_from_node = [ edge for edge in edge_variables if node in edge ]

        if len(frame_edges_from_node) > 1:

            for i in range(0,len(frame_edges_from_node)-1):

                target = get_other_node_in_edge(node,frame_edges_from_node[i])
                frame_edge_s_t = edge_variables[frame_edges_from_node[i]]["FRAME"]

                cf_s_ci_t = access_clamped_edge_dictionaries(node,target,cf_ci_endpoints)
                ci_s_cf_t = access_clamped_edge_dictionaries(node,target,cf_ci_endpoints,reverse=True)

                value_valuem_source = value_valuem_attr_indicators[node]
                cf_t = core_frame_attr_indicators[target]

                label_s_label_t = access_clamped_edge_dictionaries(node,target,label_label_endpoints)
                value_s_value_t = access_clamped_edge_dictionaries(node,target,value_value_endpoints)
                valuem_s_valuem_t = access_clamped_edge_dictionaries(node,target,valuem_valuem_endpoints)

                for j in range(i+1,len(frame_edges_from_node)):

                    target_ = get_other_node_in_edge(node,frame_edges_from_node[j])
                    frame_edge_s_t_ = edge_variables[frame_edges_from_node[j]]["FRAME"]

                    cf_t_ = core_frame_attr_indicators[target_]
                    ci_t_ = core_instance_attr_indicators[target_]
                    label_t_ = node_variables[target_]["LABEL"]
                    value_t_ = node_variables[target_]["VALUE"]
                    valuem_t_ = node_variables[target_]["VALUEM"]

                    ## get triangle-completing frame edge variable, if none exists, instantiate dummy variable and force it equal to zero
                    frame_edge_t_t_ = get_edge_variable(target,target_,edge_variables,"FRAME")

                    if frame_edge_t_t_ == None:
                        if (target,target_) in dummy_variables:
                            frame_edge_t_t_ = dummy_variables[(target,target_)]
                        elif (target_,target) in dummy_variables:
                            frame_edge_t_t_ = dummy_variables[(target_,target)]
                        else:
                            frame_edge_t_t_ = model.addVar(vtype=gr.GRB.BINARY)
                            model.addConstr( frame_edge_t_t_ == 0 )
                            dummy_variables[(target,target_)] = frame_edge_t_t_

                    # add frame edge_conjunction
                    open_triangle_conjunction = add_conjunction_between_two_variables(model,frame_edge_s_t,frame_edge_s_t_)

                    # condition_1
                    # source core frame, target core frame, target_ core frame
                    '''
                    condition_1 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_1,[open_triangle_conjunction, cf_s_cf_t, cf_t_ ])
                    model.addGenConstrIndicator(condition_1, True, frame_edge_t_t_ == 1)
                    '''
                    '''
                    # condition_2
                    # source core frame, target core frame, target_ core instance
                    condition_2 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_2,[open_triangle_conjunction, cf_s_cf_t, ci_t_ ])
                    model.addGenConstrIndicator(condition_2, True, frame_edge_t_t_ == 1)
                    '''
                    '''
                    # condition_3
                    # source core frame, target core instance, target_ core frame
                    condition_3 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_3,[open_triangle_conjunction, cf_s_ci_t, cf_t_ ])
                    model.addGenConstrIndicator(condition_3, True, frame_edge_t_t_ == 1)
                    '''

                    '''
                    # condition_5
                    # label source, label target, core frame target_
                    condition_5 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_5,[open_triangle_conjunction, label_s_label_t , cf_t_ ])
                    model.addGenConstrIndicator(condition_5, True, frame_edge_t_t_ == 1)
                    '''
                    '''
                    # condition_6
                    # label source, label target, core frame target_
                    condition_6 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_6,[open_triangle_conjunction, ci_s_cf_t , label_t_ ])
                    model.addGenConstrIndicator(condition_6, True, frame_edge_t_t_ == 1)
                    '''
                    # condition_1
                    # source core value/valuem target core frame, target_ core frame
                    condition_1 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_1,[open_triangle_conjunction, value_valuem_source, cf_t, cf_t_])
                    model.addGenConstrIndicator(condition_1, True, frame_edge_t_t_ == 1)

                    # condition_2
                    # value source, value target, core frame target_
                    condition_2 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_2,[open_triangle_conjunction, value_s_value_t , cf_t_ ])
                    model.addGenConstrIndicator(condition_2, True, frame_edge_t_t_ == 1)

                    # condition_3
                    # value source, core frame target , value_t
                    condition_3 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_3,[open_triangle_conjunction, ci_s_cf_t , value_t_ ])
                    model.addGenConstrIndicator(condition_3, True, frame_edge_t_t_ == 1)

                    # condition_4
                    # valuem source, valuem target, core frame target_
                    condition_4 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_4,[open_triangle_conjunction, valuem_s_valuem_t , cf_t_ ])
                    model.addGenConstrIndicator(condition_4, True, frame_edge_t_t_ == 1)

                    # condition_5
                    # valuem source, core frame target , valuem_t
                    condition_5 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_5,[open_triangle_conjunction, ci_s_cf_t , valuem_t_ ])
                    model.addGenConstrIndicator(condition_5, True, frame_edge_t_t_ == 1)


def require_mixed_edge_constraints(model,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,value_valuem_attr_indicators,edge_endpoint_conjunctions,target_clamped_edge_conjunctions):
    # unpack stuff

    # cf_ci_endpoints = edge_endpoint_conjunctions["CORE_FRAME_CORE_INSTANCE"]
    value_clamped_instances = target_clamped_edge_conjunctions["VALUE_INSTANCE"]

    dummy_variables = {}
    for node in node_variables:

        label_s = node_variables[node]["LABEL"]

        edges_from_node = [ edge for edge in edge_variables if node in edge ]

        if len(edges_from_node) > 1:

            for i in range(0,len(edges_from_node)-1):

                target = get_other_node_in_edge(node,edges_from_node[i])

                frame_edge_s_t = edge_variables[edges_from_node[i]]["FRAME"]
                instance_edge_s_t = edge_variables[edges_from_node[i]]["INSTANCE"]

                value_valuem_s = value_valuem_attr_indicators[node]
                value_valuem_t = value_valuem_attr_indicators[target]
                cf_t = core_frame_attr_indicators[target]
                label_t = node_variables[target]["LABEL"]

                instance_edge_value_t = value_clamped_instances[node][edges_from_node[i]]

                for j in range(i+1,len(edges_from_node)):

                    target_ = get_other_node_in_edge(node,edges_from_node[j])

                    frame_edge_s_t_ = edge_variables[edges_from_node[j]]["FRAME"]
                    instance_edge_s_t_ = edge_variables[edges_from_node[j]]["INSTANCE"]

                    value_valuem_t_ = value_valuem_attr_indicators[target_]
                    cf_t_ = core_frame_attr_indicators[target_]
                    label_t_ = node_variables[target_]["LABEL"]

                    instance_edge_value_t_ = value_clamped_instances[node][edges_from_node[j]]

                    ## get triangle-completing frame edge variable, if none exists, instantiate dummy variable and force it equal to zero
                    frame_edge_t_t_ = get_edge_variable(target,target_,edge_variables,"FRAME")

                    if frame_edge_t_t_ is None:
                        if (target,target_) in dummy_variables:
                            frame_edge_t_t_ = dummy_variables[(target,target_)]
                        elif (target_,target) in dummy_variables:
                            frame_edge_t_t_ = dummy_variables[(target_,target)]
                        else:
                            frame_edge_t_t_ = model.addVar(vtype=gr.GRB.BINARY)
                            model.addConstr( frame_edge_t_t_ == 0 )
                            dummy_variables[(target,target_)] = frame_edge_t_t_

                    open_triangle_conjunction = add_conjunction_between_two_variables(model,frame_edge_s_t,instance_edge_s_t_)
                    open_triangle_conjunction_ = add_conjunction_between_two_variables(model,instance_edge_s_t,frame_edge_s_t_)

                    # condition_1
                    # source value/valuem, target core frame, target_ value_valuem
                    condition_1 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_1,[open_triangle_conjunction, value_valuem_s, cf_t , value_valuem_t_ ])
                    model.addGenConstrIndicator(condition_1, True, frame_edge_t_t_ == 1)

                    # condition_2
                    # source value/valuem, target value/valuem, target_ coreframe
                    condition_2 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_2,[open_triangle_conjunction_, value_valuem_s, value_valuem_t , cf_t_ ])
                    model.addGenConstrIndicator(condition_2, True, frame_edge_t_t_ == 1)

                    # condition_3
                    # source value/valuem, target label, target_ coreframe
                    condition_3 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_3,[open_triangle_conjunction, value_valuem_s, cf_t , label_t_ ])
                    model.addGenConstrIndicator(condition_3, True, frame_edge_t_t_ == 1)

                    # condition_4
                    # source value/valuem, target label, target_ coreframe
                    condition_4 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_4,[open_triangle_conjunction_, value_valuem_s, label_t , cf_t_ ])
                    model.addGenConstrIndicator(condition_4, True, frame_edge_t_t_ == 1)

                    # (negative) condition 5
                    # source label , target value , target_ value --> then target and target_ cannot be of the same frame !
                    condition_5 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_5,[label_s, instance_edge_value_t, instance_edge_value_t_])
                    model.addGenConstrIndicator(condition_5, True, frame_edge_t_t_ == 0)

    return


def require_instance_triangle_constraints(model,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,edge_endpoint_conjunctions,target_clamped_edge_conjunctions):
    '''
    # value - label / value - value-m --> value-m - label
    # value-m - label / value-m - value --> value - label
    '''
    label_clamped_instances = target_clamped_edge_conjunctions["LABEL_INSTANCE"]
    value_clamped_instances = target_clamped_edge_conjunctions["VALUE_INSTANCE"]

    dummy_variables = {}
    for source in node_variables:

        value_s = node_variables[source]["VALUE"]
        valuem_s  = node_variables[source]["VALUEM"]

        edges_from_node = [ edge for edge in edge_variables if source in edge  ]

        for i in range(len(edges_from_node)-1):

            target = get_other_node_in_edge(source,edges_from_node[i])
            instance_edge_s_t = edge_variables[edges_from_node[i]]["INSTANCE"]

            label_clamped_instance_edge_t = label_clamped_instances[source][edges_from_node[i]]
            value_clamped_instance_edge_t = value_clamped_instances[source][edges_from_node[i]]

            valuem_t = node_variables[target]["VALUEM"]

            for j in range(i+1,len(edges_from_node)):

                target_ = get_other_node_in_edge(source,edges_from_node[j])
                instance_edge_s_t_ = edge_variables[edges_from_node[j]]["INSTANCE"]

                label_clamped_instance_edge_t_ = label_clamped_instances[source][edges_from_node[j]]
                value_clamped_instance_edge_t_ = value_clamped_instances[source][edges_from_node[j]]

                valuem_t_ = node_variables[target_]["VALUEM"]

                instance_edge_t_t_ = get_edge_variable(target,target_,edge_variables,"INSTANCE")
                if instance_edge_t_t_ is None:
                    if (target,target_) in dummy_variables:
                        instance_edge_t_t_ = dummy_variables[(target,target_)]
                    elif (target_,target) in dummy_variables:
                        instance_edge_t_t_ = dummy_variables[(target_,target)]
                    else:
                        instance_edge_t_t_ = model.addVar(vtype=gr.GRB.BINARY)
                        model.addConstr( instance_edge_t_t_ == 0 )
                        dummy_variables[(target,target_)] = instance_edge_t_t_

                #value - label - value-m
                condition_1 = model.addVar(vtype=gr.GRB.BINARY)
                model.addGenConstrAnd(condition_1, [value_s, label_clamped_instance_edge_t , valuem_t_ , instance_edge_s_t_ ])
                model.addGenConstrIndicator(condition_1, True, instance_edge_t_t_ == 1)

                #value - label - value-m
                condition_2 = model.addVar(vtype=gr.GRB.BINARY)
                model.addGenConstrAnd(condition_2, [value_s, label_clamped_instance_edge_t_ , valuem_t , instance_edge_s_t ])
                model.addGenConstrIndicator(condition_2, True, instance_edge_t_t_ == 1)

                #valuem - label - value
                condition_3 = model.addVar(vtype=gr.GRB.BINARY)
                model.addGenConstrAnd(condition_3, [valuem_s, label_clamped_instance_edge_t , value_clamped_instance_edge_t_ ])
                model.addGenConstrIndicator(condition_3, True, instance_edge_t_t_ == 1)

                #valuem - label - value
                condition_4 = model.addVar(vtype=gr.GRB.BINARY)
                model.addGenConstrAnd(condition_4, [valuem_s, label_clamped_instance_edge_t_ , value_clamped_instance_edge_t ])
                model.addGenConstrIndicator(condition_4, True, instance_edge_t_t_ == 1)


'''
# ILP for numerical contrast parsing
'''
def construct_and_solve_sentence_ilp(sentence_datum):
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
        node_variables[node] = { label:instantiate_variable(model,label,"node",node,node_scores[i].get(label)) for label in all_node_labels }

    edge_variables = {}
    for i,edge in enumerate(edges):
        edge_variables[edge] = { label:instantiate_variable(model,label,"edge",edge,edge_scores[i].get(label)) for label in all_edge_labels }

    core_frame_attr_indicators = {}
    for node in node_variables:
        core_frame_attr_indicators[node] = add_disjunction_over_variable_set(model,node,node_variables,core_frame_labels)
        core_frame_attr_indicators[node].VarName = stringify_node(node) + "_CORE_FRAME"

    core_instance_attr_indicators = {}
    for node in node_variables:
        core_instance_attr_indicators[node] = add_disjunction_over_variable_set(model,node,node_variables,core_instance_labels)
        core_instance_attr_indicators[node].VarName = stringify_node(node) + "_CORE_INSTANCE"

    value_valuem_attr_indicators = {}
    for node in node_variables:
        value_valuem_attr_indicators[node] = add_disjunction_over_variable_set(model,node,node_variables,["VALUE","VALUEM"])
        core_instance_attr_indicators[node].VarName = stringify_node(node) + "_VALUE_VALUEM_DISJUNCTION"

    # resuable conjunction variables
    edge_endpoint_conjunctions = {}
    for attr, attr_ in endpoint_clamp_tuples:
        key,value = instantiate_clamped_endpoint_edge_constraints(model,attr,attr_,edge_variables,node_variables,core_frame_attr_indicators,core_instance_attr_indicators)
        edge_endpoint_conjunctions[key] = value

    target_clamped_edge_conjunctions = {}
    for attr, edge_type in clamped_edge_tuples:
        key,value = instantiate_partial_clamped_endpoint_edge_constraints(model,attr,edge_type,edge_variables,node_variables,core_frame_attr_indicators,core_instance_attr_indicators)
        target_clamped_edge_conjunctions[key] = value

    '''
    # Apply constraints to model
    '''
    # node choose one constraints
    require_choose_one(model,node_variables,"node")
    # edge choose one constraints
    require_choose_one(model,edge_variables,"edge")

    # no two overlapping spans can have non-null labels
    require_node_span_non_overlap(model,node_variables)

    # arc endpoint constraints
    require_distinct_instance_arc_endpoints(model,node_variables,edge_variables)
    require_instance_labels_for_instance_arcs(model,node_variables,edge_variables,core_instance_attr_indicators)
    require_endpoints_for_frame_arcs(model,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,edge_endpoint_conjunctions)
    
    # require any solution to contain at least one frame edge between two labels
    require_alo_label_label_frame_edge(model,edge_variables,edge_endpoint_conjunctions)
    require_alo_y_for_attr_x(model,"LABEL","VALUE","INSTANCE",node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,target_clamped_edge_conjunctions)
    require_alo_y_for_attr_x(model,"VALUE","VALUE","FRAME",node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,target_clamped_edge_conjunctions)
    require_alo_y_for_attr_x(model,"CORE_FRAME","LABEL","FRAME",node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,target_clamped_edge_conjunctions)

    # require any solution to contain exactly one dependent for each anchor, if it appears
    require_exactly_one_y_for_attr_x(model,"VALUE","LABEL","INSTANCE",node_variables,edge_variables,target_clamped_edge_conjunctions)
    require_exactly_one_y_for_attr_x(model,"THEME_MOD","THEME","FRAME",node_variables,edge_variables,target_clamped_edge_conjunctions)
    require_exactly_one_y_for_attr_x(model,"UNIT_RESTRICTION","UNIT","FRAME",node_variables,edge_variables,target_clamped_edge_conjunctions)
    require_exactly_one_y_for_attr_x(model,"LABEL_MOD","LABEL","INSTANCE",node_variables,edge_variables,target_clamped_edge_conjunctions)

    # triangle constraints
    require_frame_triangle_constraints(model,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,value_valuem_attr_indicators,edge_endpoint_conjunctions)
    require_mixed_edge_constraints(model,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,value_valuem_attr_indicators,edge_endpoint_conjunctions,target_clamped_edge_conjunctions)
    require_instance_triangle_constraints(model,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,edge_endpoint_conjunctions,target_clamped_edge_conjunctions)


    model.update()
    model.optimize()

    if model.Status == gr.GRB.INFEASIBLE:
        return {span: Counter({"NULL":1.}) for span in node_variables}, {(span, span_): Counter({"NULL":1.}) for span, span_ in edge_variables}
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
        "nodes": [(span, {(key and key.upper()) or "NULL": value for key, value in attr.items()}) for span, attr, _, _ in graph.nodes],
        "edges": [((span, span_), {(key and key.upper()) or "NULL": value for key, value in attr.items()}) for span, span_, attr in graph.edges],
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

def test_alo_label_label_frame_edge():


    sentence_datum = { "nodes":[ ((0,1),{"LABEL":.1,"THEME":.9}),
                                 ((1,2),{"LABEL":.1,"THEME":.9})],
                       "edges":[ (((0,1),(1,2)),{"FRAME":.5}) ] }

    nodes,node_scores = zip(*sentence_datum["nodes"])
    edges,edge_scores = zip(*sentence_datum["edges"])

    ## pad
    for score_dict in node_scores:
        for label in node_labels:
            if label not in score_dict:
                score_dict[label] = 0.0

    for score_dict in edge_scores:
        for label in edge_labels:
            if label not in score_dict:
                score_dict[label] = 0.0

    model = gr.Model()

    node_variables = {}
    for i,node in enumerate(nodes):
        node_variables[node] = { label:instantiate_variable(model,label,"node",node,node_scores[i].get(label)) for label in node_scores[i] }

    edge_variables = {}
    for i,edge in enumerate(edges):
        edge_variables[edge] = { label:instantiate_variable(model,label,"edge",edge,edge_scores[i].get(label)) for label in edge_scores[i] }

    core_instance_attr_indicators = {}
    for node in node_variables:
        core_instance_attr_indicators[node] = add_disjunction_over_variable_set(model,node,node_variables,core_instance_labels)

    core_frame_attr_indicators = {}
    for node in node_variables:
        core_frame_attr_indicators[node] = add_disjunction_over_variable_set(model,node,node_variables,core_frame_labels)

    # resuable conjunction variables
    edge_endpoint_conjunctions = {}
    for attr, attr_ in endpoint_clamp_tuples:
        key,value = instantiate_clamped_endpoint_edge_constraints(model,attr,attr_,edge_variables,node_variables,core_frame_attr_indicators,core_instance_attr_indicators)
        edge_endpoint_conjunctions[key] = value

    require_choose_one(model,node_variables,"node")
    require_choose_one(model,edge_variables,"edge")


    require_endpoints_for_frame_arcs(model,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,edge_endpoint_conjunctions)

    model.update()
    model.optimize()

    print( "BEFORE" )
    for edge in edge_variables:
        print( edge )
        source,target = edge
        for label in edge_variables[edge]:
            print( label, edge_variables[edge][label].X )
        print( source )
        for label in node_variables[source]:
            print( label, node_variables[source][label].X )
        print( target )
        for label in node_variables[target]:
            print( label, node_variables[target][label].X )

    print( "\n" )

    require_alo_label_label_frame_edge(model,edge_variables,edge_endpoint_conjunctions)

    model.update()
    model.optimize()

    print( "AFTER" )
    for edge in edge_variables:
        print( edge )
        source,target = edge
        for label in edge_variables[edge]:
            print( label, edge_variables[edge][label].X )
        print( source )
        for label in node_variables[source]:
            print( label, node_variables[source][label].X )
        print( target )
        for label in node_variables[target]:
            print( label, node_variables[target][label].X )

def test_alo_frame_label_edge():

    print( "\n" )
    print( "test_alo_frame_label_edge_from_each_label")

    sentence_datum = { "nodes":[ ((0,1),{"THEME":.99,"NULL":.01}),
                                 ((1,2),{"LABEL":.99, "NULL":.01}),
                                 ((2,3),{"LABEL":.99,"NULL":.01})
                                  ],
                       "edges":[ (((0,1),(1,2)),{"FRAME":.1, "NULL":.9} ),
                                 (((1,2),(2,3)),{"FRAME":.1, "NULL":.9} )   ] }

    nodes,node_scores = zip(*sentence_datum["nodes"])
    edges,edge_scores = zip(*sentence_datum["edges"])

    ## pad
    for score_dict in node_scores:
        for label in node_labels:
            if label not in score_dict:
                score_dict[label] = 0.0

    for score_dict in edge_scores:
        for label in edge_labels:
            if label not in score_dict:
                score_dict[label] = 0.0

    model = gr.Model()

    node_variables = {}
    for i,node in enumerate(nodes):
        node_variables[node] = { label:instantiate_variable(model,label,"node",node,node_scores[i].get(label)) for label in node_scores[i] }

    edge_variables = {}
    for i,edge in enumerate(edges):
        edge_variables[edge] = { label:instantiate_variable(model,label,"edge",edge,edge_scores[i].get(label)) for label in edge_scores[i] }

    core_instance_attr_indicators = {}
    for node in node_variables:
        core_instance_attr_indicators[node] = add_disjunction_over_variable_set(model,node,node_variables,core_instance_labels)

    core_frame_attr_indicators = {}
    for node in node_variables:
        core_frame_attr_indicators[node] = add_disjunction_over_variable_set(model,node,node_variables,core_frame_labels)

    # resuable conjunction variables
    edge_endpoint_conjunctions = {}
    for attr, attr_ in endpoint_clamp_tuples:
        key,value = instantiate_clamped_endpoint_edge_constraints(model,attr,attr_,edge_variables,node_variables,core_frame_attr_indicators,core_instance_attr_indicators)
        edge_endpoint_conjunctions[key] = value


    target_clamped_edge_conjunctions = {}
    for attr, edge_type in clamped_edge_tuples:
        key,value = instantiate_partial_clamped_endpoint_edge_constraints(model,attr,edge_type,edge_variables,node_variables,core_frame_attr_indicators,core_instance_attr_indicators)
        target_clamped_edge_conjunctions[key] = value

    require_choose_one(model,node_variables,"node")
    require_choose_one(model,edge_variables,"edge")


    model.update()
    model.optimize()

    print( "BEFORE" )
    for edge in edge_variables:
        print( edge )
        source,target = edge
        for label in edge_variables[edge]:
            print( label, edge_variables[edge][label].X )
        print( source )
        for label in node_variables[source]:
            print( label, node_variables[source][label].X )
        print( target )
        for label in node_variables[target]:
            print( label, node_variables[target][label].X )

    print( "\n" )

    require_alo_y_for_attr_x(model,"LABEL","LABEL","FRAME",node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,target_clamped_edge_conjunctions)
    #require_alo_y_for_attr_x(model,"VALUE","VALUE","FRAME",node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,target_clamped_edge_conjunctions)
    require_alo_y_for_attr_x(model,"CORE_FRAME","LABEL","FRAME",node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,target_clamped_edge_conjunctions)

    model.update()
    model.optimize()

    print( "AFTER" )
    for edge in edge_variables:
        print( edge )
        source,target = edge
        for label in edge_variables[edge]:
            print( label, edge_variables[edge][label].X )
        print( source )
        for label in node_variables[source]:
            print( label, node_variables[source][label].X )
        print( target )
        for label in node_variables[target]:
            print( label, node_variables[target][label].X )

    print( "\n" )


def test_frame_triangles():

    print( "\n" )
    print( "testing frame triangle constraints")

    sentence_datum = { "nodes" :[  ((0, 1),{"THEME":.99,"NULL":.01}),
                                   ((1, 2),{"THEME":.99,"NULL":.01}),
                                   ((2, 3),{"THEME":.99,"NULL":.01}),

                                   ((3, 4),{"THEME":.99,"NULL":.01}),
                                   ((4, 5),{"THEME":.99,"NULL":.01}),
                                   ((5, 6),{"VALUE":.99,"NULL":.01}),

                                   ((6, 7),{"THEME":.99,"NULL":.01}),
                                   ((7, 8),{"VALUE":.99,"NULL":.01}),
                                   ((8, 9),{"THEME":.99,"NULL":.01}),

                                   ((9, 10),{"LABEL":.99,"NULL":.01}),
                                   ((10, 11),{"THEME":.99,"NULL":.01}),
                                   ((11, 12),{"THEME":.99,"NULL":.01}),

                                   ((12, 13),{"LABEL":.99,"NULL":.01}),
                                   ((13, 14),{"LABEL":.99,"NULL":.01}),
                                   ((14, 15),{"THEME":.99,"NULL":.01}),

                                   ((15, 16),{"LABEL":.99,"NULL":.01}),
                                   ((16, 17),{"THEME":.99,"NULL":.01}),
                                   ((17, 18),{"LABEL":.99,"NULL":.01}),

                                   ((18, 19),{"VALUE":.99,"NULL":.01}),
                                   ((19, 20),{"VALUE":.99,"NULL":.01}),
                                   ((20, 21),{"POSSESSOR":.99,"NULL":.01}),

                                   ((21, 22),{"VALUE":.99,"NULL":.01}),
                                   ((22, 23),{"POSSESSOR":.99,"NULL":.01}),
                                   ((23, 24),{"VALUE":.99,"NULL":.01}),

                                   ((24, 25),{"VALUEM":.99,"NULL":.01}),
                                   ((25, 26),{"VALUEM":.99,"NULL":.01}),
                                   ((26, 27),{"POSSESSOR":.99,"NULL":.01}),

                                   ((27, 28),{"VALUEM":.99,"NULL":.01}),
                                   ((28, 29),{"POSSESSOR":.99,"NULL":.01}),
                                   ((29, 30),{"VALUEM":.99,"NULL":.01}),

                                ],

                      "edges" : [ (((0, 1), (1, 2)), {'NULL': 0.01, 'FRAME': 0.99}),
                                  (((0, 1), (2, 3)), {'NULL': 0.01, 'FRAME': 0.99}),
                                  (((1, 2), (2, 3)), {'NULL': 0.9, 'FRAME': 0.1}),

                                (((3, 4), (4, 5)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((3, 4), (5, 6)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((4, 5), (5, 6)), {'NULL': 0.9, 'FRAME': 0.1}),

                                (((6, 7), (7, 8)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((6, 7), (8, 9)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((7, 8), (8, 9)), {'NULL': 0.9, 'FRAME': 0.1}),

                                (((9, 10), (10, 11)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((9, 10), (11, 12)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((10, 11), (11, 12)), {'NULL': 0.9, 'FRAME': 0.1}),

                                (((12, 13), (13, 14)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((12, 13), (14, 15)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((13, 14), (14, 15)), {'NULL': 0.9, 'FRAME': 0.1}),

                                (((15, 16), (16, 17)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((15, 16), (17, 18)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((16, 17), (17, 18)), {'NULL': 0.9, 'FRAME': 0.1}),

                                (((18, 19), (19, 20)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((18, 19), (20, 21)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((19, 20), (20, 21)), {'NULL': 0.9, 'FRAME': 0.1}),

                                (((21, 22), (22, 23)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((21, 22), (23, 24)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((22, 23), (23, 24)), {'NULL': 0.9, 'FRAME': 0.1}),

                                (((24, 25), (25, 26)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((24, 25), (26, 27)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((25, 26), (26, 27)), {'NULL': 0.9, 'FRAME': 0.1}),

                                (((27, 28), (28, 29)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((27, 28), (29, 30)), {'NULL': 0.01, 'FRAME': 0.99}),
                                (((28, 29), (29, 30)), {'NULL': 0.9, 'FRAME': 0.1}),

                                ] }

    nodes,node_scores = zip(*sentence_datum["nodes"])
    edges,edge_scores = zip(*sentence_datum["edges"])

    ## pad
    for score_dict in node_scores:
        for label in node_labels:
            if label not in score_dict:
                score_dict[label] = 0.0

    for score_dict in edge_scores:
        for label in edge_labels:
            if label not in score_dict:
                score_dict[label] = 0.0

    model = gr.Model()

    node_variables = {}
    for i,node in enumerate(nodes):
        node_variables[node] = { label:instantiate_variable(model,label,"node",node,node_scores[i].get(label)) for label in node_scores[i] }

    edge_variables = {}
    for i,edge in enumerate(edges):
        edge_variables[edge] = { label:instantiate_variable(model,label,"edge",edge,edge_scores[i].get(label)) for label in edge_scores[i] }

    core_instance_attr_indicators = {}
    for node in node_variables:
        core_instance_attr_indicators[node] = add_disjunction_over_variable_set(model,node,node_variables,core_instance_labels)

    core_frame_attr_indicators = {}
    for node in node_variables:
        core_frame_attr_indicators[node] = add_disjunction_over_variable_set(model,node,node_variables,core_frame_labels)

    # resuable conjunction variables
    edge_endpoint_conjunctions = {}
    for attr, attr_ in endpoint_clamp_tuples:
        key,value = instantiate_clamped_endpoint_edge_constraints(model,attr,attr_,edge_variables,node_variables,core_frame_attr_indicators,core_instance_attr_indicators)
        edge_endpoint_conjunctions[key] = value

    require_choose_one(model,node_variables,"node")
    require_choose_one(model,edge_variables,"edge")

    model.update()
    model.optimize()

    print( "BEFORE" )
    for i,edge in enumerate(edge_variables):
        if i % 3 == 0:
            print( edge )
            print( label, edge_variables[edge][label].X )


    print( "\n" )

    require_frame_triangle_constraints(model,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,edge_endpoint_conjunctions)

    model.update()
    model.optimize()

    print( "AFTER" )
    for i,edge in enumerate(edge_variables):
        if i % 3 == 0:
            print( edge )
            for label in edge_variables[edge]:
                print( label, edge_variables[edge][label].X )


    print( "\n" )

def test_exactly_one_value_label_instance_edge():

    print( "\n" )
    print( "test_alo_frame_label_edge_from_each_label")

    sentence_datum = { "nodes":[ ((0,1),{"VALUE":.999,"NULL":.001}),
                                 ((1,2),{"LABEL":.1, "NULL":.8})
                                  ],
                       "edges":[ (((0,1),(1,2)),{"INSTANCE":.4, "NULL":.6} ) ]
                       }

    nodes,node_scores = zip(*sentence_datum["nodes"])
    edges,edge_scores = zip(*sentence_datum["edges"])

    ## pad
    for score_dict in node_scores:
        for label in node_labels:
            if label not in score_dict:
                score_dict[label] = 0.0

    for score_dict in edge_scores:
        for label in edge_labels:
            if label not in score_dict:
                score_dict[label] = 0.0

    model = gr.Model()

    node_variables = {}
    for i,node in enumerate(nodes):
        node_variables[node] = { label:instantiate_variable(model,label,"node",node,node_scores[i].get(label)) for label in node_scores[i] }

    edge_variables = {}
    for i,edge in enumerate(edges):
        edge_variables[edge] = { label:instantiate_variable(model,label,"edge",edge,edge_scores[i].get(label)) for label in edge_scores[i] }

    core_instance_attr_indicators = {}
    for node in node_variables:
        core_instance_attr_indicators[node] = add_disjunction_over_variable_set(model,node,node_variables,core_instance_labels)

    core_frame_attr_indicators = {}
    for node in node_variables:
        core_frame_attr_indicators[node] = add_disjunction_over_variable_set(model,node,node_variables,core_frame_labels)

    # resuable conjunction variables
    edge_endpoint_conjunctions = {}
    for attr, attr_ in endpoint_clamp_tuples:
        key,value = instantiate_clamped_endpoint_edge_constraints(model,attr,attr_,edge_variables,node_variables,core_frame_attr_indicators,core_instance_attr_indicators)
        edge_endpoint_conjunctions[key] = value

    target_clamped_edge_conjunctions = {}
    for attr, edge_type in clamped_edge_tuples:
        key,value = instantiate_partial_clamped_endpoint_edge_constraints(model,attr,edge_type,edge_variables,node_variables,core_frame_attr_indicators,core_instance_attr_indicators)
        target_clamped_edge_conjunctions[key] = value

    require_choose_one(model,node_variables,"node")
    require_choose_one(model,edge_variables,"edge")


    model.update()
    model.optimize()

    print( "BEFORE" )
    for edge in edge_variables:
        print( edge )
        source,target = edge
        for label in edge_variables[edge]:
            print( label, edge_variables[edge][label].X )
        print( source )
        for label in node_variables[source]:
            print( label, node_variables[source][label].X )
        print( target )
        for label in node_variables[target]:
            print( label, node_variables[target][label].X )

    print( "\n" )

    require_exactly_one_y_for_attr_x(model,"VALUE","LABEL","INSTANCE",node_variables,edge_variables,target_clamped_edge_conjunctions)

    model.update()
    model.optimize()

    print( "AFTER" )
    for edge in edge_variables:
        print( edge )
        source,target = edge
        for label in edge_variables[edge]:
            print( label, edge_variables[edge][label].X )
        print( source )
        for label in node_variables[source]:
            print( label, node_variables[source][label].X )
        print( target )
        for label in node_variables[target]:
            print( label, node_variables[target][label].X )

    print( "\n" )

'''
def test_mixed_triangles():

    print( "\n" )
    print( "testing mixed constraints")

    sentence_datum = { "nodes" :[  ((0, 1),{"VALUE":.99,"NULL":.01}),
                                   ((1, 2),{"LABEL":.99,"NULL":.01}),
                                   ((2, 3),{"VALUEM":.99,"NULL":.01}),

                                   ((3, 4),{"VALUE":.99,"NULL":.01}),
                                   ((4, 5),{"VALUEM":.99,"NULL":.01}),
                                   ((5, 6),{"LABEL":.99,"NULL":.01}),

                                   ((6, 7),{"VALUEM":.99,"NULL":.01}),
                                   ((7, 8),{"LABEL":.99,"NULL":.01}),
                                   ((8, 9),{"VALUE":.99,"NULL":.01}),

                                   ((9, 10),{"VALUEM":.99,"NULL":.01}),
                                   ((10, 11),{"VALUE":.99,"NULL":.01}),
                                   ((11, 12),{"LABEL":.99,"NULL":.01}),

                                ],

                      "edges" : [ (((0, 1), (1, 2)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                  (((0, 1), (2, 3)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                  (((1, 2), (2, 3)), {'NULL': 0.9, 'INSTANCE': 0.1}),

                                (((3, 4), (4, 5)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                (((3, 4), (5, 6)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                (((4, 5), (5, 6)), {'NULL': 0.9, 'INSTANCE': 0.1}),

                                (((6, 7), (7, 8)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                (((6, 7), (8, 9)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                (((7, 8), (8, 9)), {'NULL': 0.2, 'INSTANCE': 0.8}),

                                (((9, 10), (10, 11)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                (((9, 10), (11, 12)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                (((10, 11), (11, 12)), {'NULL': 0.9, 'INSTANCE': 0.1})


                                ] }

    nodes,node_scores = zip(*sentence_datum["nodes"])
    edges,edge_scores = zip(*sentence_datum["edges"])

    ## pad
    for score_dict in node_scores:
        for label in node_labels:
            if label not in score_dict:
                score_dict[label] = 0.0

    for score_dict in edge_scores:
        for label in edge_labels:
            if label not in score_dict:
                score_dict[label] = 0.0

    model = gr.Model()

    node_variables = {}
    for i,node in enumerate(nodes):
        node_variables[node] = { label:instantiate_variable(model,label,"node",node,node_scores[i].get(label)) for label in node_scores[i] }

    edge_variables = {}
    for i,edge in enumerate(edges):
        edge_variables[edge] = { label:instantiate_variable(model,label,"edge",edge,edge_scores[i].get(label)) for label in edge_scores[i] }

    core_instance_attr_indicators = {}
    for node in node_variables:
        core_instance_attr_indicators[node] = add_disjunction_over_variable_set(model,node,node_variables,core_instance_labels)

    core_frame_attr_indicators = {}
    for node in node_variables:
        core_frame_attr_indicators[node] = add_disjunction_over_variable_set(model,node,node_variables,core_frame_labels)

    # resuable conjunction variables
    edge_endpoint_conjunctions = {}
    for attr, attr_ in endpoint_clamp_tuples:
        key,value = instantiate_clamped_endpoint_edge_constraints(model,attr,attr_,edge_variables,node_variables,core_frame_attr_indicators,core_instance_attr_indicators)
        edge_endpoint_conjunctions[key] = value


    target_clamped_edge_conjunctions = {}
    for attr, edge_type in clamped_edge_tuples:
        key,value = instantiate_partial_clamped_endpoint_edge_constraints(model,attr,edge_type,edge_variables,node_variables,core_frame_attr_indicators,core_instance_attr_indicators)
        target_clamped_edge_conjunctions[key] = value


    require_choose_one(model,node_variables,"node")
    require_choose_one(model,edge_variables,"edge")

    model.update()
    model.optimize()

    print( "BEFORE" )
    for i,edge in enumerate(edge_variables):
        print( edge )
        for label in edge_variables[edge]:
            print( label, edge_variables[edge][label].X )
        print( "______________" )


    print( "\n" )

    require_mixed_edge_constraints(model,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,edge_endpoint_conjunctions,target_clamped_edge_conjunctions)


    model.update()
    model.optimize()

    print( "AFTER" )
    for i,edge in enumerate(edge_variables):
        print( edge )
        for label in edge_variables[edge]:
            print( label, edge_variables[edge][label].X )
        print( "______________" )

    print( "\n" )
'''

def test_instance_triangles():

    print( "\n" )
    print( "testing instance triangle constraints")

    sentence_datum = { "nodes" :[  ((0, 1),{"VALUE":.99,"NULL":.01}),
                                   ((1, 2),{"LABEL":.99,"NULL":.01}),
                                   ((2, 3),{"VALUEM":.99,"NULL":.01}),

                                   ((3, 4),{"VALUE":.99,"NULL":.01}),
                                   ((4, 5),{"VALUEM":.99,"NULL":.01}),
                                   ((5, 6),{"LABEL":.99,"NULL":.01}),

                                   ((6, 7),{"VALUEM":.99,"NULL":.01}),
                                   ((7, 8),{"LABEL":.99,"NULL":.01}),
                                   ((8, 9),{"VALUE":.99,"NULL":.01}),

                                   ((9, 10),{"VALUEM":.99,"NULL":.01}),
                                   ((10, 11),{"VALUE":.99,"NULL":.01}),
                                   ((11, 12),{"LABEL":.99,"NULL":.01}),

                                ],

                      "edges" : [ (((0, 1), (1, 2)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                  (((0, 1), (2, 3)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                  (((1, 2), (2, 3)), {'NULL': 0.9, 'INSTANCE': 0.1}),

                                (((3, 4), (4, 5)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                (((3, 4), (5, 6)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                (((4, 5), (5, 6)), {'NULL': 0.9, 'INSTANCE': 0.1}),

                                (((6, 7), (7, 8)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                (((6, 7), (8, 9)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                (((7, 8), (8, 9)), {'NULL': 0.2, 'INSTANCE': 0.8}),

                                (((9, 10), (10, 11)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                (((9, 10), (11, 12)), {'NULL': 0.01, 'INSTANCE': 0.99}),
                                (((10, 11), (11, 12)), {'NULL': 0.9, 'INSTANCE': 0.1})


                                ] }

    nodes,node_scores = zip(*sentence_datum["nodes"])
    edges,edge_scores = zip(*sentence_datum["edges"])

    ## pad
    for score_dict in node_scores:
        for label in node_labels:
            if label not in score_dict:
                score_dict[label] = 0.0

    for score_dict in edge_scores:
        for label in edge_labels:
            if label not in score_dict:
                score_dict[label] = 0.0

    model = gr.Model()

    node_variables = {}
    for i,node in enumerate(nodes):
        node_variables[node] = { label:instantiate_variable(model,label,"node",node,node_scores[i].get(label)) for label in node_scores[i] }

    edge_variables = {}
    for i,edge in enumerate(edges):
        edge_variables[edge] = { label:instantiate_variable(model,label,"edge",edge,edge_scores[i].get(label)) for label in edge_scores[i] }

    core_instance_attr_indicators = {}
    for node in node_variables:
        core_instance_attr_indicators[node] = add_disjunction_over_variable_set(model,node,node_variables,core_instance_labels,"core_instance")

    core_frame_attr_indicators = {}
    for node in node_variables:
        core_frame_attr_indicators[node] = add_disjunction_over_variable_set(model,node,node_variables,core_frame_labels,"core_frame")

    # resuable conjunction variables
    edge_endpoint_conjunctions = {}
    for attr, attr_ in endpoint_clamp_tuples:
        key,value = instantiate_clamped_endpoint_edge_constraints(model,attr,attr_,edge_variables,node_variables,core_frame_attr_indicators,core_instance_attr_indicators)
        edge_endpoint_conjunctions[key] = value


    target_clamped_edge_conjunctions = {}
    for attr, edge_type in clamped_edge_tuples:
        key,value = instantiate_partial_clamped_endpoint_edge_constraints(model,attr,edge_type,edge_variables,node_variables,core_frame_attr_indicators,core_instance_attr_indicators)
        target_clamped_edge_conjunctions[key] = value


    require_choose_one(model,node_variables,"node")
    require_choose_one(model,edge_variables,"edge")

    model.update()
    model.optimize()

    print( "BEFORE" )
    for i,edge in enumerate(edge_variables):
        print( edge )
        for label in edge_variables[edge]:
            print( label, edge_variables[edge][label].X )
        print( "______________" )


    print( "\n" )

    require_instance_triangle_constraints(model,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators,edge_endpoint_conjunctions,target_clamped_edge_conjunctions)

    model.update()
    model.optimize()

    print( "AFTER" )
    for i,edge in enumerate(edge_variables):
        print( edge )
        for label in edge_variables[edge]:
            print( label, edge_variables[edge][label].X )
        print( "______________" )

    print( "\n" )
