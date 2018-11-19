"""
Exposes solve_ilp, which takes a sentence graph as input and output.
"""

import pdb
from collections import Counter

import gurobipy as gr
import numpy as np

from .schema import SentenceGraph
from .util import cargmax

node_labels=["VALUE", "VALUEM", "LABEL", "LABEL_MOD", "THEME", "THEME_MOD", "POSSESSOR", "UNIT", "UNIT_RESTRICTION", "TYPE"]
instance_labels=["VALUE","VALUEM","LABEL","LABEL_MOD"]
edge_labels=["FRAME", "INSTANCE"]
core_frame_labels=["THEME","POSSESSOR","TYPE"]
core_instance_labels=["VALUE","VALUEM","LABEL"]
mod_label_pairs=[("THEME","THEME_MOD"),("UNIT","UNIT_RESTRICTION"),("LABEL","LABEL_MOD"),("VALUE","VALUEM")]
#frame_labels=["MANNER", "SIGN"]

'''
# Accessor/helper methods
'''
def get_overlapping_node_variables(node_variables):
    '''
    Return a [ (node variable,[conflicting node variables]) ] list covering all node overlaps but not repeating
    '''
    nodes = list(node_variables.keys())

    conflict_sets = []
    for i in range(len(nodes)):
        node = nodes[i]
        for label in node_variables[node]:

            variable = node_variables[node][label][0]

            equivalent_conflicting_nodes = [ node_variables[node][l][0] for l in node_variables[node] if l != label ]

            additional_conflicting_nodes = []
            if i < len(nodes) - 1:
                additional_conflicting_nodes = [ node_variables[nodes[j]][l][0] for j in range(i+1,len(nodes)) for l in node_variables[nodes[j]] if check_node_overlap(node,nodes[j]) ]

            if len(equivalent_conflicting_nodes) + len(additional_conflicting_nodes) > 0:
                conflict_sets.append((variable,equivalent_conflicting_nodes+additional_conflicting_nodes))

    return conflict_sets

def get_positive_vars_with_labels(node_variables,labels):
    '''
    Get all positive variables associated with all nodes associated with the elements of labels, a list
    '''
    my_vars = [ node_variables[node][label][0] for label in labels for node in node_variables if label in node_variables[node] ]

    return my_vars

def positive_vars_for_node_with_labels(node_variables,node,labels):
    '''
    Get all positive variables associated with node associated with the elements of labels
    '''
    my_vars = [ node_variables[node][label][0] for label in labels if label in node_variables[node] ]

    return my_vars

def check_node_overlap(node_1,node_2):

    node_1 = set(range(node_1[0],node_1[1]))
    node_2 = set(range(node_2[0],node_2[1]))

    return len(node_1.intersection(node_2)) != 0

def get_edge_variable(node_1,node_2,edge_variables,edge_type,positive=True):
    var = None

    if (node_1,node_2) in edge_variables:
        var = edge_variables[(node_1,node_2)].get(edge_type)

    elif (node_2,node_1) in edge_variables:
        var = edge_variables[(node_2,node_1)].get(edge_type)

    if var != None:
        if positive:
            return var[0]
        else:
            return var[1]
    else:
        return var

def stringify_node(node):
    return "(" + str(node[0]) + "," + str(node[1]) + ")"

def stringify_edge(edge):
    return "((" + str(edge[0][0]) + "," + str(edge[0][1]) + "),(" + str(edge[1][0]) + "," + str(edge[1][1]) + "))"

def get_edges_from_node(node,edge_variables):
    return [ edge for edge in edge_variables if edge[0] == node or edge[1] == node ]

def get_other_node_in_edge(node,edge):
    assert edge[0] == node or edge[1] == node

    if edge[0] == node:
        return edge[1]
    else:
        return edge[0]

def instantiate_variable_indicator(model,elem,variable_dict,label):
    new_var = model.addVar(vtype=gr.GRB.BINARY)
    if elem in variable_dict:
        var = variable_dict[elem].get(label)
        if var != None:
            model.addConstr(new_var == var[0])
    else:
        model.addConstr(new_var == 0)


'''
# Variable creation methods
'''
def instantiate_variable_pair(model,label,var_type,var,score):
    '''
    Instantiate positive and negative variable instances
        -ilp_model := gurobi model
        -label := classification label (e.g. "VALUE" or "FRAME")
        -var_type := "node" or "edge"
        -index := unique integer identifier
        -score := 0 < s < 1
    '''
    if var_type == "node":
        node_name = stringify_node(var)
        positive_variable = model.addVar(obj=-np.log(score), vtype=gr.GRB.BINARY, name=label + "_" + var_type + "_positive_" + node_name )
        negative_variable = model.addVar(obj=-np.log(1-score), vtype=gr.GRB.BINARY, name=label + "_" + var_type + "_negative_" + node_name )
    else:
        edge_name = stringify_edge(var)
        positive_variable = model.addVar(obj=-np.log(score), vtype=gr.GRB.BINARY, name=label + "_" + var_type + "_positive_" + edge_name )
        negative_variable = model.addVar(obj=-np.log(1-score), vtype=gr.GRB.BINARY, name=label + "_" + var_type + "_negative_" + edge_name )

    model.update()

    return (positive_variable,negative_variable)

'''
# Constraint creation
'''
def require_choose_one(model,variables,elem_to_index,var_type,test=False):
    '''
    # For any given candidate (e.g. "span (0,1)") and any given candidate label:
    # either the positive variable or the negative variable must be active in a solution
    '''

    choose_one_constraints = [ model.addConstr(variables[elem][label][0]+variables[elem][label][1]==1, name=label+"_"+str(elem_to_index[elem])) for elem in variables for label in variables[elem] ]

    return choose_one_constraints

def require_node_span_non_overlap(model,node_variables):
    '''
    # If the positive variable of some span is active,
    # then none of the positive variabels of spans which overlap with it is active
    '''

    positive_overlaps = get_overlapping_node_variables(node_variables)
    non_overlap_constraints = [ model.addGenConstrIndicator(var, True, gr.quicksum(conflicting_vars) == 0, name=var.getAttr("VarName") + "_nonoverlap") for var,conflicting_vars in positive_overlaps ]

    return non_overlap_constraints

def require_frame_instance_edge_exclusivity(model,edge_variables):
    '''
    # Given some candidate arc, it can either be an INSTANCE arc or a FRAME arc but not both
    '''

    frame_edge_exclusivities = []
    for edge in edge_variables:
        if "FRAME" in edge_variables[edge] and "INSTANCE" in edge_variables[edge]:
            frame_var = edge_variables[edge]["FRAME"][0]
            inst_var = edge_variables[edge]["INSTANCE"][0]
            frame_edge_exclusivities.append(model.addConstr(frame_var+inst_var<=1))

    return frame_edge_exclusivities

def require_label_label_frame_edge(model,edge_variables,node_variables):
    '''
    # a solution must contain at least one FRAME edge linking two LABEL nodes
    '''
    frame_edges_between_labels = []
    for edge in edge_variables:
        if "FRAME" in edge_variables[edge]:
            positive_edge_var = edge_variables[edge]["FRAME"][0]
            source,target = edge

            if "LABEL" in node_variables[source] and "LABEL" in node_variables[target]:

                positive_source_label_var = node_variables[source]["LABEL"][0]
                positive_target_label_var = node_variables[target]["LABEL"][0]

                # z = 1 in m iff. source and target of edge are positively classified as LABELs
                z = model.addVar(vtype=gr.GRB.BINARY)
                model.addGenConstrAnd(z,[positive_edge_var,positive_source_label_var,positive_target_label_var])

                frame_edges_between_labels.append(z)

    return model.addConstr(gr.quicksum(frame_edges_between_labels)>=1,name="label_label_frame_edge_MUST_exist")

def require_distinct_instance_arcs(model,edge_variables,node_variables,node_scores,node_to_index):
    '''
    # instance arcs cannot apply between equivalent labels
    '''
    these_constraints = []
    for edge in edge_variables:
        if "INSTANCE" in edge_variables[edge]:
            positive_edge_var = edge_variables[edge]["INSTANCE"][0]
            source,target = edge
            source_index,target_index = node_to_index[source],node_to_index[target]

            for label in set(node_scores[source_index].keys()).intersection(set(node_scores[target_index].keys())):
                # add constraint
                if label in instance_labels:
                    source_pos_var = node_variables[source][label][0]
                    target_pos_var = node_variables[target][label][0]

                    these_constraints.append(model.addGenConstrIndicator(positive_edge_var, True, source_pos_var+target_pos_var<=1))

    return these_constraints

def require_instance_labels_for_instance_arcs(model,edge_variables,node_variables):
    '''
    # given an instance arc, both of its nodes must be instance attributes
    # label_mods only attach to label
    '''
    these_constraints = []
    for edge in edge_variables:
        if "INSTANCE" in edge_variables[edge]:

            positive_edge_var = edge_variables[edge]["INSTANCE"][0]
            source,target = edge

            pos_source_instance_vars = positive_vars_for_node_with_labels(node_variables,source,["LABEL","VALUE","VALUEM"])
            pos_target_instance_vars = positive_vars_for_node_with_labels(node_variables,target,["LABEL","VALUE","VALUEM"])

            source_label = node_variables[source].get("LABEL")
            if source_label != None:
                source_label = source_label[0]
            target_label = node_variables[target].get("LABEL")
            if target_label != None:
                target_label = target_label[0]
            source_label_mod = node_variables[source].get("LABEL_MOD")
            if source_label_mod != None:
                source_label_mod = source_label_mod[0]
            target_label_mod = node_variables[target].get("LABEL_MOD")
            if target_label_mod != None:
                target_label_mod = target_label_mod[0]

            source_instance = model.addVar(vtype=gr.GRB.BINARY)
            model.addConstr(source_instance == gr.quicksum(pos_source_instance_vars))
            target_instance = model.addVar(vtype=gr.GRB.BINARY)
            model.addConstr(target_instance == gr.quicksum(pos_target_instance_vars))

            s_i_t_i = model.addVar(vtype=gr.GRB.BINARY)
            model.addGenConstrAnd(s_i_t_i,[source_instance,target_instance])

            s_l_t_lm = model.addVar(vtype=gr.GRB.BINARY)
            if source_label != None and target_label_mod != None:
                model.addGenConstrAnd(s_l_t_lm,[source_label,target_label_mod])
            else:
                model.addConstr(s_l_t_lm == 0)

            s_lm_t_l = model.addVar(vtype=gr.GRB.BINARY)
            if source_label_mod != None and target_label != None:
                model.addGenConstrAnd(s_lm_t_l,[source_label_mod,target_label])
            else:
                model.addConstr(s_lm_t_l == 0)

            these_constraints.append(model.addGenConstrOr(positive_edge_var,[s_i_t_i, s_l_t_lm, s_lm_t_l]))

    return these_constraints

def require_frame_labels_for_frame_arcs(model,edge_variables,node_variables):
    '''
    # Given a frame arc, either:
    # 1) at least one node is a frame label
    # 2) nodes are not part of the same index
    # equivalent to: both nodes must be active, if there is a corresponding instance arc candidate, the frame arc does node_to_index
    '''
    these_constraints = []
    for edge in edge_variables:
        if "FRAME" in edge_variables[edge]:

            positive_edge_var = edge_variables[edge]["FRAME"][0]
            source,target = edge

            source_core_frame_vars = positive_vars_for_node_with_labels(node_variables,source,core_frame_labels)
            target_core_frame_vars = positive_vars_for_node_with_labels(node_variables,target,core_frame_labels)

            source_noncore_frame_vars = positive_vars_for_node_with_labels(node_variables,source,["THEME_MOD","UNIT_RESTRICTION"])
            target_noncore_frame_vars = positive_vars_for_node_with_labels(node_variables,target,["THEME_MOD","UNIT_RESTRICTION"])

            source_core_instance_vars = positive_vars_for_node_with_labels(node_variables,source,["LABEL","VALUE"])
            target_core_instance_vars = positive_vars_for_node_with_labels(node_variables,target,["LABEL","VALUE"])

            s_label_vars = node_variables[source].get("LABEL")
            t_label_vars = node_variables[target].get("LABEL")

            s_value_vars = node_variables[source].get("VALUE")
            t_value_vars = node_variables[target].get("VALUE")

            source_core_frame_true = model.addVar(vtype=gr.GRB.BINARY)
            model.addConstr(source_core_frame_true == gr.quicksum(source_core_frame_vars))

            target_core_frame_true = model.addVar(vtype=gr.GRB.BINARY)
            model.addConstr(target_core_frame_true == gr.quicksum(target_core_frame_vars))

            source_noncore_frame_true = model.addVar(vtype=gr.GRB.BINARY)
            model.addConstr(source_noncore_frame_true == gr.quicksum(source_noncore_frame_vars))

            target_noncore_frame_true = model.addVar(vtype=gr.GRB.BINARY)
            model.addConstr(target_noncore_frame_true == gr.quicksum(target_noncore_frame_vars))

            source_core_instance_true = model.addVar(vtype=gr.GRB.BINARY)
            model.addConstr(source_core_instance_true == gr.quicksum(source_core_instance_vars))

            target_core_instance_true = model.addVar(vtype=gr.GRB.BINARY)
            model.addConstr(target_core_instance_true == gr.quicksum(target_core_instance_vars))

            s_cf_t_cf = model.addVar(vtype=gr.GRB.BINARY)
            model.addGenConstrAnd(s_cf_t_cf,[source_core_frame_true,target_core_frame_true])

            s_cf_t_nf = model.addVar(vtype=gr.GRB.BINARY)
            model.addGenConstrAnd(s_cf_t_nf,[source_core_frame_true,target_noncore_frame_true])

            s_nf_t_cf = model.addVar(vtype=gr.GRB.BINARY)
            model.addGenConstrAnd(s_nf_t_cf,[source_noncore_frame_true,target_core_frame_true])

            s_cf_t_i = model.addVar(vtype=gr.GRB.BINARY)
            model.addGenConstrAnd(s_cf_t_i,[source_core_frame_true,target_core_instance_true])

            s_i_t_cf = model.addVar(vtype=gr.GRB.BINARY)
            model.addGenConstrAnd(s_i_t_cf,[source_core_instance_true,target_core_frame_true])

            s_label = model.addVar(vtype=gr.GRB.BINARY)
            t_label = model.addVar(vtype=gr.GRB.BINARY)

            if s_label_vars != None:
                model.addConstr(s_label == s_label_vars[0])
            else:
                model.addConstr(s_label == 0)
            if t_label_vars != None:
                model.addConstr(t_label == t_label_vars[0])
            else:
                model.addConstr(t_label == 0)

            s_value = model.addVar(vtype=gr.GRB.BINARY)
            t_value = model.addVar(vtype=gr.GRB.BINARY)

            if s_value_vars != None:
                model.addConstr(s_value == s_value_vars[0])
            else:
                model.addConstr(s_value == 0)
            if t_value_vars != None:
                model.addConstr(t_value == t_value_vars[0])
            else:
                model.addConstr(t_value == 0)

            s_l_t_l = model.addVar(vtype=gr.GRB.BINARY)
            model.addGenConstrAnd(s_l_t_l,[s_label,t_label])

            s_v_t_v = model.addVar(vtype=gr.GRB.BINARY)
            model.addGenConstrAnd(s_v_t_v,[s_value,t_value])

            # if frame edge obtains, then one of the above binary variables must be true
            model.addGenConstrOr(positive_edge_var,[s_cf_t_cf, s_cf_t_nf, s_nf_t_cf, s_cf_t_i, s_i_t_cf, s_l_t_l, s_v_t_v])

    model.update()

    return these_constraints

def require_unique_label_for_value(model,edge_variables,node_variables):
    '''
    # for every value, there must exist exactly one label to which it is linked
    '''
    for node in node_variables:
        if "VALUE" in node_variables[node]:

            these_instance_edges = {}
            for edge in edge_variables:
                if "INSTANCE" in edge_variables[edge] and ( edge[0] == node or edge[1] == node ):
                    these_instance_edges[edge] = edge_variables[edge]["INSTANCE"][0]

            positive_node_var = node_variables[node]["VALUE"][0]

            if len(these_instance_edges) != 0:

                label_value_edges = []
                for edge in these_instance_edges:

                    target = None
                    if edge[0] == node:
                        target = edge[1]
                    else:
                        target = edge[0]
                    if "LABEL" not in node_variables[target]:
                        break

                    positive_value_var = node_variables[target]["LABEL"][0]

                    z = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(z,[these_instance_edges[edge],positive_node_var,positive_value_var])

                    label_value_edges.append(z)

                model.addConstr(gr.quicksum(label_value_edges)==1)

            else:
                model.addConstr(positive_node_var==0)

    return None

def require_alo_y_for_attr_x(model,x,y,edge_type,edge_variables,node_variables):

    '''
    # require_alo_y_for_attr_x(m,"LABEL","VALUE",edge_variables,node_variables)
    # for every X, there must exist at least one Y to which it is linked
    require_alo_y_for_attr_x(m,"LABEL","VALUE","INSTANCE",edge_variables,node_variables)
    '''
    for node in node_variables:
        if x in node_variables[node]:

            positive_x_variable = node_variables[node][x][0]
            edges_from_node = [ edge for edge in get_edges_from_node(node,edge_variables) if edge_type in edge_variables[edge] ]

            indicators = []
            for edge in edges_from_node:
                edge_variable = edge_variables[edge][edge_type][0]
                target = get_other_node_in_edge(node,edge)

                if y in node_variables[target]:
                    positive_y_var = node_variables[target][y][0]
                    z = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(z,[positive_y_var,edge_variable])
                    indicators.append(z)

            # if postive label variable, there must be at least one instance arc linking it to a value node
            model.addGenConstrIndicator(positive_x_variable, True, gr.quicksum(indicators) >= 1)
            model.update()


def require_attribute_mod_anchor(model,anchor_label,mod_label,node_variables,edge_variables):
    '''
    Require, e.g. that a THEME_MOD must be attached to a THEME , and a VALUEM must be attached to a VALUE
    '''
    edge_type = None
    if anchor_label in instance_labels:
        edge_type = "INSTANCE"
    else:
        edge_type = "FRAME"

    for node in node_variables:
        if mod_label in node_variables[node]:

            positive_mod_variable = node_variables[node][mod_label][0]
            edges_from_node = [ edge for edge in get_edges_from_node(node,edge_variables) if edge_type in edge_variables[edge] ]
            indicators = []
            for edge in edges_from_node:
                edge_variable = edge_variables[edge][edge_type][0]
                target = get_other_node_in_edge(node,edge)
                if anchor_label in node_variables[target]:
                    positive_anchor_var = node_variables[target][anchor_label][0]
                    z = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(z, [positive_anchor_var,edge_variable])
                    indicators.append(z)

            model.addGenConstrIndicator(positive_mod_variable, True, gr.quicksum(indicators) == 1)
            model.update()

def require_frame_triangle_constraints(model,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators):

    for node in node_variables:
        source_is_core_frame = core_frame_attr_indicators.get(node)
        source_is_core_instance = core_instance_attr_indicators.get(node)

        frame_edges_from_node = [ edge for edge in edge_variables if node in edge and "FRAME" in edge_variables[edge] ]

        # make sure open triangles are even possible
        if len(frame_edges_from_node) > 1:

            # begin upper triangular iteration
            for i in range(0,len(frame_edges_from_node)-1):

                target_1 = get_other_node_in_edge(node,frame_edges_from_node[i])

                target_1_is_core_frame = core_frame_attr_indicators.get(target_1)
                target_1_is_core_instance = core_instance_attr_indicators.get(target_1)

                frame_edge_s_t1 = edge_variables[frame_edges_from_node[i]]["FRAME"][0]

                for j in range(i+1,len(frame_edges_from_node)):

                    target_2 = get_other_node_in_edge(node,frame_edges_from_node[j])

                    target_2_is_core_frame = core_frame_attr_indicators.get(target_2)
                    target_2_is_core_instance = core_instance_attr_indicators.get(target_2)

                    frame_edge_s_t2 = edge_variables[frame_edges_from_node[j]]["FRAME"][0]

                    ## get triangle-completing frame edge variable, if none exists, instantiate dummy variable and force it equal to zero
                    frame_edge_t1_t2 = get_edge_variable(target_1,target_2,edge_variables,"FRAME")
                    if frame_edge_t1_t2 == None:
                        frame_edge_t1_t2 = model.addVar(vtype=gr.GRB.BINARY)
                        model.addConstr( frame_edge_t1_t2 == 0 )

                    # condition_1
                    condition_1 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_1,[source_is_core_frame,target_1_is_core_frame,target_2_is_core_frame,frame_edge_s_t1,frame_edge_s_t2])
                    model.addGenConstrIndicator(condition_1, True, frame_edge_t1_t2 == 1)

                    # condition_2
                    condition_2 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_2,[source_is_core_frame,target_1_is_core_frame,target_2_is_core_instance,frame_edge_s_t1,frame_edge_s_t2])
                    model.addGenConstrIndicator(condition_2, True, frame_edge_t1_t2 == 1)

                    # condition_3
                    condition_3 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_3,[source_is_core_frame,target_1_is_core_instance,target_2_is_core_frame,frame_edge_s_t1,frame_edge_s_t2])
                    model.addGenConstrIndicator(condition_3, True, frame_edge_t1_t2 == 1)

                    # condition_4
                    condition_4 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_4,[source_is_core_instance,target_1_is_core_frame,target_2_is_core_frame,frame_edge_s_t1,frame_edge_s_t2])
                    model.addGenConstrIndicator(condition_4, True, frame_edge_t1_t2 == 1)

                    # condition_5
                    condition_5 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_5,[source_is_core_instance,target_1_is_core_instance,target_2_is_core_frame,frame_edge_s_t1,frame_edge_s_t2])
                    model.addGenConstrIndicator(condition_5, True, frame_edge_t1_t2 == 1)

                    # condition_6
                    condition_6 = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition_6,[source_is_core_instance,target_1_is_core_frame,target_2_is_core_instance,frame_edge_s_t1,frame_edge_s_t2])
                    model.addGenConstrIndicator(condition_6, True, frame_edge_t1_t2 == 1)


def require_mixed_edge_triangle_constraint(model,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators):

    for node in node_variables:

        source_is_core_instance = core_instance_attr_indicators.get(node)

        frame_edges_from_node = [ edge for edge in edge_variables if node in edge and "FRAME" in edge_variables[edge] ]
        instance_edges_from_node = [ edge for edge in edge_variables if node in edge and "INSTANCE" in edge_variables[edge] ]

        if len(frame_edges_from_node) > 1:

            for i in range(0,len(frame_edges_from_node)):

                target_1 = get_other_node_in_edge(node,frame_edges_from_node[i])

                target_1_is_core_frame = core_frame_attr_indicators.get(target_1)
                frame_edge_s_t1 = edge_variables[frame_edges_from_node[i]]["FRAME"][0]

                for j in range(0,len(instance_edges_from_node)):

                    target_2 = get_other_node_in_edge(node,instance_edges_from_node[j])

                    if target_2 == target_1:
                        continue

                    target_2_is_core_instance = core_instance_attr_indicators.get(target_2)
                    instance_edge_s_t2 = edge_variables[instance_edges_from_node[j]]["INSTANCE"][0]

                    ## get triangle-completing frame edge variable, if none exists, instantiate dummy variable and force it equal to zero
                    frame_edge_t1_t2 = get_edge_variable(target_1,target_2,edge_variables,"FRAME")
                    if frame_edge_t1_t2 == None:
                        frame_edge_t1_t2 = model.addVar(vtype=gr.GRB.BINARY)
                        model.addConstr( frame_edge_t1_t2 == 0 )

                    # condition
                    condition = model.addVar(vtype=gr.GRB.BINARY)
                    model.addGenConstrAnd(condition,[source_is_core_instance,target_1_is_core_frame,target_2_is_core_instance,frame_edge_s_t1,instance_edge_s_t2])
                    model.addGenConstrIndicator(condition, True, frame_edge_t1_t2 == 1)


def require_instance_triangle_constraints(model,nodes,node_variables,edge_variables):
    '''
    # value - label / value - value-m --> value-m - label
    # value-m - label / value-m - value --> value - label
    '''
    for node in nodes:

        node_value = model.addVar(vtype=gr.GRB.BINARY)
        node_value_vars = node_variables[node].get("VALUE")
        if node_value_vars != None:
            model.addConstr(node_value == node_value_vars[0])

        node_valuem = model.addVar(vtype=gr.GRB.BINARY)
        node_valuem_vars = node_variables[node].get("VALUEM")
        if node_valuem_vars != None:
            model.addConstr(node_valuem == node_valuem_vars[0])

        if node_value_vars == None and node_valuem_vars == None:
            break

        instance_edges_involving_node = [ edge for edge in edge_variables if node in edge and "INSTANCE" in edge_variables[edge] ]

        for i in range(len(instance_edges_involving_node)-1):

            edge_1 = instance_edges_involving_node[i]
            target_1 = get_other_node_in_edge(node,edge_1)
            edge_1 = edge_variables[edge_1]["INSTANCE"][0]

            target_1_label = model.addVar(vtype=gr.GRB.BINARY)
            target_1_label_vars = node_variables[target_1].get("LABEL")
            if target_1_label_vars != None:
                model.addConstr(target_1_label == target_1_label_vars[0])
            else:
                model.addConstr(target_1_label == 0)

            target_1_valuem = model.addVar(vtype=gr.GRB.BINARY)
            target_1_valuem_vars = node_variables[target_1].get("VALUEM")
            if target_1_valuem_vars != None:
                model.addConstr(target_1_valuem == target_1_valuem_vars[0])
            else:
                model.addConstr(target_1_valuem == 0)

            target_1_value = model.addVar(vtype=gr.GRB.BINARY)
            target_1_value_vars = node_variables[target_1].get("VALUE")
            if target_1_value_vars != None:
                model.addConstr(target_1_value == target_1_value_vars[0])
            else:
                model.addConstr(target_1_value == 0)


            for j in range(i+1,len(instance_edges_involving_node)):

                edge_2 = instance_edges_involving_node[j]
                target_2 = get_other_node_in_edge(node,edge_2)
                edge_2 = edge_variables[edge_2]["INSTANCE"][0]

                edge_3 = get_edge_variable(target_1,target_2,edge_variables,"INSTANCE")

                target_2_label = model.addVar(vtype=gr.GRB.BINARY)
                target_2_label_vars = node_variables[target_2].get("LABEL")
                if target_2_label_vars != None:
                    model.addConstr(target_2_label == target_2_label_vars[0])
                else:
                    model.addConstr(target_2_label == 0)

                target_2_valuem = model.addVar(vtype=gr.GRB.BINARY)
                target_2_valuem_vars = node_variables[target_2].get("VALUEM")
                if target_2_valuem_vars != None:
                    model.addConstr(target_2_valuem == target_2_valuem_vars[0])
                else:
                    model.addConstr(target_2_valuem == 0)

                target_2_value = model.addVar(vtype=gr.GRB.BINARY)
                target_2_value_vars = node_variables[target_2].get("VALUE")
                if target_1_value_vars != None:
                    model.addConstr(target_2_value == target_2_value_vars[0])
                else:
                    model.addConstr(target_2_value == 0)

                #value - label - value-m
                cond_1 = model.addVar(vtype=gr.GRB.BINARY)
                model.addGenConstrAnd(cond_1,[edge_1,edge_2,node_value,target_1_label,target_2_valuem])
                if edge_3 != None:
                    model.addGenConstrIndicator(cond_1, True, edge_3 == 1)
                else:
                    model.addConstr(cond_1==0)

                #value - value-m - label
                cond_2 = model.addVar(vtype=gr.GRB.BINARY)
                model.addGenConstrAnd(cond_2,[edge_1,edge_2,node_value,target_1_valuem,target_2_label])
                if edge_3 != None:
                    model.addGenConstrIndicator(cond_2, True, edge_3 == 1)
                else:
                    model.addConstr(cond_2==0)

                #value-m - label - value
                cond_3 = model.addVar(vtype=gr.GRB.BINARY)
                model.addGenConstrAnd(cond_3,[edge_1,edge_2,node_valuem,target_1_label,target_2_value])
                if edge_3 != None:
                    model.addGenConstrIndicator(cond_3, True, edge_3 == 1)
                else:
                    model.addConstr(cond_3==0)

                #value-m - value - label
                cond_4 = model.addVar(vtype=gr.GRB.BINARY)
                model.addGenConstrAnd(cond_4,[edge_1,edge_2,node_valuem,target_1_value,target_2_label])
                if edge_3 != None:
                    model.addGenConstrIndicator(cond_4, True, edge_3 == 1)
                else:
                    model.addConstr(cond_4==0)


'''
# ILP for numerical contrast parsing
'''
def construct_and_solve_sentence_ilp(sentence_datum,test=False):
    '''
    # Initialize gurobi model

    '''
    m = gr.Model()

    '''
    # Unpack candidates
    '''
    nodes,node_scores = zip(*sentence_datum["nodes"])
    edges,edge_scores = zip(*sentence_datum["edges"])

    node_to_index = { node:i for i,node in enumerate(nodes) }
    edge_to_index = { edge:i for i,edge in enumerate(edges) }

    '''
    # Instantiate variabels
    '''
    node_variables = {}
    for i,node in enumerate(nodes):
        node_variables[node] = { label:instantiate_variable_pair(m,label,"node",node,node_scores[i].get(label)) for label in node_scores[i] if node_scores[i].get(label) not in [0,None] }

    edge_variables = {}
    for i,edge in enumerate(edges):
        edge_variables[edge] = { label:instantiate_variable_pair(m,label,"edge",edge,edge_scores[i].get(label)) for label in edge_scores[i] if edge_scores[i].get(label) not in [0,None] }

    core_frame_attr_indicators = {}
    for node in node_variables:
        v = m.addVar(vtype=gr.GRB.BINARY)
        m.addConstr( v == gr.quicksum(positive_vars_for_node_with_labels(node_variables,node,core_frame_labels)) )
        core_frame_attr_indicators[node] = v

    core_instance_attr_indicators = {}
    for node in node_variables:
        v = m.addVar(vtype=gr.GRB.BINARY)
        m.addConstr( v == gr.quicksum(positive_vars_for_node_with_labels(node_variables,node,core_instance_labels)) )
        core_instance_attr_indicators[node] = v

    '''
    # Apply constraints to model
    '''
    node_choose_one_constraints = require_choose_one(m,node_variables,node_to_index,"node")
    edge_choose_one_constraints = require_choose_one(m,edge_variables,edge_to_index,"edge")

    node_non_overlap_constraints = require_node_span_non_overlap(m,node_variables)
    edge_non_overlap_constraints = require_frame_instance_edge_exclusivity(m,edge_variables)

    # Arc endpoint constraints
    require_distinct_instance_arcs(m,edge_variables,node_variables,node_scores,node_to_index)
    require_instance_labels_for_instance_arcs(m,edge_variables,node_variables)
    require_frame_labels_for_frame_arcs(m,edge_variables,node_variables)

    # Minimum existence constraints
    label_label_frame_edge_constraint = require_label_label_frame_edge(m,edge_variables,node_variables)

    require_alo_y_for_attr_x(m,"LABEL","VALUE","INSTANCE",edge_variables,node_variables)
    require_alo_y_for_attr_x(m,"LABEL","LABEL","FRAME",edge_variables,node_variables)
    for attr in core_frame_labels:
        require_alo_y_for_attr_x(m,attr,"LABEL","FRAME",edge_variables,node_variables)

    # Maximum existence constraints
    require_unique_label_for_value(m,edge_variables,node_variables)
    for (anchor_label,mod_label) in mod_label_pairs:
        require_attribute_mod_anchor(m,anchor_label,mod_label,node_variables,edge_variables)

    # triangle constraints
    require_frame_triangle_constraints(m,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators)
    require_mixed_edge_triangle_constraint(m,node_variables,edge_variables,core_frame_attr_indicators,core_instance_attr_indicators)
    require_instance_triangle_constraints(m,nodes,node_variables,edge_variables)

    m.update()
    m.optimize()

    return convert_solution(node_variables, edge_variables)

'''
TEST METHODS
'''
def test_frame_triangle_constraints():

    '''
    ############# CONDITION 1 TEST ###########
    # n1_frame,n2_frame,n3_frame
    # e_12_frame,e_13_frame
    sentence_datum = { "nodes":[    ((4,5),{"THEME":.9}),
                                    ((5,6),{"POSSESSOR":.9}),
                                    ((6,7),{"TYPE":.9}) ] ,

                        "edges":[   (((4,5),(5,6)),{ "FRAME":.9 }),
                                    (((4,5),(6,7)),{ "FRAME":.9 }),
                                    (((5,6),(6,7)),{ "FRAME":.9 }) ] }
    '''
    '''
    ############# CONDITION 2/3 TEST ###########
    # n1_frame,n2_frame,n3_instance
    # e_12_frame,e_13_frame
    sentence_datum = { "nodes":[    ((4,5),{"THEME":.9}),
                                    ((5,6),{"POSSESSOR":.9}),
                                    ((6,7),{"LABEL":.9}) ] ,

                        "edges":[   (((4,5),(5,6)),{ "FRAME":.99 }),
                                    (((4,5),(6,7)),{ "FRAME":.99 }),
                                    (((5,6),(6,7)),{ "FRAME":.1 }) ] }
    '''
    '''
    ############# CONDITION 4 TEST ###########
    # n1_frame,n2_frame,n3_instance
    # e_12_frame,e_13_frame
    sentence_datum = { "nodes":[    ((4,5),{"LABEL":.9}),
                                    ((5,6),{"POSSESSOR":.9}),
                                    ((6,7),{"THEME":.9}) ] ,

                        "edges":[   (((4,5),(5,6)),{ "FRAME":.99 }),
                                    (((4,5),(6,7)),{ "FRAME":.99 }),
                                    (((5,6),(6,7)),{ "FRAME":.1 }) ] }
    '''
    '''

    ############# CONDITION 5/6 TEST ###########
    # n1_frame,n2_frame,n3_instance
    # e_12_frame,e_13_frame
    sentence_datum = { "nodes":[    ((4,5),{"LABEL":.9}),
                                    ((5,6),{"LABEL":.9}),
                                    ((6,7),{"THEME":.9}) ] ,

                        "edges":[   (((4,5),(5,6)),{ "FRAME":.99 }),
                                    (((4,5),(6,7)),{ "FRAME":.99 }),
                                    (((5,6),(6,7)),{ "FRAME":.1 }) ] }
    '''

    ############# CONDITION 7/8 TEST ###########
    # n1_frame,n2_frame,n3_instance
    # e_12_frame,e_13_frame
    sentence_datum = { "nodes":[    ((4,5),{"LABEL":.99}),
                                    ((5,6),{"VALUE":.99}),
                                    ((6,7),{"THEME":.99}) ] ,

                        "edges":[   (((4,5),(5,6)),{ "INSTANCE":.99 }),
                                    (((4,5),(6,7)),{ "FRAME":.99 }),
                                    (((5,6),(6,7)),{ "FRAME":.1 }) ] }


    model = gr.Model()

    nodes,node_scores = zip(*sentence_datum["nodes"])
    edges,edge_scores = zip(*sentence_datum["edges"])

    node_to_index = { node:i for i,node in enumerate(nodes) }
    edge_to_index = { edge:i for i,edge in enumerate(edges) }

    '''
    # Instantiate variabels
    '''
    node_variables = {}
    for i,node in enumerate(nodes):
        node_variables[node] = { label:instantiate_variable_pair(model,label,"node",node,node_scores[i].get(label)) for label in node_scores[i] if node_scores[i].get(label) not in [0,None] }

    edge_variables = {}
    for i,edge in enumerate(edges):
        edge_variables[edge] = { label:instantiate_variable_pair(model,label,"edge",edge,edge_scores[i].get(label)) for label in edge_scores[i] if edge_scores[i].get(label) not in [0,None] }

    node_choose_one_constraints = require_choose_one(model,node_variables,node_to_index,"node")
    edge_choose_one_constraints = require_choose_one(model,edge_variables,edge_to_index,"edge")

    require_frame_labels_for_frame_arcs(model,edge_variables,node_variables)

    model.update()
    model.optimize()

    #require_frame_triangles(model,nodes,node_variables,edge_variables)
    print( "TESTING frame-frame-frame triangle condition")
    print( "BEFORE" )
    for node in node_variables:
        print(node)
        for label in node_variables[node]:
            print(label, node_variables[node][label][0].X)
    for edge in edge_variables:
        print(edge)
        for label in edge_variables[edge]:
            print(label, edge_variables[edge][label][0].X)

    #require_frame_triangle_constraints(model,nodes,node_variables,edge_variables)

    model.optimize()

    print( "AFTER" )
    for node in node_variables:
        print(node)
        for label in node_variables[node]:
            print(label, node_variables[node][label][0].X)
    for edge in edge_variables:
        print(edge)
        for label in edge_variables[edge]:
            print(label, edge_variables[edge][label][0].X)

    #for elem in node_dummies:
    #    print( elem.getAttr("VarName") , elem.X )

def test_frame_arc_endpoint_constraints():

    ############# CONDITION 7/8 TEST ###########
    # n1_frame,n2_frame,n3_instance
    # e_12_frame,e_13_frame
    sentence_datum = { "nodes":[    ((1,2),{"LABEL":.99}),
                                    ((2,3),{"LABEL":.99}),
                                    ((3,4),{"THEME":.99}),
                                    ((4,5),{"THEME_MOD":.99}),
                                    ((5,6),{"POSSESSOR":.99}),
                                    ((6,7),{"VALUE":.99}),
                                    ((7,8),{"VALUE":.99}) ] ,

                        "edges":[   (((1,2),(2,3)),{ "FRAME":.99 }),
                                    (((1,2),(6,7)),{ "FRAME":.99 }),
                                    (((1,2),(3,4)),{ "FRAME":.99 }),
                                    (((2,3),(4,5)),{ "FRAME":.99 })
                                    ]}


    model = gr.Model()

    nodes,node_scores = zip(*sentence_datum["nodes"])
    edges,edge_scores = zip(*sentence_datum["edges"])

    node_to_index = { node:i for i,node in enumerate(nodes) }
    edge_to_index = { edge:i for i,edge in enumerate(edges) }

    '''
    # Instantiate variabels
    '''
    node_variables = {}
    for i,node in enumerate(nodes):
        node_variables[node] = { label:instantiate_variable_pair(model,label,"node",node,node_scores[i].get(label)) for label in node_scores[i] if node_scores[i].get(label) not in [0,None] }

    edge_variables = {}
    for i,edge in enumerate(edges):
        edge_variables[edge] = { label:instantiate_variable_pair(model,label,"edge",edge,edge_scores[i].get(label)) for label in edge_scores[i] if edge_scores[i].get(label) not in [0,None] }

    node_choose_one_constraints = require_choose_one(model,node_variables,node_to_index,"node")
    edge_choose_one_constraints = require_choose_one(model,edge_variables,edge_to_index,"edge")

    model.update()
    model.optimize()

    #require_frame_triangl_constraints(model,nodes,node_variables,edge_variables)
    print( "TESTING frame-frame endpoints condition")
    print( "BEFORE" )
    for node in node_variables:
        print(node)
        for label in node_variables[node]:
            print(label, node_variables[node][label][0].X)
    for edge in edge_variables:
        print(edge)
        for label in edge_variables[edge]:
            print(label, edge_variables[edge][label][0].X)

    require_frame_labels_for_frame_arcs(model,edge_variables,node_variables)

    model.update()
    model.optimize()

    print( "AFTER" )
    for node in node_variables:
        print(node)
        for label in node_variables[node]:
            print(label, node_variables[node][label][0].X)
    for edge in edge_variables:
        print(edge)
        for label in edge_variables[edge]:
            print(label, edge_variables[edge][label][0].X)

def test_construct_and_solve_sentence_ilp():
    '''
    # run tests on ilp constraints
    # todo: write a robust/easy-to-run test suite
    '''

    sentence_datum = { "nodes":[    ((0,1),{"VALUE":.9}) ,
                                    ((1,2),{"VALUE":.9}) ,
                                    ((2,3),{"LABEL":.9}) ,
                                    ((3,4),{"LABEL":.9}) ,
                                    ((4,5),{"THEME":.9}) ,
                                    ((5,6),{"THEME_MOD":.99}) ,
                                    ((6,7),{"LABEL_MOD":.99}) ],

                       "edges":[    (((2,3),(3,4)),{ "FRAME":.9 }),
                                    (((0,1),(2,3)),{ "INSTANCE":.9 }),
                                    (((1,2),(3,4)),{ "INSTANCE":.9 }),
                                    (((4,5),(5,6)),{ "FRAME":.4 }),
                                    (((2,3),(6,7)),{ "INSTANCE":.4 })  ] }


    construct_and_solve_sentence_ilp(sentence_datum)

def convert_solution(node_variables, edge_variables):
    """
    Converts a solution with gurobi variables into a more familiar graph.
    """
    ret_nodes, ret_edges = {}, {}
    for span, attrs in node_variables.items():
        # Convert attrs into a Counter.
        ret_nodes[span] = Counter({attr: pos_var.x for attr, (pos_var, _) in attrs.items()})
    for (span, span_), attrs in edge_variables.items():
        # Convert attrs into a Counter.
        ret_edges[(span, span_)] = Counter({attr: pos_var.x for attr, (pos_var, _) in attrs.items()})
    return ret_nodes, ret_edges

def to_datum(graph):
    ret = {
        "nodes": [(span, {key.upper(): value for key, value in attr.items() if key}) for span, attr, _, _ in graph.nodes],
        "edges": [((span, span_), {key.upper(): value for key, value in attr.items() if key}) for span, span_, attr in graph.edges],
        }
    return ret

def solve_ilp(graph):
    """
    Wrapper around construct_and_solve_sentence_ilp that converts into/out of sentence graph.
    """
    ret_nodes, ret_edges = construct_and_solve_sentence_ilp(to_datum(graph))

    assert len(ret_nodes) == len(graph.nodes)
    assert len(ret_edges) == len(graph.edges)

    # Update elements
    graph_ = SentenceGraph()
     # TODO: properly support sign and manner attributes.
    for span, _, sign, manner in graph.nodes:
        attr = Counter({key.lower(): value for key, value in ret_nodes[span].items()})
        attr[None] = 1.0 - sum(attr.values())
        graph_.nodes.append((span, attr, cargmax(sign), cargmax(manner)))
    for span, span_, _ in graph.edges:
        attr = Counter({key.lower(): value for key, value in ret_edges[span, span_].items()})
        attr[None] = 1.0 - sum(attr.values())
        graph_.edges.append((span, span_, attr))

    return graph_

def test_solve_ilp():
    graph = SentenceGraph()
    graph.nodes.extend([
        ((0,1), Counter({"value":.9, None:.1})      , Counter({None:1.}), Counter({None:1.}),),
        ((1,2), Counter({"value":.9, None:.1})      , Counter({None:1.}), Counter({None:1.}),),
        ((2,3), Counter({"label":.9, None:.1})      , Counter({None:1.}), Counter({None:1.}),),
        ((3,4), Counter({"label":.9, None:.1})      , Counter({None:1.}), Counter({None:1.}),),
        ((4,5), Counter({"theme":.9, None:.1})      , Counter({None:1.}), Counter({None:1.}),),
        ((5,6), Counter({"theme_mod":.99, None:.01}), Counter({None:1.}), Counter({None:1.}),),
        ((6,7), Counter({"label_mod":.99, None:.01}), Counter({None:1.}), Counter({None:1.}),),
        ])
    graph.edges.extend([
        ((2,3),(3,4), Counter({"frame":.9   , None: .1 })),
        ((0,1),(2,3), Counter({"instance":.9, None: .1 })),
        ((1,2),(3,4), Counter({"instance":.9, None: .1 })),
        ((4,5),(5,6), Counter({"frame":.4   , None: .6 })),
        ((2,3),(6,7), Counter({"instance":.4, None: .6 })),
        ])

    ret = solve_ilp(graph)
    assert len(ret.nodes) == len(graph.nodes)
    assert len(ret.edges) == len(graph.edges)
    # TODO: Add more qualitative checks.
