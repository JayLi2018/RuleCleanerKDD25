import re
from typing import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from rbbm_src.labelling_func_src.src.TreeRules import *
from rbbm_src.classes import FixMonitor, RepairConfig
from rbbm_src.dc_src.src.classes import (
    parse_rule_to_where_clause, 
    dc_violation_template,
    ops,
    eq_op,
    non_symetric_op,
    const_detect,
    get_operator
    )
import psycopg2
from string import Template
from dataclasses import dataclass
from typing import *
from collections import deque
from itertools import product
import copy
import time
from rbbm_src.holoclean.examples.holoclean_repair import main
import random
from math import ceil
import logging
from collections import defaultdict
from math import floor
from datetime import datetime
import os
from rbbm_src.dc_src.DCQueryTranslator import convert_dc_to_muse_rule,convert_dc_to_get_violation_tuple_ids
from rbbm_src.muse.running_example.running_example_adult import *
from rbbm_src import logconfig


logger = logging.getLogger(__name__)

dc_tuple_violation_template_targeted_t1=Template("SELECT DISTINCT t2.* FROM $table t1, $table t2 WHERE $dc_desc AND $tuple_desc")
dc_tuple_violation_template_targeted_t2=Template("SELECT DISTINCT t1.* FROM $table t1, $table t2 WHERE $dc_desc AND $tuple_desc")
dc_tuple_violation_template=Template("SELECT DISTINCT t2.* FROM $table t1, $table t2 WHERE $dc_desc;")


def parse_dc_to_tree_rule(dc_text):
    # input:a dc raw text
    # output: a TreeRule object
    # each predicate comes with 2 nodes:
    #  1.predicate node: describing the predicate text
    #  2.label node: a left child of the predicate node from 1 that says CLEAN
    #  3.set right node of node from 1 to the next predicate. if no predicate is left
    #    then set the right node to DIRTY
    cur_number=0
    predicates = dc_text.split('&')
    root_predicate = predicates[2]
    sign=get_operator(root_predicate)
    if(sign=='='):
        sign='=='
    root_node= PredicateNode(number=cur_number, pred=DCAttrPredicate(pred=root_predicate, operator=sign))
    cur_number+=1
    root_left_child= LabelNode(number=cur_number, label=CLEAN, pairs={DIRTY:[], CLEAN:[]}, used_predicates=set([]))
    tree_size=2
    root_node.left=root_left_child
    root_left_child.parent=root_node
    parent=root_node
    cur_number+=1
    for pred in predicates[3:]:
        sign=get_operator(pred)
        if(sign=='='):
            sign='=='
        if(not const_detect.search(pred)):
            cur_node = PredicateNode(number=cur_number, pred=DCAttrPredicate(pred=pred, operator=sign))
        else:
            cur_node = PredicateNode(number=cur_number, pred=DCConstPredicate(pred=pred, operator=sign))
        cur_number+=1
        
        left_child=LabelNode(number=cur_number, label=CLEAN, pairs={DIRTY:[], CLEAN:[]}, used_predicates=set([]))
        cur_node.left=left_child
        left_child.parent=cur_node

        parent.right=cur_node
        cur_node.parent=parent
        parent=cur_node
        tree_size+=2
        cur_number+=1

    last_right=LabelNode(number=cur_number, label=DIRTY, pairs={DIRTY:[], CLEAN:[]}, used_predicates=set([]))
    cur_number+=1
    parent.right=last_right
    last_right.parent=parent
    tree_size+=1
    return TreeRule(rtype='dc', root=root_node, size=tree_size, max_node_id=cur_number)

def find_tuples_in_violation(t_interest, conn, dc_text, target_table, targeted=True):
    if(non_symetric_op.search(dc_text)):
        q1= construct_query_for_violation(t_interest, 't1', dc_text, target_table, targeted)
        q2 = construct_query_for_violation(t_interest, 't2', dc_text, target_table, targeted)
        res = {'t1': pd.read_sql(q1, conn).to_dict('records'), 
        't2': pd.read_sql(q2, conn).to_dict('records')}
    else:
        q1= construct_query_for_violation(t_interest, 't1', dc_text, target_table, targeted)
        res = {'t1':pd.read_sql(q1, conn).to_dict('records')}
    return res 

def construct_query_for_violation(t_interest, role, dc_text, target_table, targeted): 
    predicates = dc_text.split('&')
    constants=[]
    need_tid=True 
    # if the constraint only has equals, we need to add an artificial
    # key (_tid_) to differentiate tuples in violation with the tuple it
    # self
    for pred in predicates[2:]:
        if(not eq_op.search(pred)):
            need_tid=False
        attr = re.search(r't[1|2]\.([-\w]+)', pred).group(1).lower()
        constants.append(f'{role}.{attr}=\'{t_interest[attr]}\'')
    constants_clause = ' AND '.join(constants)
    if(role=='t1'):
        template=dc_tuple_violation_template_targeted_t1
    else:
        template=dc_tuple_violation_template_targeted_t2
    if(targeted):
        r_q  = template.substitute(table=target_table, dc_desc=parse_rule_to_where_clause(dc_text),
                                           tuple_desc=constants_clause)

    else:
        r_q  = template.substitute(table=target_table, dc_desc=parse_rule_to_where_clause(dc_text))

    if(need_tid):
        if(role=='t1'):
            r_q+=f" AND t2._tid_!={t_interest['_tid_']}"
        else:
            r_q+=f" AND t1._tid_!={t_interest['_tid_']}"

    return r_q

def user_choose_confirmation_assignment(dds, desired_dcs, conn, table_name,):
    
    violations_to_rule_dict = {}
    violations_dict = {}
    print("choose user specified dd assignments!")
    logger.debug(f"desired_dcs: {desired_dcs}")
    for d in dds:
        complaint_df = pd.read_sql(f"select * from {table_name} where _tid_ = {d}", conn)
        complaint_tuples = complaint_df.to_dict('records')
        complaint_dicts = [{'tuple':c, 'expected_label':DIRTY} for c in complaint_tuples]
        for c in desired_dcs:
            violations_to_rule_dict, violations_dict = populate_violations(tree_rule=None, conn=conn, 
                rule_text=c, complaint=complaint_dicts[0], table_name=table_name, violations_dict=violations_dict, violations_to_rule_dict=violations_to_rule_dict,
            complaint_selection=True, check_existence_only=False, 
            user_input_pairs=False,
            pre_selected_pairs=None)

    print("violations_to_rule_dict:")
    print(violations_to_rule_dict)
    print("violations_dict")
    print(violations_dict)
    return violations_to_rule_dict, violations_dict
    # exit()

def gen_repaired_tree_rule(t_interest, desired_label, t_in_violation, target_attribute, tree_rule, role):
    # given a tuple of interest, the desired label (clean/dirty)
    # a tuple of violation, and a target attribute you want to fix
    # the rule with, return a modified tree rule object that return the 
    # desired label for the pair of tuples
    
    # Step #1, traverse through the given tree rule using the tuple pair (t_interest, t_in_violation)
    pair_to_eval_dict = {'t1':t_interest, 't2':t_in_violation}
    parent, end_node = tree_rule.evaluate(pair_to_eval_dict)
    cur_label = end_node.label
    # Step #2, add a new branch based on the target_attribute and its value in t_interest
    # It is possible the value between t_interest and t_in_violation has the same value in 
    # target_attribute, which wouldnt help
    cur_number=treerule.tree_size
    new_branch = PredicateNode(number=cur_number, pred=DCConstPredicate(f"EQ({role}.{target_attribute},'{t_interest[target_attribute]}')"))
    cur_number+=1
    new_branch.left=LabelNode(number=cur_number, label=cur_label, pairs={DIRTY:[], CLEAN:[]}, used_predicates=set([]))
    cur_number+=1
    new_branch.right=LabelNode(number=cur_number, label=desired_label, pairs={DIRTY:[], CLEAN:[]}, used_predicates=set([]))
    # Holoclean constant assignment example EQ(t1.HospitalOwner,'proprietary')
    if(parent.right is end_node):
        parent.right=new_branch
    else:
        parent.left=new_branch
        
    return tree_rule

def fix_violation(t_interest, desired_label, t_in_violation, tree_rule, role):
    unusable_attrs=['_tid_']
    # check "unavailable nodes", a node is unavailable when
    # the node contains equal operator
    cur_node_parent=None
    cur_node = tree_rule.root
    queue = deque([tree_rule.root])
    while(queue):
        cur_node = queue.popleft()
        if(isinstance(cur_node,PredicateNode)):
            if(cur_node.pred.operator=='=='):
                unusable_attrs.append(cur_node.pred.attr)
        if(cur_node.left):
            queue.append(cur_node.left)
        if(cur_node.right):
            queue.append(cur_node.right)

    available_attrs=list(set(set(list(t_interest)).difference(set(unusable_attrs))))
    if(not available_attrs):
        return None
    repaired_tree_rule=gen_repaired_tree_rule(t_interest, desired_label, t_in_violation, 
                                              available_attrs[0], tree_rule, role)
    return repaired_tree_rule

def construct_domain_dict(connection, table_name, exclude_cols=['_tid_']):
    res = {}
    cols = list(pd.read_sql(f'select * from "{table_name}" limit 1', connection))
    
    for c in exclude_cols:
        cols.remove(c)

    cur=connection.cursor()
    for c in cols:
        cur.execute(f'select distinct "{c}" from {table_name}')
        res[c]=set([x[0] for x in cur.fetchall()])

    return res

def fix_rules(repair_config, original_rules, conn, table_name, exclude_cols=['_tid_'], user_specify_pairs=False, pre_selected_pairs=None):
    rules = original_rules
    all_fixed_rules = []
    cur_fixed_rules = []
    domain_value_dict = construct_domain_dict(conn, table_name=table_name, exclude_cols=exclude_cols)
    fix_book_keeping_dict = {k:{} for k in original_rules}
    for r in rules:
        fix_book_keeping_dict[r]['deleted']=False
        treerule = parse_dc_to_tree_rule(r)
        fix_book_keeping_dict[r]['rule']=treerule
        fix_book_keeping_dict[r]['pre_fix_size']=treerule.size
        leaf_nodes = []
        if(not user_specify_pairs):
            for c in repair_config.complaints:
                leaf_nodes_with_complaints = populate_violations(tree_rule=treerule, conn=conn, rule_text=r, 
                    complaint=c, table_name=table_name, violations_dict=None,violations_to_rule_dict=None,
                    complaint_selection=False, check_existence_only=False, 
                    user_input_pairs=user_specify_pairs,
                    pre_selected_pairs=pre_selected_pairs)
        else:
            leaf_nodes_with_complaints = populate_violations(tree_rule=treerule, conn=conn, rule_text=r, 
                complaint=None, table_name=table_name, violations_dict=None,violations_to_rule_dict=None,
                complaint_selection=False, check_existence_only=False, 
                user_input_pairs=user_specify_pairs,
                pre_selected_pairs=pre_selected_pairs)

        for ln in leaf_nodes_with_complaints:
            if(ln not in leaf_nodes):
                # if node is already in leaf nodes, dont
                # need to add it again
                leaf_nodes.append(ln)

        for ln in leaf_nodes:
            ln=filter_assignments(ln)

        if(leaf_nodes):
            # its possible for certain rule we dont have any violations
            fixed_treerule = fix_violations(r, treerule, repair_config, leaf_nodes, domain_value_dict)
            fix_book_keeping_dict[r]['after_fix_size']=fixed_treerule.size
            if(fixed_treerule.size/fix_book_keeping_dict[r]['pre_fix_size']*repair_config.deletion_factor>=1):
                fix_book_keeping_dict[r]['deleted']=True
            fixed_treerule_text = treerule.serialize()
            fix_book_keeping_dict[r]['fixed_treerule_text']=fixed_treerule_text
        else:
            fix_book_keeping_dict[r]['after_fix_size']=treerule.size
            fixed_treerule_text = treerule.serialize()
            fix_book_keeping_dict[r]['fixed_treerule_text']=fixed_treerule_text

    return fix_book_keeping_dict

def filter_assignments(leaf_node):
    # given a leaf node
    # if label is DIRTY: do nothing
    # if label is CLEAN: if the assignment is DD, remove it
    if(leaf_node.label==DIRTY):
        return leaf_node
    leaf_node.pairs[DIRTY]=[]
    # leaf_node.pairs[CLEAN] = [x for x in leaf_node.pairs[CLEAN] if x['expected_label']!=DIRTY]
    return leaf_node

def print_fix_book_keeping_stats(config, bkeepdict, fix_performance):
    sizes_diff = []
    for k,v in bkeepdict.items():
        sizes_diff.append(v['after_fix_size']-v['pre_fix_size'])
    logger.debug('**************************\n')
    logger.debug(f"strategy: {config.strategy}, complaint_size={len(config.complaints)}, runtime: {config.runtime}")
    logger.debug(f"average size increase: {sum(sizes_diff)/len(sizes_diff)}")
    logger.debug('**************************\n')

    logger.debug("fixed rules")

    res = {'strategy':config.strategy, 'complaint_size': len(config.complaints), 'runtime': config.runtime, 'avg_size_increase': {sum(sizes_diff)/len(sizes_diff)}}
    res.update(fix_performance)
    return res

def find_available_repair(clean_pair, dirty_pair, domain_value_dict, used_predicates, all_possible=False):
    """
    given a leafnode, we want to find an attribute that can be used
    to give a pair of tuples the desired label.

    there are 2 possible resources:
    1: attribute equals/not equal attribute (NOTE: limited to the same attribute for now)
    2: attribute equals/not equals constant: 
        1. equals is just constant itself
        2. not equals is find any other value exist in the domain
    """
    # loop through every attribute to find one
    res = []
    # check clean pair

    if(not (clean_pair['t1']['is_dirty']==False and clean_pair['t2']['is_dirty']==False)):
        return res

    equal_assign_sign='!='
    not_equal_assign_sign='=='

        # start with attribute level and then constants
    cand=None
    for k in domain_value_dict:
        if((clean_pair['t1'][k]==clean_pair['t2'][k]) and \
           (dirty_pair['t1'][k]!=dirty_pair['t2'][k])):
            # check if the predicate is already present
            # in the current constraint
            cand = ('t1', k, 't2', k, equal_assign_sign)
            if(cand not in used_predicates):
                if(not all_possible):
                    return cand
                else:
                    res.append(cand)

        if((clean_pair['t1'][k]!=clean_pair['t2'][k]) and \
           (dirty_pair['t1'][k]==dirty_pair['t2'][k])):
            cand = ('t1', k, 't2', k, not_equal_assign_sign)
            if(cand not in used_predicates):
                if(not all_possible):
                    return cand
                else:
                    res.append(cand)

    for k in domain_value_dict:
        for v in domain_value_dict[k]:
            if((clean_pair['t1'][k]==v) and (dirty_pair['t1'][k]!=v)):
                cand = ('t1', k, v, equal_assign_sign)
                if(cand and cand not in used_predicates):
                    if(not all_possible):
                        return cand
                    else:
                        res.append(cand)
            if((clean_pair['t1'][k]!=v) and (dirty_pair['t1'][k]==v)):
                cand = ('t1', k, v, not_equal_assign_sign)                    
                if(cand and cand not in used_predicates):
                    if(not all_possible):
                        return cand
                    else:
                        res.append(cand)
            if((clean_pair['t2'][k]==v) and (dirty_pair['t2'][k]!=v)):
                cand = ('t2', k, v, equal_assign_sign)
                if(cand and cand not in used_predicates):
                    if(not all_possible):
                        return cand
                    else:
                        res.append(cand)

            if((clean_pair['t2'][k]!=v) and (dirty_pair['t2'][k]==v)):
                cand = ('t2', k, v, not_equal_assign_sign)
                if(cand and cand not in used_predicates):
                    if(not all_possible):
                        return cand
                    else:
                        res.append(cand)

        return res

def convert_tuple_fix_to_pred(the_fix, reverse=False):
    if(len(the_fix)==4):
        role, attr, const, sign = the_fix
        text_sign=None
        if(sign=='=='):
            if(reverse):
                sign='!='
                text_sign='IQ'
            else:
                text_sign='EQ'
        else:
            if(reverse):
                sign='=='
                text_sign='EQ'
            else:
                text_sign='IQ'

        return f"{text_sign}({role}.{attr},'{const}')", (role, attr, const, sign)

    elif(len(the_fix)==5):
        role1, attr1, role2, attr2, sign = the_fix
        if(sign=='=='):
            if(reverse):
                sign='!='
                text_sign='IQ'
            else:
                text_sign='EQ'
        else:
            if(reverse):
                sign='=='
                text_sign='EQ'
            else:
                text_sign='IQ'

        return f"{text_sign}({role1}.{attr1},{role2}.{attr2})", (role1, attr1, role2, attr2, sign)

def calculate_gini(node, the_fix):
    sign=None
    if(len(the_fix)==4):
        _, _, _, sign = the_fix
    elif(len(the_fix)==5):
        _, _, _, _, sign = the_fix
    
    right_leaf_dirty_cnt=0
    right_leaf_clean_cnt=0
    left_leaf_dirty_cnt=0
    left_leaf_clean_cnt=0

    if(len(the_fix)==4):
        role, attr, const, sign = the_fix
        for k in [CLEAN, DIRTY]:
            for p in node.pairs[k]:
                if(eval(f"p['{role}']['{attr}']{sign}'{const}'")):
                    if(k==DIRTY):
                        right_leaf_dirty_cnt+=1
                    else:
                        right_leaf_clean_cnt+=1
                else:
                    if(k==DIRTY):
                        left_leaf_dirty_cnt+=1
                    else:
                        left_leaf_clean_cnt+=1

    elif(len(the_fix)==5):
        role1, attr1, role2, attr2, sign = the_fix
        for k in [CLEAN, DIRTY]:
            for p in node.pairs[k]:
                if(eval(f"p['{role1}']['{attr1}']{sign}p['{role2}']['{attr2}']")):
                    if(k==DIRTY):
                        right_leaf_dirty_cnt+=1
                    else:
                        right_leaf_clean_cnt+=1
                else:
                    if(k==DIRTY):
                        left_leaf_dirty_cnt+=1
                    else:
                        left_leaf_clean_cnt+=1

    reverse_condition = False
    left_dirty_rate=(left_leaf_dirty_cnt)/(left_leaf_dirty_cnt+left_leaf_clean_cnt)
    right_dirty_rate=(right_leaf_dirty_cnt)/(right_leaf_clean_cnt+right_leaf_dirty_cnt)
    if(left_dirty_rate > right_dirty_rate):
        reverse_condition=True

    left_total_cnt = left_leaf_dirty_cnt+left_leaf_clean_cnt
    right_total_cnt = right_leaf_dirty_cnt+right_leaf_clean_cnt

    total_cnt=left_total_cnt+right_total_cnt

    gini_impurity = (left_total_cnt/total_cnt)*(1-((left_leaf_dirty_cnt/left_total_cnt)**2+(left_leaf_clean_cnt/left_total_cnt)**2))+ \
    (right_total_cnt/total_cnt)*(1-((right_leaf_dirty_cnt/right_total_cnt)**2+(right_leaf_clean_cnt/right_total_cnt)**2))

    
    return gini_impurity, reverse_condition

def redistribute_after_fix(tree_rule, node, the_fix, reverse=False):
    # there are some possible "side effects" after repair for a pair of violations
    # which is solving one pair can simutaneously fix some other pairs so we need 
    # to redistribute the pairs in newly added nodes if possible
    sign=None
    # cur_number=tree_rule.size

    cur_number=tree_rule.max_node_id+1
    if(len(the_fix)==4):
        _, _, _, sign = the_fix
    elif(len(the_fix)==5):
        _, _, _, _, sign = the_fix
    new_pred, modified_fix = convert_tuple_fix_to_pred(the_fix, reverse)
    if(len(the_fix)==4):
        new_predicate_node = PredicateNode(number=cur_number, pred=DCConstPredicate(pred=new_pred, operator=sign))
    elif(len(the_fix)==5):
        new_predicate_node = PredicateNode(number=cur_number, pred=DCAttrPredicate(pred=new_pred, operator=sign))
    new_predicate_node.is_added=True
    cur_number+=1
    new_predicate_node.left= LabelNode(number=cur_number, label=CLEAN, pairs={DIRTY:[], CLEAN:[]}, used_predicates=set([]))
    new_predicate_node.left.is_added=True
    cur_number+=1
    new_predicate_node.right=LabelNode(number=cur_number, label=DIRTY, pairs={DIRTY:[], CLEAN:[]}, used_predicates=set([]))
    new_predicate_node.right.is_added=True
    new_predicate_node.left.parent= new_predicate_node
    new_predicate_node.right.parent= new_predicate_node
    tree_rule.max_node_id=max(cur_number, tree_rule.max_node_id)

    if(node.label==CLEAN):
        node.parent.left=new_predicate_node
    else:
        node.parent.right=new_predicate_node

    new_predicate_node.parent = node.parent

    if(len(modified_fix)==4):
        role, attr, const, sign = modified_fix
        for k in [CLEAN, DIRTY]:
            for p in node.pairs[k]:
                if(eval(f"p['{role}']['{attr}']{sign}'{const}'")):
                    new_predicate_node.right.pairs[p['expected_label']].append(p)
                    new_predicate_node.right.used_predicates.add(modified_fix)
                else:
                    new_predicate_node.left.pairs[p['expected_label']].append(p)
                    new_predicate_node.left.used_predicates.add(modified_fix)

    elif(len(modified_fix)==5):
        role1, attr1, role2, attr2, sign = modified_fix
        for k in [CLEAN, DIRTY]:
            for p in node.pairs[k]:
                if(eval(f"p['{role1}']['{attr1}']{sign}p['{role2}']['{attr2}']")):
                    new_predicate_node.right.pairs[p['expected_label']].append(p)
                    new_predicate_node.right.used_predicates.add(modified_fix)
                else:
                    new_predicate_node.left.pairs[p['expected_label']].append(p)
                    new_predicate_node.left.used_predicates.add(modified_fix)


    new_predicate_node.pairs={CLEAN:{}, DIRTY:{}}

    return new_predicate_node

def check_tree_purity(tree_rule, start_number=0):
    root = locate_node(tree_rule, start_number)
    queue = deque([root])
    leaf_nodes = []
    while(queue):
        cur_node = queue.popleft()
        if(isinstance(cur_node,LabelNode)):
            leaf_nodes.append(cur_node)
        if(cur_node.left):
            queue.append(cur_node.left)
        if(cur_node.right):
            queue.append(cur_node.right)

    for n in leaf_nodes:
        for k in [CLEAN,DIRTY]:
            for p in n.pairs[k]:
                if(p['expected_label']!=n.label):
                    return False
    return True

def reverse_node_parent_condition(node):
    if(node.parent.pred.operator=='!='):
        node.parent.pred.operator='=='
        node.parent.pred.pred = node.parent.pred.pred.replace('IQ', 'EQ')
    else:
        node.parent.pred.operator='!='
        node.parent.pred.pred = node.parent.pred.pred.replace('EQ', 'IQ')

    old_left, old_right = node.parent.left, node.parent.right
    old_right.label=CLEAN 
    old_left.label=DIRTY
    node.parent.left=old_right
    node.parent.right=old_left
    node.parent.is_reversed=True

def fix_violations(tree_rule_text, treerule, repair_config, leaf_nodes, domain_value_dict):
    if(repair_config.strategy=='naive'):
        # initialize the queue to work with
        queue = deque([])
        for ln in leaf_nodes:
            queue.append(ln)
        while(queue):
            node = queue.popleft()
            new_parent_node=None
            # need to find a pair of violations that get the different label
            # in order to differentiate them
            if(node.pairs[CLEAN] and node.pairs[DIRTY]):
                the_fix = find_available_repair(node.pairs[CLEAN][0],
                 node.pairs[DIRTY][0], domain_value_dict, node.used_predicates)
                new_parent_node=redistribute_after_fix(treerule, node, the_fix)
            # handle the left and right child after redistribution
            else:
                if(not node.pairs[CLEAN] and not node.pairs[DIRTY]):
                    continue
                elif(node.pairs[CLEAN] and not node.pairs[DIRTY]):
                    if(node.label!=CLEAN):
                        reverse_node_parent_condition(node)
                        treerule.setsize(treerule.size+2)
                else:
                    if(node.label!=DIRTY):
                        reverse_node_parent_condition(node)
                        treerule.setsize(treerule.size+2)

                # if(check_tree_purity(treerule)):
                #     return treerule

            if(new_parent_node):
                still_inpure=False
                for k in [CLEAN,DIRTY]:
                    if(still_inpure):
                        break
                    for p in new_parent_node.left.pairs[k]:
                        if(p['expected_label']!=new_parent_node.left.label):
                            queue.append(new_parent_node.left)
                            still_inpure=True
                            break
                still_inpure=False          
                for k in [CLEAN,DIRTY]:
                    if(still_inpure):
                        break
                    for p in new_parent_node.right.pairs[k]:
                            if(p['expected_label']!=new_parent_node.right.label):
                                queue.append(new_parent_node.right)
                                still_inpure=True
                                break
                treerule.setsize(treerule.size+2)
        return treerule

    elif(repair_config.strategy=='information_gain'):
        # new implementation
        # 1. ignore the label of nodes at first
        # calculate the gini index of the split
        # and choose the best one as the solution
        # then based on the resulted majority expected labels
        # to assign the label for the children, if left get dirty
        # we flip the condition to preserve the ideal dc structure'
        # 2. during step 1, keep track of the used predicates to avoid
        # redundant repetitions
        queue = deque([])
        for ln in leaf_nodes:
            queue.append(ln)
        while(queue):
            node = queue.popleft()
            new_parent_node=None
            # need to find a pair of violations that get the different label
            # in order to differentiate them
            min_gini=1
            best_fix = None
            reverse_condition=False
            if(node.pairs[CLEAN] and node.pairs[DIRTY]):
                # need to examine all possible pair combinations
                considered_fixes = set()
                for pair in list(product(node.pairs[CLEAN], node.pairs[DIRTY])):
                    the_fixes = find_available_repair(pair[0],
                     pair[1], domain_value_dict, node.used_predicates,
                     all_possible=True)
                    for f in the_fixes:
                        if(f in considered_fixes):
                            continue
                        gini, reverse_cond =calculate_gini(node, f)
                        logger.debug(f"the fix: {f}, gini : {gini}")
                        considered_fixes.add(f)
                        if(gini<min_gini):
                            min_gini=gini
                            best_fix=f
                            best_fix_pair=pair
                            reverse_condition=reverse_cond
                if(best_fix):
                    logger.debug(f"best_fix for rule: {tree_rule_text}")
                    logger.debug(f"best_fix: {best_fix}")
                    logger.debug(f"best_fix_pair: {best_fix_pair}")
                    new_parent_node=redistribute_after_fix(treerule, node, best_fix, reverse_condition)
            else:
                if(not node.pairs[CLEAN] and not node.pairs[DIRTY]):
                    continue
                elif(node.pairs[CLEAN] and not node.pairs[DIRTY]):
                    if(node.label!=CLEAN):
                        reverse_node_parent_condition(node)
                        treerule.setsize(treerule.size+2)
                else:
                    if(node.label!=DIRTY):
                        reverse_node_parent_condition(node)
                        treerule.setsize(treerule.size+2)

            # handle the left and right child after redistribution
            if(new_parent_node):
                still_inpure=False
                for k in [CLEAN,DIRTY]:
                    if(still_inpure):
                        break
                    for p in new_parent_node.left.pairs[k]:
                        if(p['expected_label']!=new_parent_node.left.label):
                            queue.append(new_parent_node.left)
                            still_inpure=True
                            break
                still_inpure=False          
                for k in [CLEAN,DIRTY]:
                    if(still_inpure):
                        break
                    for p in new_parent_node.right.pairs[k]:
                            if(p['expected_label']!=new_parent_node.right.label):
                                queue.append(new_parent_node.right)
                                still_inpure=True
                                break
                treerule.setsize(treerule.size+2)

        return treerule

    elif(repair_config.strategy=='optimal'):
        # 1. create a queue with tree nodes
        # 2. need to deepcopy the tree in order to enumerate all possible trees
        for ln in leaf_nodes:
            queue = deque([])
            queue.append(ln.number)
        cur_fixed_tree = treerule
        while(queue):
            sub_root_number = queue.popleft()
            subqueue=deque([(cur_fixed_tree, sub_root_number, sub_root_number)])
            # triples are needed here, since: we need to keep track of the 
            # updated(if so) subtree root i n order to check purity from that node
            sub_node_pure=False
            while(subqueue and not sub_node_pure):
                prev_tree, leaf_node_number, subtree_root_number = subqueue.popleft()
                node = locate_node(prev_tree, leaf_node_number)
                if(node.pairs[CLEAN] and node.pairs[DIRTY]):
                    # need to examine all possible pair combinations
                    considered_fixes = set()
                    found_fix = False
                    for pair in list(product(node.pairs[CLEAN], node.pairs[DIRTY])):
                        if(found_fix):
                            break
                        the_fixes = find_available_repair(pair[0],
                         pair[1], domain_value_dict, node.used_predicates,
                         all_possible=True)
                        for f in the_fixes:
                            new_parent_node=None
                            if(f in considered_fixes):
                                continue
                            considered_fixes.add(f)
                            new_tree = copy.deepcopy(prev_tree)
                            node = locate_node(new_tree, node.number)
                            new_parent_node = redistribute_after_fix(new_tree, node, f)
                            if(leaf_node_number==sub_root_number):
                                # first time replacing subtree root, 
                                # the node number will change so we need 
                                # to replace it
                                subtree_root_number=new_parent_node.number
                            new_tree.setsize(new_tree.size+2)
                            if(check_tree_purity(new_tree, subtree_root_number)):
                                cur_fixed_tree = new_tree
                                found_fix=True
                                sub_node_pure=True
                                break
                            # else:
                            #     logger.debug("not pure yet, need to enqueue")
                            # handle the left and right child after redistribution
                            still_inpure=False
                            for k in [CLEAN,DIRTY]:
                                if(still_inpure):
                                    break
                                for p in new_parent_node.left.pairs[k]:
                                    if(p['expected_label']!=new_parent_node.left.label):
                                        new_tree = copy.deepcopy(new_tree)
                                        parent_node = locate_node(new_tree, new_parent_node.number)
                                        subqueue.append((new_tree, parent_node.left.number, subtree_root_number))
                                        still_inpure=True
                                        break
                            still_inpure=False          
                            for k in [CLEAN,DIRTY]:
                                if(still_inpure):
                                    break
                                for p in new_parent_node.right.pairs[k]:
                                    if(p['expected_label']!=new_parent_node.right.label):
                                        new_tree = copy.deepcopy(new_tree)
                                        parent_node = locate_node(new_tree, new_parent_node.number)
                                        # new_parent_node=redistribute_after_fix(new_tree, new_node, f)
                                        subqueue.append((new_tree, parent_node.right.number, subtree_root_number))
                                        still_inpure=True
                                        break
                    if(found_fix):
                        #break out of outer loop
                        break
                else:
                    if(node.pairs[CLEAN] and not node.pairs[DIRTY]):
                        if(node.label!=CLEAN):
                            reverse_node_parent_condition(node)
                            prev_tree.setsize(treerule.size+2)
                    elif(node.pairs[DIRTY] and not node.pairs[CLEAN]):
                        if(node.label!=DIRTY):
                            reverse_node_parent_condition(node)
                            prev_tree.setsize(treerule.size+2)
                    # if(check_tree_purity(prev_tree, subtree_root_number)):
                    #     found_fix=True
                    #     cur_fixed_tree = prev_tree
                    #     sub_node_pure=True

        return cur_fixed_tree 
        # list_of_repaired_trees = sorted(list_of_repaired_trees, key=lambda x: x[0].size, reverse=True)
        # return list_of_repaired_trees[0] 

    else:
        logger.debug("not a valid repair option")
        exit()

def locate_node(tree, number):
    queue = deque([tree.root])
    while(queue):
        cur_node = queue.popleft()
        if(cur_node.number==number):
            return cur_node
        if(cur_node.left):
            queue.append(cur_node.left)
        if(cur_node.right):
            queue.append(cur_node.right)
    logger.debug('cant find the node!')
    exit()

def populate_violations(tree_rule, conn, rule_text, complaint, table_name, violations_dict=None, violations_to_rule_dict=None,complaint_selection=False, check_existence_only=False, user_input_pairs=False,pre_selected_pairs=None):
    if(not user_input_pairs):
        pairs = []
        tuples_inviolation=find_tuples_in_violation(complaint['tuple'], conn, rule_text, table_name, targeted=True)
        if(check_existence_only):
            if('t1' in tuples_inviolation):
                if(tuples_inviolation['t1']):
                    return True 
            if('t2' in tuples_inviolation):
                if(tuples_inviolation['t2']):
                    return True 

        if(complaint_selection):
            if(complaint['tuple']['_tid_'] not in violations_dict):
                violations_dict[complaint['tuple']['_tid_']]=set()
        if('t1' in tuples_inviolation):
            for v in tuples_inviolation['t1']:
                pair = {'t1':complaint['tuple'], 't2':v, 'expected_label':complaint['expected_label']}
                pair_id = f"{complaint['tuple']['_tid_']}_{v['_tid_']}"
                pairs.append(pair)
                if(complaint_selection):
                    if(pair_id not in violations_to_rule_dict):
                        violations_to_rule_dict[pair_id] = {'pair':pair, 'rules':[]}
                    violations_to_rule_dict[pair_id]['rules'].append(rule_text)
                    violations_dict[complaint['tuple']['_tid_']].add(v['_tid_'])
        if('t2' in tuples_inviolation):
            for v in tuples_inviolation['t2']:
                pair = {'t1':v, 't2':complaint['tuple'], 'expected_label':complaint['expected_label']}
                pair_id = f"{v['_tid_']}_{complaint['tuple']['_tid_']}"
                pairs.append(pair)
                if(complaint_selection):
                    if(pair_id not in violations_to_rule_dict):
                        violations_to_rule_dict[pair_id] = {'pair':pair, 'rules':[]}
                    violations_to_rule_dict[pair_id]['rules'].append(rule_text)
                    violations_dict[complaint['tuple']['_tid_']].add(v['_tid_'])
    else:
        pairs=[x['pair'] for x in pre_selected_pairs]

    if(not complaint_selection):
        leaf_nodes = []
        for p in pairs:
            leaf_node = tree_rule.evaluate(p, ret='node')
            leaf_node.pairs[p['expected_label']].append(p)
            if(leaf_node not in leaf_nodes):
                leaf_nodes.append(leaf_node)
        return leaf_nodes
    else:
        return violations_to_rule_dict, violations_dict

def get_muse_results(table_name, conn, muse_dirties):
    # muse input are a list of tuples being deleted, need to identify the 
    cur=conn.cursor()
    cur.execute(f"select _tid_, is_dirty from {table_name}")
    res=cur.fetchall()
    real_dirties=[x[0] for x in res if x[1]==True]
    real_cleans=[x[0] for x in res if x[1]==False]
    logger.debug(f"real dirties:")
    logger.debug(real_dirties)
    logger.debug(f"real_cleans:")
    logger.debug(real_cleans)
    muse_dirty_ids=[int(x[-2]) for x in muse_dirties]
    logger.debug(f"muse_dirty_ids:")
    logger.debug(muse_dirty_ids)
    dc = set([x for x in muse_dirty_ids if x in real_cleans])
    dd = set([x for x in muse_dirty_ids if x in real_dirties])
    cc = set([x for x in real_cleans if x not in muse_dirty_ids])
    cd = set([x for x in real_dirties if x not in muse_dirty_ids])
    return dc, dd, cc, len(res), cd

def calculate_retrained_results(complaints, confirmations, new_dirties):

    new_dirties_ids = [int(x[-2]) for x in new_dirties]
    logger.debug(f"complaints:{complaints}")
    logger.debug(f'confirmations: {confirmations}')
    logger.debug(f"new_dirties: {new_dirties_ids}")
    complaint_fix_rate=1-len([x for x in complaints if x in new_dirties_ids])/len(complaints)
    confirm_preserve_rate=len([x for x in confirmations if x in new_dirties_ids])/len(confirmations)

    return complaint_fix_rate, confirm_preserve_rate

def dc_main(dc_input):
    # dc_file='/home/opc/chenjie/RBBM/experiments/dc/dc_sample_10'
    # table_name='adult'
    # dc_file='/home/opc/chenjie/RBBM/rbbm_src/muse/data/mas/flights_dcfinder_rules.txt'
    # dc_file='/home/opc/chenjie/RBBM/rbbm_src/muse/data/mas/tax_rules.txt'
    table_name=dc_input.table_name
    semantic_version=dc_input.semantic_version
    desired_dcs_file=dc_input.desired_dcs_file
    deletion_factor=dc_input.deletion_factor
    user_specify_pairs=dc_input.user_specify_pairs
    res=[]
    # rule_texts=[]
    filtered_rules=[]
    gt_good_rules=[]
    gt_bad_rules=[]

    try:
        log_map = { 'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
        }
        print(logconfig.root)
        logconfig.root.setLevel(log_map[dc_input.log_level])
        print(dc_input.log_level)
        print(logconfig.root)
    except KeyError as e:
        print('no such log level')

    try:
        with open(dc_input.dc_file, "r") as file:
            for line in file:
                # rule_texts.append(line.strip())
                # rules_from_line = [(table_name, x) for x in convert_dc_to_muse_rule(line, 'adult', 't1')]
                rule,gt=line.strip('\n').split(":")
                if(gt=='G'):
                    gt_good_rules.append(rule)
                elif(gt=='B'):
                    gt_bad_rules.append(rule)
                # muse_format=convert_dc_to_muse_rule(line, 'flights_new', 't1')
                # prefilter_format=convert_dc_to_get_violation_tuple_ids(line, 'flights_new','t1')
                muse_format=convert_dc_to_muse_rule(line, table_name)
                prefilter_format=convert_dc_to_get_violation_tuple_ids(line, table_name)
                res.append([table_name, rule, muse_format, prefilter_format])
        logger.debug(res)
        # exit()
    except FileNotFoundError:
        logger.debug("File not found.")
    except IOError:
        logger.debug("Error reading the file.")

    # query for real dirties
    db = DatabaseEngine("cr")
    database_reset(db,['adult','tax','flights_new'])
    db.close_connection()
    conn=psycopg2.connect('dbname=cr user=postgres')
    real_dirty_query = f"SELECT _tid_ from {table_name} where is_dirty=True"
    dataset_size_query = f"SELECT count(*) as cnt from {table_name}"
    cur=conn.cursor()
    cur.execute(real_dirty_query)
    real_dirty_ids = [x[0] for x in cur.fetchall()]
    logger.debug("real dirty ids")
    logger.debug(real_dirty_ids)
    cur.execute(dataset_size_query)
    dataset_size=cur.fetchone()[0]
    logger.debug("dataset size")
    logger.debug(dataset_size)
    pre_filter_thresh=dc_input.pre_filter_thresh
    # pre_filter_thresh=0.4
    # pre_filter_thresh=0.3
    # pre_filter_thresh=0.2
    # pre_filter_thresh=0.1

    # conn.close()

    for r in res:
        deleted_ids = []
        for q in r[-1]:
            logger.debug(q)
            cur.execute(q)
            r_deleted=[x[0] for x in cur.fetchall()]
            deleted_ids.extend(r_deleted)
        deleted_unique_ids=set(deleted_ids)
        incorrect_deleted_cnt=len([x for x in deleted_unique_ids if x not in real_dirty_ids])
        if(1-incorrect_deleted_cnt/dataset_size>pre_filter_thresh):
            filtered_rules.append(r)

    hit_gt_good_rules =  [x[1] for x in filtered_rules if x[1] in gt_good_rules]
    missed_gt_good_rules = [x for x in gt_good_rules if x not in hit_gt_good_rules]
    wrongly_kept_rules = [x[1] for x in filtered_rules if x[1] in gt_bad_rules]

    logger.debug('filtered_rules')
    logger.debug(filtered_rules)
    logger.debug(len(filtered_rules))
    logger.debug('\n')
    logger.debug('gt_good_rules')
    logger.debug(gt_good_rules)
    logger.debug(len(gt_good_rules))
    logger.debug('\n')
    logger.debug('gt_bad_rules')
    logger.debug(gt_bad_rules)
    logger.debug(len(gt_bad_rules))
    logger.debug('\n')
    logger.debug("hit_gt_good_rules")
    logger.debug(hit_gt_good_rules)
    logger.debug(len(hit_gt_good_rules))
    logger.debug('\n')
    logger.debug("missed_gt_good_rules")
    logger.debug(missed_gt_good_rules)
    logger.debug(len(missed_gt_good_rules))
    logger.debug('\n')
    logger.debug('wrongly_kept_rules')
    logger.debug(wrongly_kept_rules)
    logger.debug('\n')
    # exit()
    conn.close()
    muse_input_rules = [[x[0],x[2][0]] for x in filtered_rules]
    muse_dirties = muse_find_mss(rules=muse_input_rules, semantic_version='ind')
    # exit()
    tupled_muse_dirties=[x[1].strip('()').split(',') for x in muse_dirties]

    conn=psycopg2.connect('dbname=cr user=postgres')
    # dc, dd, cc, total_cnt = get_muse_results('flights_new', conn, tupled_muse_dirties)
    dc, dd, cc, total_cnt, cd = get_muse_results(table_name, conn, tupled_muse_dirties)


    # do a check and construct user complaint set
    complain_violations_dict={}
    complain_violations_to_rule_dict = {}
    confirm_violations_dict={}
    confirm_violations_to_rule_dict = {}
    for w in dc:
        complaint_df = pd.read_sql(f"select * from {table_name} where _tid_ = {w}", conn)
        complaint_tuples = complaint_df.to_dict('records')
        complaint_dicts = [{'tuple':c, 'expected_label':CLEAN} for c in complaint_tuples]
        for r in res:
            complain_violations_to_rule_dict, complain_violations_dict=populate_violations(tree_rule=None, 
                conn=conn, rule_text=r[1], complaint=complaint_dicts[0], table_name=table_name, 
                violations_to_rule_dict=complain_violations_to_rule_dict,
                violations_dict=complain_violations_dict, complaint_selection=True,
                check_existence_only=False, user_input_pairs=False,
                pre_selected_pairs=None)

    logger.debug(f"dc:{dc}")
    logger.debug(f"len(dc)={len(dc)}")
    logger.debug(f"dd:{dd}")
    logger.debug(f"len(dd)={len(dd)}")
    logger.debug(f"cc:{cc}")
    logger.debug(f"len(cc)={len(cc)}")
    logger.debug(f"cd:{cd}")
    logger.debug(f"len(cd)={len(cd)}")
    logger.debug(f"before fix, the global accuracy is {(len(dd)+len(cc))/(total_cnt)}")

    complaint_size=floor(dc_input.user_input_size*dc_input.complaint_ratio)
    confirmation_size=dc_input.user_input_size-complaint_size
    current_complaint_cnt=0
    complaint_set_completed=False


    test_rules = [x[1] for x in filtered_rules]

    if(user_specify_pairs==True):
        complaint_tuples = set([])
        confirmation_tuples = set([])
        complaints_input=[]
        confirmations_input=[]
        complaint_assignment_ids = set([])
        comfirm_assignment_ids = set([])
        logger.debug("complain_violations_dict")
        logger.debug(complain_violations_dict)
        logger.debug("confirm_violations_dict")
        logger.debug(confirm_violations_dict)

        total_length = sum(len(sub_list) for sub_list in complain_violations_dict.values())
        complain_probabilities = [len(sub_list) / total_length for sub_list in complain_violations_dict.values()]

        cur_complaint_cnt = 0

        while(cur_complaint_cnt<complaint_size):
            logger.debug("building complaint set...")
            logger.debug(f"complain_violations_dict: {complain_violations_dict}")
            logger.debug(f"complain_probabilities: {complain_probabilities}")
            # exit()
            selected_key = random.choices(list(complain_violations_dict.keys()), complain_probabilities)[0]
            selected_list = complain_violations_dict[selected_key]
            # we only want DC-DC to be included
            selected_list = [x for x in selected_list if x in dc]
            logger.debug(f"selected_list: {selected_list}")
            if(selected_list):
                selected_element = random.choice(list(selected_list))
                # cand_id1, cand_id2 = f'{str(selected_key)}_{str(selected_element)}',f'{str(selected_element)}_{str(selected_key)}'
                cand_id1 = f'{str(selected_key)}_{str(selected_element)}'
                logger.debug(f"cand_id1: {cand_id1}")
                if(cand_id1 in complain_violations_to_rule_dict and cand_id1 not in complaint_assignment_ids):
                    complaint_assignment_ids.add(cand_id1)
                    complaints_input.append(complain_violations_to_rule_dict[cand_id1])
                    cur_complaint_cnt+=1
                    complaint_tuples.add(selected_key)

        logger.debug(f"complaint_tuples: {complaint_tuples}")
        logger.debug(f"complaint_assignment_ids: {complaint_assignment_ids}")
        logger.debug("building confirmation set...")
        
        desired_dcs=[]
        try:
            with open(desired_dcs_file, "r") as file:
                for line in file:
                    desired_dcs.append(line)
        except FileNotFoundError:
            logger.debug("File not found.")
        except IOError:
            logger.debug("Error reading the file.")

        confirm_violations_to_rule_dict, confirm_violations_dict = user_choose_confirmation_assignment(dds=dd, 
            desired_dcs=desired_dcs,conn=conn, table_name=table_name)

        total_length = sum(len(sub_list) for sub_list in confirm_violations_dict.values())
        confirm_probabilities = [len(sub_list) / total_length for sub_list in confirm_violations_dict.values()]
        cur_confirmation_cnt = 0
        while(cur_confirmation_cnt<confirmation_size):
            selected_key =  random.choices(list(confirm_violations_dict.keys()), confirm_probabilities)[0]
            selected_list = confirm_violations_dict[selected_key]
            logger.debug(f"selected_list: {selected_list}")
            # we only want DD-DD to be included
            selected_list = [x for x in selected_list if x in dd]
            if(selected_list):
                selected_element = random.choice(list(selected_list))
                # cand_id1, cand_id2 = f'{str(selected_key)}_{str(selected_element)}',f'{str(selected_element)}_{str(selected_key)}'
                cand_id1=f'{str(selected_key)}_{str(selected_element)}'
                logger.debug(f"cand_id1: {cand_id1}")
                if(cand_id1 in confirm_violations_to_rule_dict and cand_id1 not in comfirm_assignment_ids):
                    comfirm_assignment_ids.add(cand_id1)
                    confirmations_input.append(confirm_violations_to_rule_dict[cand_id1])
                    cur_confirmation_cnt+=1
                    confirmation_tuples.add(selected_key)

        logger.debug(f"confirmation_tuples: {confirmation_tuples}")
        logger.debug(f"comfirm_assignment_ids: {comfirm_assignment_ids}")
        logger.debug("complaints:")
        logger.debug(complaints_input)

        for cp in complaints_input:
            df=pd.read_sql(f"select * from {table_name} where _tid_={cp['pair']['t1']['_tid_']} or _tid_={cp['pair']['t2']['_tid_']}", con=conn)
            logger.debug('---------------------------------------')
            logger.debug(df)
            logger.debug(f"violated rules: {cp['rules']}")
            logger.debug(f"violated rules count: {len(cp['rules'])}")
            logger.debug('---------------------------------------\n')

        logger.debug("confirmations:")
        logger.debug(confirmations_input)
        for cf in confirmations_input:
            df=pd.read_sql(f"select * from {table_name} where _tid_={cf['pair']['t1']['_tid_']} or _tid_={cf['pair']['t2']['_tid_']}", con=conn)
            logger.debug('---------------------------------------')
            logger.debug(df)
            logger.debug(f"violated rules: {cf['rules']}")
            logger.debug(f"violated rules count: {len(cf['rules'])}")
            logger.debug('---------------------------------------\n')

        user_input=[]
        user_input.extend(complaints_input)
        user_input.extend(confirmations_input)
        rc = RepairConfig(strategy=dc_input.strategy, deletion_factor=0.000001, complaints=user_input, acc_threshold=0.8, runtime=0)
        start = time.time()
        bkeepdict = fix_rules(repair_config=rc, original_rules=test_rules, conn=conn, table_name=table_name, exclude_cols=['_tid_','is_dirty'], user_specify_pairs=user_specify_pairs,
            pre_selected_pairs=user_input)
        end = time.time()
        new_rules = []
        deleted_cnt = 0

        timestamp = datetime.now()
        timestamp_str = timestamp.strftime('%Y%m%d%H%M%S')
        result_dir = f'./{dc_input.experiment_name}_{timestamp}'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        ki = 1
        for kt,vt in bkeepdict.items():
            with open(f"{result_dir}/{timestamp_str}_tree_{ki}_dot_file", 'a') as file:
                comments=f"// presize: {bkeepdict[kt]['pre_fix_size']}, after_size: {bkeepdict[kt]['after_fix_size']}, \
                deleted: {bkeepdict[kt]['deleted']}"
                dot_file=bkeepdict[kt]['rule'].gen_dot_string(comments)
                file.write(dot_file)                
            ki+=1
            
        for k in bkeepdict:
            if(bkeepdict[k]['deleted']==False):
                new_rules.extend(bkeepdict[k]['fixed_treerule_text'])
            else:
                deleted_cnt+=1

        logger.debug(f"new_rules")
        logger.debug(new_rules)
        logger.debug(f'"len(new_rules): {len(new_rules)}')
        conn.close()
        new_muse_program= []
        logger.debug("new_rules")
        logger.debug(new_rules)
        for r in new_rules:
            # new_muse_program.extend([(table_name, x) for x in convert_dc_to_muse_rule(r, 'flights_new', 't1')])
            new_muse_program.extend([(table_name, x) for x in convert_dc_to_muse_rule(r, table_name)])
        muse_dirties_new = muse_find_mss(rules=new_muse_program, semantic_version='ind')
        tupled_muse_dirties_new=[x[1].strip('()').split(',') for x in muse_dirties_new]
        logger.debug(len(tupled_muse_dirties_new))
        logger.debug(tupled_muse_dirties_new)

        conn=psycopg2.connect('dbname=cr user=postgres')
        new_dc, new_dd, new_cc, total_cnt_new, new_cd = get_muse_results(table_name, conn, tupled_muse_dirties_new)

        logger.debug(f"new_dc:{new_dc}")
        logger.debug(f"len(new_dc)={len(new_dc)}")
        logger.debug(f"new_dd:{new_dd}")
        logger.debug(f"len(new_dd)={len(new_dd)}")
        logger.debug(f"new_cc:{new_cc}")
        logger.debug(f"len(new_cc)={len(new_cc)}")
        logger.debug(f"new_cd:{cd}")
        logger.debug(f"len(new_cd)={len(new_cd)}")
        # 'complaints', 'confirmations', and 'new_dirties'
        fix_rate, confirm_preserve_rate = calculate_retrained_results(complaints=complaint_tuples, confirmations=confirmation_tuples, new_dirties=tupled_muse_dirties_new)
        logger.debug(f"before fix, the global accuracy is {(len(dd)+len(cc))/(total_cnt)}")
        logger.debug(f"after fix, fix_rate={fix_rate}, confirm_preserve_rate={confirm_preserve_rate}, the global accuracy is {(len(new_dd)+len(new_cc))/total_cnt_new}")
        logger.debug(f"before initial training, we had {len(res)} rules as input, after predeletion, we have {len(test_rules)} rules as input, after fix, we deleted {deleted_cnt} rules")

        # for k in bkeepdict:
        #     new_rules.extend(bkeepdict[k]['fixed_treerule_text'])
        logger.debug(bkeepdict)
        rc.runtime=end-start
        for r in new_rules:
            logger.debug(r+'\n')

    else:
        complaint_input = []
        for vk,vv in violations_dict.items():
            if(complaint_set_completed):
                break
            current_dc_added = False
            for c in vv:
                if(c in dc):
                    logger.debug(f"(c, c) case found ({vk}, {c})")
                    logger.debug(f"complaint_input added {c}")
                    complaint_input.add(c)
                    current_complaint_cnt=len(complaint_input)
                    if(not current_dc_added):
                        complaint_input.add(vk)
                        current_complaint_cnt=len(complaint_input)
                        current_dc_added=True
                    if(current_complaint_cnt>=complaint_size):
                        complaint_set_completed=True
                        break
        complaints_input=list(complaint_input)
        if(not complaint_set_completed):
            size_needed=complaint_size-len(complaint_input)
            rest_avaiable_dcs = [x for x in dc if x not in complaint_input]
            complaints_input.extend(random.sample(rest_avaiable_dcs, size_needed))

        # complaint_input = random.sample(dc, complaint_size)
        logger.debug(f"complaint_input:{complaint_input}")
        confirmations_input = random.sample(dd, confirmation_size)
        logger.debug(f"confirm_input:{confirm_input}")

        # stats being used to delete rules pre fix algorithm
        cnt_non_violation_on_confirmation=cnt_violation_on_complaints=0

        rules_with_violations = {r[1]:{'violate_confirm_cnt':0, 'violate_complaint_cnt':0} for r in res}

        for l in complaints_input:
            complaint_df = pd.read_sql(f"select * from {table_name} where _tid_ = {l}", conn)
            complaint_tuples = complaint_df.to_dict('records')
            complaint_dicts = [{'tuple':c, 'expected_label':CLEAN} for c in complaint_tuples]
            for r in res:
                has_violation=populate_violations(tree_rule=None, conn=conn, rule_text=r[1], complaint=complaint_dicts[0], table_name=table_name, 
                    violations_dict=None, complaint_selection=False, check_existence_only=True,
                    user_specify_pairs=False,user_input_pairs=False,pre_selected_pairs=None)

                if(has_violation):
                    rules_with_violations[r[1]]['violate_complaint_cnt']+=1

        for f in confirmations_input:
            confirm_df = pd.read_sql(f"select * from {table_name} where _tid_ = {f}", conn)
            confirm_tuples = confirm_df.to_dict('records')
            confirm_dicts = [{'tuple':c, 'expected_label':DIRTY} for c in confirm_tuples]
            for r in res:
                has_violation=populate_violations(tree_rule=None, conn=conn, rule_text=r[1], complaint=confirm_dicts[0], table_name=table_name, 
                    violations_dict=None, complaint_selection=False, check_existence_only=True,
                    user_specify_pairs=False,user_input_pairs=False,pre_selected_pairs=None)
                if(has_violation):
                    rules_with_violations[r[1]]['violate_confirm_cnt']+=1

        logger.debug('rules_with_violations')
        logger.debug(rules_with_violations)

        logger.debug(f" running user input: size: complaint(dirties but should be clean): {len(complaint_input)}, confirmation(clean and should be clean):\
         {len(confirm_input)}, version: information gain....")
        logger.debug(f"on complaint size {c}")
        user_input = []

        if(complaint_size>0):
            complaints_df = pd.read_sql(f"select * from {table_name} where _tid_ in ({','.join([str(x) for x in complaint_input])})", conn)
            logger.debug(f"select * from {table_name} where _tid_ in ({','.join([str(x) for x in complaint_input])})")
            logger.debug(complaints_df)
            complaints_df.to_csv('complaint_input_tuples.csv', index=False)
            complaint_tuples = complaints_df.to_dict('records')
            complaint_dicts = [{'tuple':c, 'expected_label':CLEAN} for c in complaint_tuples]
        else:
            complaint_dicts=[]

        if(confirmation_size>0):
            confirm_df = pd.read_sql(f"select * from {table_name} where _tid_ in ({','.join([str(x) for x in confirm_input])})", conn)
            logger.debug(confirm_df)
            confirm_df.to_csv('confirm_input_tuples.csv', index=False)
            logger.debug(f"select * from {table_name} where _tid_ in ({','.join([str(x) for x in confirm_input])})")
            confirm_tuples = confirm_df.to_dict('records')
            confirm_dicts = [{'tuple':c, 'expected_label':DIRTY} for c in confirm_tuples]
        else:
            confirm_dicts = []
        user_input.extend(complaint_dicts)
        user_input.extend(confirm_dicts)

        # rule_file = open(dc_file, 'r')
        # test_rules = [l.strip() for l in rule_file.readlines()]

        rc = RepairConfig(strategy=strateg, deletion_factor=0.5, complaints=user_input, acc_threshold=0.8, runtime=0)
        start = time.time()
        bkeepdict = fix_rules(repair_config=rc, original_rules=test_rules, conn=conn, table_name=table_name, exclude_cols=['_tid_','is_dirty'])
        end = time.time()
        new_rules = []
        deleted_cnt = 0
        for k in bkeepdict:
            if(bkeepdict[k]['deleted']==False):
                new_rules.extend(bkeepdict[k]['fixed_treerule_text'])
            else:
                deleted_cnt+=1


        logger.debug(f"new_rules")
        logger.debug(new_rules)
        logger.debug(f'"len(new_rules): {len(new_rules)}')
        conn.close()
        new_muse_program= []
        logger.debug("new_rules")
        logger.debug(new_rules)
        for r in new_rules:
            # new_muse_program.extend([(table_name, x) for x in convert_dc_to_muse_rule(r, 'flights_new', 't1')])
            new_muse_program.extend([(table_name, x) for x in convert_dc_to_muse_rule(r, table_name)])

        logger.debug(f"new_muse_program: {new_muse_program}")
        muse_dirties_new = muse_find_mss(rules=new_muse_program, semantic_version='ind')
        tupled_muse_dirties_new=[x[1].strip('()').split(',') for x in muse_dirties_new]
        logger.debug(len(tupled_muse_dirties_new))
        logger.debug(tupled_muse_dirties_new)
        # conn=psycopg2.connect('dbname=cr user=postgres')
        # cur = conn.cursor()
        conn=psycopg2.connect('dbname=cr user=postgres')

        # dc_new, dd_new, cc_new, total_cnt_new = get_muse_results('flights_new', conn, tupled_muse_dirties_new)
        dc_new, dd_new, cc_new, total_cnt_new, cd_new= get_muse_results(table_name, conn, tupled_muse_dirties_new)

        logger.debug(f"complaint_input:{complaint_input}")
        logger.debug(f"confirm_input:{confirm_input}")
        logger.debug(f"dc:{dc}")
        logger.debug(f"dd:{dd}")
        logger.debug(f"cc:{cc}")
        logger.debug(f"new_dc:{dc_new}")
        logger.debug(f"new_dd:{dd_new}")
        logger.debug(f"new_cc:{cc_new}")
        fix_rate, confirm_preserve_rate = calculate_retrained_results(complaints=complaint_input, confirmations=confirm_input, new_dirties=tupled_muse_dirties_new)
        logger.debug(f"before fix, the global accuracy is {(len(dd)+len(cc))/(total_cnt)}")
        logger.debug(f"after fix, fix_rate={fix_rate}, confirm_preserve_rate={confirm_preserve_rate}, the global accuracy is {(len(dd_new)+len(cc_new))/total_cnt_new}")
        logger.debug(f"before initial training, we had {len(res)} rules as input, after predeletion, we have {len(test_rules)} rules as input, after fix, we deleted {deleted_cnt} rules")

        # for k in bkeepdict:
        #     new_rules.extend(bkeepdict[k]['fixed_treerule_text'])
        logger.debug(bkeepdict)

        rc.runtime=end-start

        for r in new_rules:
            logger.debug(r+'\n')

    # if(not os.path.exists(result_dir+'/'+timestamp_str+'_experiment_stats')):
    #     with open(result_dir+'/'+timestamp_str+'_experiment_stats', 'w') as file:
    #         # Write some text to the file
    #         file.write('strat,runtime,avg_tree_size_increase,num_complaints,confirmation_cnt,global_accuracy,fix_rate,confirm_preserve_rate,new_global_accuracy,prev_signaled_cnt,new_signaled_cnt,' +\
    #             'num_functions,deletion_factor,post_fix_num_funcs,num_of_funcs_processed_by_algo,complaint_reached_max,confirm_reached_max\n')
    # with open(result_dir+'/'+timestamp_str+'_experiment_stats', 'a') as file:
    #     # Write the row to the file
    #     file.write(f'{strat},{runtime},{avg_tree_size_increase},{num_complaints},{num_confirm},{round(global_accuracy,3)},{round(fixed_rate,3)},{round(confirm_preserve_rate,3)},'+\
    #         f'{round(new_global_accuracy,3)},{old_signaled_cnt},{new_signaled_cnt},{num_funcs},{deletion_factor},{post_fix_num_funcs},{num_of_funcs_processed_by_algo},{complaint_reached_max},{confirm_reached_max}\n')

    
# if __name__ == '__main__':
#     conn=psycopg2.connect('dbname=holo user=postgres')
#     cur = conn.cursor()

#     input_csv_dir = '/home/opc/chenjie/RBBM/experiments/dc/'
#     # input_csv_dir = '/home/opc/chenjie/holoclean/testdata/'
#     # input_csv_file = 'hospital_1000.csv'
#     # input_csv_file = 'gt_emp.csv'
#     # input_csv_file = 'dirty_hospital.csv'
#     # input_csv_file = 'hospital.csv'
#     input_csv_file = 'dirty_hospital_coosa.csv'
#     # input_csv_file='dirty_2_col_hospital.csv'

#     input_dc_dir = input_csv_dir
#     # input_dc_file = 'hospital_1000_constraints.txt'
#     input_dc_file = 'hospital_1000_constraints.txt'

#     ground_truth_dir = input_csv_dir
#     # ground_truth_file = 'gt_emp_hoclean_format.csv'
#     # ground_truth_file = 'gt_hospital_holoclean_format.csv'
#     ground_truth_file = 'gt_hospital_1k.csv'
#     # ground_truth_file = 'hospital_1000_clean.csv'
#     # ground_truth_file='gt_hospital_2col_1k_manually_changed.csv'
#     conn.autocommit=True
#     cur=conn.cursor()
#     input_file=input_csv_dir+input_csv_file
#     table_name=input_csv_file.split('.')[0]
#     cols=None
#     num_lines=0
#     try:
#         with open(input_file) as f:
#             first_line = f.readline()
#             num_lines = sum(1 for line in f)
#     except Exception as e:
#         logger.debug(f'cant read file {input_file}')
#         exit()
#     else:
#         cols=first_line.split(',')
#         cols=[c.strip() for c in cols]
#     # drop preexisted repaired records 
#     select_old_repairs_q = f"""
#     SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
#     WHERE TABLE_NAME LIKE '{table_name}_repaired_%' AND TABLE_TYPE = 'BASE TABLE'
#     """
#     cur.execute(select_old_repairs_q)

#     for records in cur.fetchall():
#         drop_q = f"drop table if exists {records[0]}"
#         cur.execute(drop_q)

#     main(table_name=table_name, csv_dir=input_csv_dir, 
#         csv_file=input_csv_file, dc_dir=input_dc_dir, dc_file=input_dc_file, gt_dir=ground_truth_dir, 
#         gt_file=ground_truth_file, initial_training=True)

#     # confirm_dirty, expected_clean, expected_dirty, before_fix_correct_repair_cnt, before_fix_repair_cnt
#     dirties, dc, ddw, ddc, cc, cd, concated_df = get_holoclean_results(table_name, conn, cols)
#     ddw_dfs = []
#     ddc_dfs = []
#     dc_dfs = []
#     for k in ddw:
#         ddw_dfs.append(concated_df[concated_df['_tid_']==k])
#     for k in ddc:
#         ddc_dfs.append(concated_df[concated_df['_tid_']==k])
#     for k in dc:
#         dc_dfs.append(concated_df[concated_df['_tid_']==k])
#     # expected labels:
#     # correct_repairs: DIRTY
#     # clean_tuples: CLEAN
#     # still_dirty_tuples: DIRTY
#     if(ddw_dfs):
#         pd.concat(ddw_dfs).to_csv('ddws.csv', index=False)
#     if(ddc_dfs):
#         pd.concat(ddc_dfs).to_csv('ddcs.csv', index=False)
#     if(dc_dfs):
#         pd.concat(dc_dfs).to_csv('dcs.csv', index=False)

#     logger.debug(f"before fixes, the accuracy of holoclean is {(len(ddc)+len(cc))/(len(dirties)+len(dc)+len(cc)+len(cd))}")

#     results = []

#     dirty_expected_clean = []
#     dirty_expected_clean.extend(list(dc))
#     dirty_expected_dirty = []
#     dirty_expected_dirty.extend(list(ddw))
#     dirty_expected_dirty.extend(list(ddc))
#     clean_expected_clean = []
#     clean_expected_clean.extend(list(cc))

#     # for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
#     for p in [1]:

#     # for p in [0.1]:
#         # confirm_dirty_input = random.sample(dirty_expected_dirty, ceil(len(dirty_expected_dirty)*p))
#         # dirty_input= random.sample(dirty_expected_dirty, ceil(len(dirty_expected_dirty)*p))
#         # clean_input= random.sample(dirty_expected_clean, ceil(len(dirty_expected_clean)*p))

#         if(dirty_expected_dirty):
#             dirty_input= random.sample(dirty_expected_dirty, 1)
#         else:
#             logger.debug("no dirties !")
#             exit()
#         if(dirty_expected_clean):
#             clean_input= random.sample(dirty_expected_clean, 1)
#         else:
#             clean_input= random.sample(clean_expected_clean, 1)


#         logger.debug(f"dirty_input: {concated_df[concated_df['_tid_']==dirty_input[0]]}")
#         logger.debug(f"clean_input: {concated_df[concated_df['_tid_']==clean_input[0]]}")







#         # complaint_size_for_each_label = [3]
#         # # complaint_size_for_each_label = [2]
#         versions = ['information gain']
#         # versions = ['optimal']

#         # for c in complaint_size_for_each_label:
#         for v in versions:
#             logger.debug(f" running complaint: size: dity: {len(dirty_input)}, clean: {len(clean_input)}, version: {v}....")

#             complaints_dirty_df = pd.read_sql(f"select * from {table_name} where _tid_ in ({','.join([str(x) for x in dirty_input])})", conn)
#             logger.debug(f"select * from {table_name} where _tid_ in ({','.join([str(x) for x in dirty_input])})")
#             complaints_clean_df = pd.read_sql(f"select * from {table_name} where _tid_ in ({','.join([str(x) for x in clean_input])})", conn)
#             logger.debug(f"select * from {table_name} where _tid_ in ({','.join([str(x) for x in clean_input])})")
#             expected_dirty_tuples = complaints_dirty_df.to_dict('records')
#             expected_clean_tuples = complaints_clean_df.to_dict('records')
#             expected_dirty_dicts = [{'tuple':c, 'expected_label':DIRTY} for c in expected_dirty_tuples]
#             expected_clean_dicts = [{'tuple':c, 'expected_label':CLEAN} for c in expected_clean_tuples]
#             complaints = []
#             complaints.extend(expected_dirty_dicts)
#             complaints.extend(expected_clean_dicts)

#             rule_file = open(input_dc_dir+input_dc_file, 'r')
#             test_rules = [l.strip() for l in rule_file.readlines()]

#             # test_rules=[
#             # 't1&t2&EQ(t1.occupation,t2.occupation)&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.race,t2.race)&IQ(t1.sex,t2.sex)',
#             # 't1&t2&EQ(t1.marital-status,t2.marital-status)&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.relationship,t2.relationship)&IQ(t1.income,t2.income)',
#             # 't1&t2&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.income,t2.income)&EQ(t1.relationship,t2.relationship)&IQ(t1.marital-status,t2.marital-status)',
#             # 't1&t2&EQ(t1.education,t2.education)&IQ(t1.sex,t2.sex)&IQ(t1.native-country,t2.native-country)&EQ(t1.relationship,t2.relationship)',
#             # 't1&t2&EQ(t1.education,t2.education)&EQ(t1.marital-status,t2.marital-status)&IQ(t1.relationship,t2.relationship)&EQ(t1.sex,t2.sex)&IQ(t1.workclass,t2.workclass)',
#             # 't1&t2&EQ(t1.marital-status,t2.marital-status)&EQ(t1.age,t2.age)&IQ(t1.race,t2.race)',
#             # 't1&t2&EQ(t1.education,t2.education)&EQ(t1.occupation,t2.occupation)&IQ(t1.race,t2.race)&IQ(t1.income,t2.income)',
#             # 't1&t2&EQ(t1.education,t2.education)&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.race,t2.race)&IQ(t1.workclass,t2.workclass)',
#             # 't1&t2&EQ(t1.occupation,t2.occupation)&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.income,t2.income)&EQ(t1.relationship,t2.relationship)',
#             # 't1&t2&EQ(t1.education,t2.education)&EQ(t1.marital-status,t2.marital-status)&EQ(t1.workclass,t2.workclass)&IQ(t1.native-country,t2.native-country)',
#             # 't1&t2&EQ(t1.education,t2.education)&IQ(t1.native-country,t2.native-country)&EQ(t1.relationship,t2.relationship)&IQ(t1.workclass,t2.workclass)',
#             # 't1&t2&EQ(t1.occupation,t2.occupation)&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.race,t2.race)&IQ(t1.workclass,t2.workclass)',
#             # 't1&t2&EQ(t1.marital-status,t2.marital-status)&EQ(t1.occupation,t2.occupation)&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.race,t2.race)',
#             # 't1&t2&EQ(t1.education,t2.education)&EQ(t1.occupation,t2.occupation)&EQ(t1.age,t2.age)&EQ(t1.relationship,t2.relationship)',
#             # 't1&t2&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.sex,t2.sex)&IQ(t1.native-country,t2.native-country)&EQ(t1.relationship,t2.relationship)',
#             # 't1&t2&EQ(t1.education,t2.education)&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.race,t2.race)&IQ(t1.income,t2.income)',
#             # 't1&t2&IQ(t1.age,t2.age)&IQ(t1.race,t2.race)&IQ(t1.native-country,t2.native-country)&IQ(t1.income,t2.income)&EQ(t1.relationship,t2.relationship)',
#             # 't1&t2&EQ(t1.education,t2.education)&EQ(t1.occupation,t2.occupation)&IQ(t1.race,t2.race)&IQ(t1.workclass,t2.workclass)',
#             # 't1&t2&EQ(t1.age,t2.age)&IQ(t1.sex,t2.sex)&EQ(t1.relationship,t2.relationship)',
#             # 't1&t2&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.native-country,t2.native-country)&IQ(t1.income,t2.income)&EQ(t1.relationship,t2.relationship)&IQ(t1.workclass,t2.workclass)'
#             # ]
#             rc = RepairConfig(strategy=v, complaints=complaints, monitor=FixMonitor(rule_set_size=20), acc_threshold=0.8, runtime=0)
#             start = time.time()
#             bkeepdict = fix_rules(repair_config=rc, original_rules=test_rules, conn=conn, table_name=table_name)
#             end = time.time()
#             new_rules = []

#             for k in bkeepdict:
#                 new_rules.extend(bkeepdict[k]['fixed_treerule_text'])

#             rc.runtime=end-start

#             timestr = time.strftime("%Y%m%d-%H%M%S")

#             with open(f"{input_dc_dir}{timestr}", 'w') as f:
#                 for line in new_rules:
#                     f.write(f"{line}\n")
                    

#             main(table_name=table_name, csv_dir=input_csv_dir, 
#                 csv_file=input_csv_file, dc_dir=input_dc_dir, dc_file=timestr, gt_dir=ground_truth_dir, 
#                 gt_file=ground_truth_file, initial_training=True)

#             new_dirties, new_dc, new_ddw, new_ddc, new_cc, new_cd, new_concated_df = get_holoclean_results(table_name, conn, cols)
#             logger.debug(f"after fixes, the accuracy of holoclean is {(len(new_ddc)+len(new_cc))/(len(dirties)+len(dc)+len(cc)+len(cd))}")

#             fixed_dirty_cnt, fixed_clean_cnt = 0, 0
            

#             for c in clean_input:
#                 if c not in new_dirties:
#                     fixed_clean_cnt+=1

#             for d in dirty_input:
#                 if d in new_ddc:
#                     fixed_dirty_cnt+=1

#             logger.debug(f"after the fix the original dirty is fixed {fixed_dirty_cnt}/{len(dirty_input)} = {fixed_dirty_cnt/len(dirty_input)}")
#             logger.debug(f"after the fix the original clean is fixed {fixed_clean_cnt}/{len(clean_input)} = {fixed_clean_cnt/len(clean_input)}")

#             fix_result = {"dirty_fix_rate": fixed_dirty_cnt/len(dirty_input), "clean_preserving_rate": fixed_clean_cnt/len(clean_input)}
#             result_dict = print_fix_book_keeping_stats(rc, bkeepdict, fix_result)


#             results.append(result_dict)

#     results_df = pd.DataFrame.from_dict(results)
#     results_df.to_csv('results_overall.csv', index=False)




# accuracy over complaint/user input: out of total expected result, how many actually got it right
# record the breakdown of sum I(ddc->ddcc) + I(dc->c) (aslo seperately)
# side record over the all data. (all combinations)
# report number of tuple violations














