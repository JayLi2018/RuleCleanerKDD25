import re
from typing import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from rbbm_src.labelling_func_src.src.TreeRules import *
from rbbm_src.dc_src.src.classes import parse_rule_to_where_clause, dc_violation_template
import psycopg2
from string import Template
from dataclasses import dataclass
from typing import *
from collections import deque
from itertools import product
import copy
import time
from rbbm_src.holoclean.examples.holoclean_repair import main

dc_tuple_violation_template_targeted_t1=Template("SELECT DISTINCT t2.* FROM $table t1, $table t2 WHERE $dc_desc AND $tuple_desc;")
dc_tuple_violation_template_targeted_t2=Template("SELECT DISTINCT t1.* FROM $table t1, $table t2 WHERE $dc_desc AND $tuple_desc;")
dc_tuple_violation_template=Template("SELECT DISTINCT t2.* FROM $table t1, $table t2 WHERE $dc_desc;")

ops = re.compile(r'IQ|EQ|LTE|GTE|GT|LT')
non_symetric_op = re.compile(r'LTE|GTE|GT|LT')
const_detect = re.compile(r'([\'|\"])')


@dataclass
class FixMonitor:
    """
    object that tracks the stats needed during the fix

    """
    counter:int=0 # count how many rules have been fixed 
    lambda_val: float=0.2
    # threshold predefined to retrain using the current fixes to see if it 
    # has already met the requirement
    rule_set_size: int=0 # total number of rules being used in the model
    overall_fixed_count: int=0 # overall total number of rules fixed so far


@dataclass
class RepairConfig:
    """
    object that contains the information 
    needed to do the repair
    """

    strategy:str # 'naive', 'information gain', 'optimal' 
    complaints:List[dict]
    monitor: FixMonitor
    acc_threshold: float 
    runtime:float
    # early stop threshold, i.e., if after fixing some rules the accuracy of 
    # the complaint set is above this threshold, we stop

    # tid:int=-1 # DC only


def get_operator(predicate):
    predicate_sign=ops.search(predicate).group()
    # print(predicate_sign)
    if(predicate_sign=='EQ'):
        sign='=='
    elif(predicate_sign=='IQ'):
        sign='!='
    elif(predicate_sign=='LTE'):
        sign='<='
    elif(predicate_sign=='LT'):
        sign='<'
    elif(predicate_sign=='GTE'):
        sign='>='
    elif(predicate_sign=='GT'):
        sign='>'
    else:
        print("non recognizable sign")
        exit()
    return sign

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
    parent.right=last_right
    last_right.parent=parent
    tree_size+=1
    # print(f"tree size: {tree_size}, cur_number={cur_number}")
    return TreeRule(rtype='dc', root=root_node, size=tree_size)

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
#     clause = parse_rule_to_where_clause(dc_text)
    constants=[]

    for pred in predicates[2:]:
        attr = re.search(r't[1|2]\.([-\w]+)', pred).group(1)
        constants.append(f'{role}.\"{attr}\"=\'{t_interest[attr]}\'')
    constants_clause = ' AND '.join(constants)
    # print(f"dc_text:{dc_text}")
    if(role=='t1'):
        template=dc_tuple_violation_template_targeted_t1
    else:
        template=dc_tuple_violation_template_targeted_t2
    if(targeted):
        r_q  = template.substitute(table=target_table, dc_desc=parse_rule_to_where_clause(dc_text),
                                           tuple_desc=constants_clause)
        # print(r_q)
        return r_q
    else:
        r_q  = template.substitute(table=target_table, dc_desc=parse_rule_to_where_clause(dc_text))
        return r_q


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
#         cur_node_parent=cur_node
#         cur_node=cur_node.right
        
    available_attrs=list(set(set(list(t_interest)).difference(set(unusable_attrs))))
    # print(available_attrs)
    if(not available_attrs):
        return None
    repaired_tree_rule=gen_repaired_tree_rule(t_interest, desired_label, t_in_violation, 
                                              available_attrs[0], tree_rule, role)
#         if(gen_repair_predicate(t_interest, t_in_violation, attr))
    return repaired_tree_rule


# def fix_rules(original_rules, complaint_tuples, conn):
#     rules = original_rules
#     for r in rules:
#     for c in complaint_tuples:
#         new_rules = []
#         i=0
#         for r in rules:
#             # if(r[1]!=-1):
#             treerule = parse_dc_to_tree_rule(r)
#             print(treerule)
#             tuples_inviolation=find_tuples_in_violation(c, conn, r, 'adult500', targeted=True)
#             print('tuples_inviolation:')
#             print(tuples_inviolation)
#             # check if we can pass by this rule
#             if(all([not v for k,v in tuples_inviolation.items()])):
#                 print('complaint not in violation, pass it by')
#                 new_rules.append(r)
#             else:
#                 for k in tuples_inviolation:
#                     if(tuples_inviolation[k]):
#                         # if there are some tuples in violationW 
#                         print(f"need to fix rule[{i}]")
#                         print(f'before fixing {c}:')
#                         print(treerule.serialize()[0])
#                     # tuples_inviolation = find_tuples_in_violation(complaint_tuple, conn, r, 'adult500')
#                         for t in tuples_inviolation[k]:
#                             fix_violation(c, CLEAN, t, treerule, k)
#                             # print(treerule)
#                             tuples_inviolation_after_fix = find_tuples_in_violation(c, conn, treerule.serialize()[0], 'adult500')
#                             if(all([not v for k,v in tuples_inviolation_after_fix.items()])):
#                                 break
#                         print(f'after fixing {c}:')
#                         print(treerule.serialize()[0])
#                         print('\n')
#                         new_rules.append(treerule.serialize()[0])
#             i+=1
#         rules=new_rules
#         # print(rules)
#         # print(treerule)
#         # print('---------------------------------------\n')
#         # fix_violation(complaint_tuple, CLEAN, tuples_inviolation[1], treerule)
#     print(rules)

def construct_domain_dict(connection, table_name):
    res = {}
    cols = list(pd.read_sql(f'select * from "{table_name}" limit 1', connection))
    cols.remove("_tid_")

    cur=connection.cursor()
    for c in cols:
        cur.execute(f'select distinct "{c}" from {table_name}')
        res[c]=set([x[0] for x in cur.fetchall()])

    return res

def fix_rules(repair_config, original_rules, conn):
    rules = original_rules
    all_fixed_rules = []
    cur_fixed_rules = []
    domain_value_dict = construct_domain_dict(conn, table_name='adult500')
    fix_book_keeping_dict = {k:{} for k in original_rules}
    # print(domain_value_dict)
    for r in rules:
        # print("before fixing the rule, the rule is")
        # print(r)
        treerule = parse_dc_to_tree_rule(r)
        fix_book_keeping_dict[r]['pre_fix_size']=treerule.size
        leaf_nodes = []
        for c in repair_config.complaints:
            leaf_nodes_with_complaints = populate_violations(treerule, conn, r, c)
            for ln in leaf_nodes_with_complaints:
                if(ln not in leaf_nodes):
                    # if node is already in leaf nodes, dont
                    # need to add it again
                    leaf_nodes.append(ln)
        # print(leaf_nodes)
        if(leaf_nodes):
            # its possible for certain rule we dont have any violations
            # print("the TreeRule we wanted to fix")
            # print(treerule)
            # print("the leaf nodes")
            # print(leaf_nodes)
            fixed_treerule = fix_violations(treerule, repair_config, leaf_nodes, domain_value_dict)
            # print(fixed_treerule)
            fix_book_keeping_dict[r]['after_fix_size']=fixed_treerule.size
            fixed_treerule_text = treerule.serialize()
            fix_book_keeping_dict[r]['fixed_treerule_text']=fixed_treerule_text
        else:
            fix_book_keeping_dict[r]['after_fix_size']=treerule.size
            fixed_treerule_text = treerule.serialize()
            fix_book_keeping_dict[r]['fixed_treerule_text']=fixed_treerule_text

        # print(fixed_treerule)
        # print(fixed_treerule_text)
        # print("fixed result leaf nodes")
    #     cur_fixed_rules.append(fixed_treerule_text)
    #     repair_config.monitor.counter+=1
    #     repair_config.monitor.overall_fixed_count+=1
    #     if((repair_config.monitor.counter / repair_config.monitor.rule_set_size)>=repair_config.monitor.lambda_val):
    #         repair_config.monitor.counter=0
    #         new_rules=cur_fixed_rules[:]
    #         new_rules.extend(original_rules[repair_config.monitor.overall_fixed_count:])
    #         accuracy = retrain_and_get_accuracy(repair_config, new_rules)
    #         all_fixed_rules.extend(cur_fixed_rules)
    #         cur_fixed_rules = []
    #         if(accuracy>=repair_config.acc_threshold):
    #             break
    # all_fixed_rules.extend(original_rules[repair_config.monitor.overall_fixed_count:])
    # print(fix_book_keeping_dict)
    return fix_book_keeping_dict

def print_fix_book_keeping_stats(config, bkeepdict):
    sizes_diff = []
    # print(bkeepdict)
    for k,v in bkeepdict.items():
        sizes_diff.append(v['after_fix_size']-v['pre_fix_size'])
    print('**************************\n')
    print(f"strategy: {config.strategy}, complaint_size={len(config.complaints)}, runtime: {config.runtime}")
    print(sizes_diff)
    print(f"average size increase: {sum(sizes_diff)/len(sizes_diff)}")
    print('**************************\n')

    return {'strategy':config.strategy, 'complaint_size': len(config.complaints), 'runtime': config.runtime, 'avg_size_increase': {sum(sizes_diff)/len(sizes_diff)}}

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
    # print(f"finding fix for {clean_pair} and {dirty_pair}")
    for k in domain_value_dict:
        # attribute
        # equal_assign_sign=None 
        # not_equal_assign_sign=None
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

            if((clean_pair['t1'][k]!=clean_pair['t2'][k]) and \
               (dirty_pair['t1'][k]==dirty_pair['t2'][k])):
                cand = ('t1', k, 't2', k, not_equal_assign_sign)

            if(cand and cand not in used_predicates):
                if(not all_possible):
                    return cand
                else:
                    res.append(cand)
                    # print(len(res))

        for k in domain_value_dict:
            for v in domain_value_dict[k]:
                if((clean_pair['t1'][k]==v) and (dirty_pair['t1'][k]!=v)):
                    cand = ('t1', k, v, equal_assign_sign)
                if((clean_pair['t1'][k]!=v) and (dirty_pair['t1'][k]==v)):
                    cand = ('t1', k, v, not_equal_assign_sign)
                if(cand and cand not in used_predicates):
                    if(not all_possible):
                        return cand
                    else:
                        res.append(cand)
                        # print(len(res))
                if((clean_pair['t2'][k]==v) and (dirty_pair['t2'][k]!=v)):
                    cand = ('t2', k, v, equal_assign_sign)

                if((clean_pair['t2'][k]!=v) and (dirty_pair['t2'][k]==v)):
                    cand = ('t2', k, v, not_equal_assign_sign)

                if(cand and cand not in used_predicates):
                    if(not all_possible):
                        return cand
                    else:
                        res.append(cand)
                        # print(len(res))
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
    # print("node:")
    # print(node)
    # print("pairs:")
    # print(node.pairs)
    sign=None
    if(len(the_fix)==4):
        _, _, _, sign = the_fix
    elif(len(the_fix)==5):
        _, _, _, _, sign = the_fix
    # new_pred = convert_tuple_fix_to_pred(the_fix)
    # new_predicate_node = PredicateNode(pred=DCAttrPredicate(pred=new_pred, operator=sign))
    # new_predicate_node.left= LabelNode(label=CLEAN, pairs={DIRTY:[], CLEAN:[]}, used_predicates=set([]))
    # new_predicate_node.right=LabelNode(label=DIRTY, pairs={DIRTY:[], CLEAN:[]}, used_predicates=set([]))

    # if(node.label==CLEAN):
    #     node.parent.left=new_predicate_node
    # else:
    #     node.parent.right=new_predicate_node
    
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
                # print(p)
                # print(f"p['{role1}']['{attr1}']{sign}p['{role2}']['{attr2}']")
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

    # print(f"gini_impurity for {the_fix} using {the_fix}: {gini_impurity}, reverse:{reverse_condition}\n")
    
    return gini_impurity, reverse_condition

def redistribute_after_fix(tree_rule, node, the_fix, reverse=False):
    # there are some possible "side effects" after repair for a pair of violations
    # which is solving one pair can simutaneously fix some other pairs so we need 
    # to redistribute the pairs in newly added nodes if possible
    sign=None
    cur_number=tree_rule.size
    if(len(the_fix)==4):
        _, _, _, sign = the_fix
    elif(len(the_fix)==5):
        _, _, _, _, sign = the_fix
    new_pred, modified_fix = convert_tuple_fix_to_pred(the_fix, reverse)
    if(len(the_fix)==4):
        new_predicate_node = PredicateNode(number=cur_number, pred=DCConstPredicate(pred=new_pred, operator=sign))
    elif(len(the_fix)==5):
        new_predicate_node = PredicateNode(number=cur_number, pred=DCAttrPredicate(pred=new_pred, operator=sign))
    
    cur_number+=1
    new_predicate_node.left= LabelNode(number=cur_number, label=CLEAN, pairs={DIRTY:[], CLEAN:[]}, used_predicates=set([]))
    cur_number+=1
    new_predicate_node.right=LabelNode(number=cur_number, label=DIRTY, pairs={DIRTY:[], CLEAN:[]}, used_predicates=set([]))
    new_predicate_node.left.parent= new_predicate_node
    new_predicate_node.right.parent= new_predicate_node

    # print(node)
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
                else:
                    new_predicate_node.left.pairs[p['expected_label']].append(p)

    elif(len(modified_fix)==5):
        role1, attr1, role2, attr2, sign = modified_fix
        for k in [CLEAN, DIRTY]:
            for p in node.pairs[k]:
                # print(p)
                # print(f"p['{role1}']['{attr1}']{sign}p['{role2}']['{attr2}']")
                if(eval(f"p['{role1}']['{attr1}']{sign}p['{role2}']['{attr2}']")):
                    new_predicate_node.right.pairs[p['expected_label']].append(p)
                else:
                    new_predicate_node.left.pairs[p['expected_label']].append(p)

    node.pairs={CLEAN:{}, DIRTY:{}}

    return new_predicate_node

def check_tree_purity(tree_rule):
    # print("checking purity...")
    # print(tree_rule)
    queue = deque([tree_rule.root])
    leaf_nodes = []
    while(queue):
        # print(queue)
        cur_node = queue.popleft()
        if(isinstance(cur_node,LabelNode)):
            leaf_nodes.append(cur_node)
        if(cur_node.left):
            queue.append(cur_node.left)
        if(cur_node.right):
            queue.append(cur_node.right)
        # print(cur_node.number)

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

def fix_violations(treerule, repair_config, leaf_nodes, domain_value_dict):
    if(repair_config.strategy=='naive'):
        # initialize the queue to work with
        queue = deque([])
        for ln in leaf_nodes:
            queue.append(ln)
        # print(queue)
        while(queue):
            node = queue.popleft()
            # print(node.pairs)
            new_parent_node=None
            # need to find a pair of violations that get the different label
            # in order to differentiate them
            if(node.pairs[CLEAN] and node.pairs[DIRTY]):
                the_fix = find_available_repair(node.pairs[CLEAN][0],
                 node.pairs[DIRTY][0], domain_value_dict, node.used_predicates)
                # print("the fix")
                # print(the_fix)
                new_parent_node=redistribute_after_fix(treerule, node, the_fix)
            # handle the left and right child after redistribution
            else:
                if(check_tree_purity(treerule)):
                    # print('its pure already!')
                    return treerule
                else:
                    reverse_node_parent_condition(node)
                    # print('its not pure?')
                    continue

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
                # print(queue)
                treerule.setsize(treerule.size+2)
                # print("adding new predicate, size+2")
                # print('\n')
                # print(f"after fix, treerule size is {treerule.size}")
            # print(f"queue size: {len(queue)}")
        return treerule

    elif(repair_config.strategy=='information gain'):
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
        # print(queue)
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
                        considered_fixes.add(f)
                        if(gini<min_gini):
                            min_gini=gini
                            best_fix=f
                            reverse_condition=reverse_cond
                if(best_fix):
                    new_parent_node=redistribute_after_fix(treerule, node, best_fix, reverse_condition)
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
                # print(f"after fix, treerule size is {treerule.size}")

        return treerule

    elif(repair_config.strategy=='optimal'):
        # 1. create a queue with tree nodes
        # 2. need to deepcopy the tree in order to enumerate all possible trees

        queue = deque([])
        for ln in leaf_nodes:
            queue.append((treerule,ln))

        # print("queue")
        # print(queue)
        while(queue):
            # print(f'len of queue: {len(queue)}')
            prev_tree, node = queue.popleft()
            # new_tree = copy.deepcopy(prev_tree)
            # print("node pairs")
            # print(node.pairs)
            # print(f'clean: {len(node.pairs[CLEAN])}')
            # print(f'dirty: {len(node.pairs[DIRTY])}')
            # print(node.label)
            # print(bool(node.pairs[CLEAN] and node.pairs[DIRTY]))
            if(node.pairs[CLEAN] and node.pairs[DIRTY]):
                # print("we need to fix it!")
                # need to examine all possible pair combinations
                considered_fixes = set()
                # print(list(product(node.pairs[CLEAN], node.pairs[DIRTY])))
                for pair in list(product(node.pairs[CLEAN], node.pairs[DIRTY])):
                    the_fixes = find_available_repair(pair[0],
                     pair[1], domain_value_dict, node.used_predicates,
                     all_possible=True)
                    # print("the_fixes")
                    # print(the_fixes)
                    # print(f"fixes: len = {len(the_fixes)}")
                    for f in the_fixes:
                        new_parent_node=None
                        if(f in considered_fixes):
                            continue
                        considered_fixes.add(f)
                        new_tree = copy.deepcopy(prev_tree)
                        node = locate_node(new_tree, node.number)
                        new_parent_node = redistribute_after_fix(new_tree, node, f)
                        new_tree.setsize(new_tree.size+2)
                        # print("new_parent_node")
                        # print(new_parent_node)
                        # print("new_tree")
                        # print(new_tree)
                        if(check_tree_purity(new_tree)):
                            # print("its pure we are done!")
                            # print("new_tree")
                            # print(new_tree)
                            return new_tree
                        # handle the left and right child after redistribution
                        if(new_parent_node):
                            # print(new_parent_node)
                            # new_tree.setsize(new_tree.size+2)
                            still_inpure=False
                            for k in [CLEAN,DIRTY]:
                                if(still_inpure):
                                    break
                                for p in new_parent_node.left.pairs[k]:
                                    if(p['expected_label']!=new_parent_node.left.label):
                                        new_tree = copy.deepcopy(new_tree)
                                        parent_node = locate_node(new_tree, new_parent_node.number)
                                        # new_parent_node=redistribute_after_fix(new_tree, new_node, f)
                                        queue.append((new_tree, parent_node.left))
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
                                        queue.append((new_tree, parent_node.right))
                                        still_inpure=True
                                        break
            else:
                if(check_tree_purity(prev_tree)):
                    print('its pure already!')
                    return prev_tree
                else:
                    # print("its not pure reverse parent condition")
                    reverse_node_parent_condition(node)
                    if(check_tree_purity(prev_tree)):
                        return prev_tree
                    continue
        return None 
        # list_of_repaired_trees = sorted(list_of_repaired_trees, key=lambda x: x[0].size, reverse=True)
        # return list_of_repaired_trees[0] 


    else:
        print("not a valid repair option")
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
    print('cant find the node!')
    exit()

def populate_violations(tree_rule, conn, rule_text, complaint):
    # given a tree rule and a complaint, populate the complaint and violation tuple pairs
    # to the leaf nodes
    tuples_inviolation=find_tuples_in_violation(complaint['tuple'], conn, rule_text, 'adult500', targeted=True)
    # print(f"tuples_inviolation with {complaint} on rule {rule_text}")
    # print(len(tuples_inviolation))
    pairs = []

    if('t1' in tuples_inviolation):
        for v in tuples_inviolation['t1']:
            pair = {'t1':complaint['tuple'], 't2':v, 'expected_label':complaint['expected_label']}
            pairs.append(pair)
    if('t2' in tuples_inviolation):
        for v in tuples_inviolation['t2']:
            pair = {'t1':v, 't2':complaint['tuple'], 'expected_label':complaint['expected_label']}
            pairs.append(pair)
    # print("total_pairs")
    # print(len(pairs))
    leaf_nodes = []

    for p in pairs:
        leaf_node = tree_rule.evaluate(p)
        leaf_node.pairs[p['expected_label']].append(p)
        if(leaf_node not in leaf_nodes):
            leaf_nodes.append(leaf_node)
    # print(leaf_nodes)
    # print(tree_rule)
    # print('\n')
    return leaf_nodes


def get_holoclean_results(table_name, conn, cols):

    union_sql = f"""
    SELECT 'before_clean' AS type, * from  {table_name}
    union all 
    SELECT 'after_clean' AS type, * from {table_name}_repaired
    order by _tid_
    """
    df_union_before_and_after = pd.read_sql(union_sql, conn) 

    j = 0
    for i in range(0,num_lines-1):
        if(j%100==0):
            print(j)
        j+=1
        df_row = pd.DataFrame(columns=['type']+cols)
        df_row.loc[0,'type'] = 'ground_truth'
        df_row.loc[0,'_tid_'] = i
        q = f"""
        SELECT * FROM {table_name}_clean WHERE _tid_={i} 
        """
        df_for_one_row = pd.read_sql(q,conn)
        for index, row in df_for_one_row.iterrows():
            df_row.loc[0, f"{row['_attribute_']}"] = row['_value_']
        df_union_before_and_after = pd.concat([df_union_before_and_after, df_row])

    # iterate this dataframe to find all wrong predictions and list them to choose
    grouped = df_union_before_and_after.groupby('_tid_')


    # for d,v in repaired_dict.items():
    #     print(f"{d}: {len(repaired_dict[d])}")

    wrong_repairs = set([])
    correct_repairs = set([])
    repairs = set([])
    clean_tuples = set([])
    still_dirty_tuples = set([])
    # iterate this dataframe to find all wrong predictions and list them to choose
    grouped = df_union_before_and_after.groupby('_tid_')
    for name, group in grouped:
        tid = pd.to_numeric(group.iloc[0]['_tid_'], downcast="integer")
        # if(k%100==0):
        #     print(k)
        if(not group[group['type']=='after_clean'][cols].reset_index(drop=True).equals(group[group['type']=='before_clean'][cols].reset_index(drop=True))):
            repairs.add(tid)
        if(not group[group['type']=='after_clean'][cols].reset_index(drop=True).equals(group[group['type']=='before_clean'][cols].reset_index(drop=True)) and \
           not group[group['type']=='after_clean'][cols].reset_index(drop=True).equals(group[group['type']=='ground_truth'][cols].reset_index(drop=True))):
            wrong_repairs.add(tid)
        if(not group[group['type']=='after_clean'][cols].reset_index(drop=True).equals(group[group['type']=='before_clean'][cols].reset_index(drop=True)) and \
            group[group['type']=='after_clean'][cols].reset_index(drop=True).equals(group[group['type']=='ground_truth'][cols].reset_index(drop=True))):
            correct_repairs.add(tid)
        if(group[group['type']=='after_clean'][cols].reset_index(drop=True).equals(group[group['type']=='before_clean'][cols].reset_index(drop=True)) and \
            group[group['type']=='after_clean'][cols].reset_index(drop=True).equals(group[group['type']=='ground_truth'][cols].reset_index(drop=True))):
            clean_tuples.add(tid)
        if(group[group['type']=='after_clean'][cols].reset_index(drop=True).equals(group[group['type']=='before_clean'][cols].reset_index(drop=True)) and \
            not group[group['type']=='after_clean'][cols].reset_index(drop=True).equals(group[group['type']=='ground_truth'][cols].reset_index(drop=True))):
            still_dirty_tuples.add(tid)

    print(f"wrongs: {len(wrong_repairs)}, corrects: {len(correct_repairs)}, total_repairs: {len(repairs)}, clean_tuples: {len(clean_tuples)}, still_dirty_tuples: {len(still_dirty_tuples)}")


if __name__ == '__main__':
    conn=psycopg2.connect('dbname=holo user=postgres')
    cur = conn.cursor()

    input_csv_dir = '/home/opc/chenjie/labelling_explanation/experiments/dc/'
    input_csv_file = 'adult500.csv'
    input_csv_dir = '/home/opc/chenjie/labelling_explanation/experiments/dc/'
    input_dc_dir = '/home/opc/chenjie/labelling_explanation/experiments/dc/'
    input_dc_file = 'dc_sample_20'
    ground_truth_dir = '/home/opc/chenjie/labelling_explanation/experiments/dc/'
    ground_truth_file = 'adult500_clean.csv'
    table_name = 'adult500'

    conn.autocommit=True
    cur=conn.cursor()
    input_file=input_csv_dir+input_csv_file
    table_name=input_csv_file.split('.')[0]
    cols=None
    num_lines=0
    try:
        with open(input_file) as f:
            first_line = f.readline()
            num_lines = sum(1 for line in f)
    except Exception as e:
        print(f'cant read file {input_file}')
        exit()
    else:
        cols=first_line.split(',')

    # drop preexisted repaired records 
    select_old_repairs_q = f"""
    SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME LIKE '{table_name}_repaired_%' AND TABLE_TYPE = 'BASE TABLE'
    """
    cur.execute(select_old_repairs_q)

    for records in cur.fetchall():
        drop_q = f"drop table if exists {records[0]}"
        cur.execute(drop_q)

    main(table_name=table_name, csv_dir=input_csv_dir, 
        csv_file=input_csv_file, dc_dir=input_dc_dir, dc_file=input_dc_file, gt_dir=ground_truth_dir, 
        gt_file=ground_truth_file, initial_training=True)


    get_holoclean_results(table_name, conn, cols)


    # complaint_size_for_each_label = [1,2,3,4,5]
    # versions = ['optimal', 'naive', 'information gain']
    # results = []
    # for c in complaint_size_for_each_label:
    #     for v in versions:
    #         print(f" running complaint: size: {c*2}, version: {v}....")
    #         # print(f"on complaint size {c}")
    #         complaints_df = pd.read_sql(f'select * from adult500 order by _tid_ limit {c*2}', conn)
    #         expected_dirty_tuples = complaints_df.iloc[:c].to_dict('records')
    #         expected_clean_tuples = complaints_df.iloc[c:].to_dict('records')
    #         expected_dirty_dicts = [{'tuple':c, 'expected_label':DIRTY} for c in expected_dirty_tuples]
    #         expected_clean_dicts = [{'tuple':c, 'expected_label':CLEAN} for c in expected_clean_tuples]
    #         complaints = []
    #         complaints.extend(expected_dirty_dicts)
    #         complaints.extend(expected_clean_dicts)
    #         # print(complaints)
    #         test_rules=[
    #         't1&t2&EQ(t1.occupation,t2.occupation)&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.race,t2.race)&IQ(t1.sex,t2.sex)',
    #         't1&t2&EQ(t1.marital-status,t2.marital-status)&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.relationship,t2.relationship)&IQ(t1.income,t2.income)',
    #         't1&t2&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.income,t2.income)&EQ(t1.relationship,t2.relationship)&IQ(t1.marital-status,t2.marital-status)',
    #         't1&t2&EQ(t1.education,t2.education)&IQ(t1.sex,t2.sex)&IQ(t1.native-country,t2.native-country)&EQ(t1.relationship,t2.relationship)',
    #         't1&t2&EQ(t1.education,t2.education)&EQ(t1.marital-status,t2.marital-status)&IQ(t1.relationship,t2.relationship)&EQ(t1.sex,t2.sex)&IQ(t1.workclass,t2.workclass)',
    #         't1&t2&EQ(t1.marital-status,t2.marital-status)&EQ(t1.age,t2.age)&IQ(t1.race,t2.race)',
    #         't1&t2&EQ(t1.education,t2.education)&EQ(t1.occupation,t2.occupation)&IQ(t1.race,t2.race)&IQ(t1.income,t2.income)',
    #         't1&t2&EQ(t1.education,t2.education)&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.race,t2.race)&IQ(t1.workclass,t2.workclass)',
    #         't1&t2&EQ(t1.occupation,t2.occupation)&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.income,t2.income)&EQ(t1.relationship,t2.relationship)',
    #         't1&t2&EQ(t1.education,t2.education)&EQ(t1.marital-status,t2.marital-status)&EQ(t1.workclass,t2.workclass)&IQ(t1.native-country,t2.native-country)',
    #         't1&t2&EQ(t1.education,t2.education)&IQ(t1.native-country,t2.native-country)&EQ(t1.relationship,t2.relationship)&IQ(t1.workclass,t2.workclass)',
    #         't1&t2&EQ(t1.occupation,t2.occupation)&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.race,t2.race)&IQ(t1.workclass,t2.workclass)',
    #         't1&t2&EQ(t1.marital-status,t2.marital-status)&EQ(t1.occupation,t2.occupation)&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.race,t2.race)',
    #         't1&t2&EQ(t1.education,t2.education)&EQ(t1.occupation,t2.occupation)&EQ(t1.age,t2.age)&EQ(t1.relationship,t2.relationship)',
    #         't1&t2&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.sex,t2.sex)&IQ(t1.native-country,t2.native-country)&EQ(t1.relationship,t2.relationship)',
    #         't1&t2&EQ(t1.education,t2.education)&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.race,t2.race)&IQ(t1.income,t2.income)',
    #         't1&t2&IQ(t1.age,t2.age)&IQ(t1.race,t2.race)&IQ(t1.native-country,t2.native-country)&IQ(t1.income,t2.income)&EQ(t1.relationship,t2.relationship)',
    #         't1&t2&EQ(t1.education,t2.education)&EQ(t1.occupation,t2.occupation)&IQ(t1.race,t2.race)&IQ(t1.workclass,t2.workclass)',
    #         't1&t2&EQ(t1.age,t2.age)&IQ(t1.sex,t2.sex)&EQ(t1.relationship,t2.relationship)',
    #         't1&t2&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.native-country,t2.native-country)&IQ(t1.income,t2.income)&EQ(t1.relationship,t2.relationship)&IQ(t1.workclass,t2.workclass)'
    #         ]
    #         rc = RepairConfig(strategy=v, complaints=complaints, monitor=FixMonitor(rule_set_size=20), acc_threshold=0.8, runtime=0)
    #         start = time.time()
    #         bkeepdict = fix_rules(repair_config=rc, original_rules=test_rules, conn=conn)
    #         end = time.time()
    #         rc.runtime=end-start
    #         result_dict = print_fix_book_keeping_stats(rc, bkeepdict)
    #         results.append(result_dict)
    # results_df = pd.DataFrame.from_dict(results)
    # results_df.to_csv('results.csv', index=False)