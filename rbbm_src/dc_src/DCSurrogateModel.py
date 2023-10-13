import pandas as pd
import psycopg2
from rbbm_src.dc_src.DCRepair import (eq_op, 
 non_symetric_op,
#  dc_tuple_violation_template_targeted_t1,
#  dc_tuple_violation_template_targeted_t2
)
import re
from rbbm_src.dc_src.src.classes import parse_rule_to_where_clause
from string import Template

def construct_query_for_violation(role, dc_text, target_table, targeted, cols): 
    predicates = dc_text.split('&')
#     clause = parse_rule_to_where_clause(dc_text)
    constants=[]
    # print(f"t_interest:{t_interest}")
    need_tid=True 
    # if the constraint only has equals, we need to add an artificial
    # key (_tid_) to differentiate tuples in violation with the tuple it
    # self
    for pred in predicates[2:]:
        if(not eq_op.search(pred)):
            need_tid=False
        attr = re.search(r't[1|2]\.([-\w]+)', pred).group(1).lower()
        # print(attr)
#         constants.append(f'{role}.{attr}=\'{t_interest[attr]}\'')
    constants_clause = ' AND '.join(constants)
    # print(f"dc_text:{dc_text}")
    t1_cols=', '.join([f't1.{x} as t1_{x}' for x in cols])
    t2_cols=', '.join([f't2.{x} as t2_{x}' for x in cols])
    
    if(role=='t1'):
        template=dc_tuple_violation_template_targeted_t1
    else:
        template=dc_tuple_violation_template_targeted_t2
    if(targeted):
        r_q  = template.substitute(t1_desc=t1_cols, t2_desc=t2_cols,
                                   table=target_table, dc_desc=parse_rule_to_where_clause(dc_text),
                                           tuple_desc=constants_clause)

    else:
        r_q  = template.substitute(t1_desc=t1_cols, t2_desc=t2_cols,
                                   table=target_table, dc_desc=parse_rule_to_where_clause(dc_text))

        r_q+=f" AND t1.id!=t2.id"
#     print(r_q)

    return r_q

def find_tuples_in_violation(conn, dc_text, target_table, cols, targeted=False):
    if(non_symetric_op.search(dc_text)):
        q1= construct_query_for_violation('t1', dc_text, target_table, targeted, cols)
        q2 = construct_query_for_violation('t2', dc_text, target_table, targeted, cols)
        res = {'t1': pd.read_sql(q1, conn).to_dict('records'), 
        't2': pd.read_sql(q2, conn).to_dict('records')}
    else:
        q1= construct_query_for_violation('t1', dc_text, target_table, targeted, cols)
        print(q1)
        res = {'t1':pd.read_sql(q1, conn).to_dict('records')}
        # print(res)
    return res 

class DCSurrogateModel:
	def __init__(self, conn, dcs):
		self.conn = conn
		self.dcs = dcs 

	def delete_tuples(self, table_name):
		# return the tuples that needs to be deleted
		# in order to hold for the dcs
		for r in self.dcs:
		    res = find_tuples_in_violation(self.conn, r, table_name, cols)
		    print(f"rule: {r}")
		    for k in res:
		        for pair in res[k]:
		#             if(pair['t1_is_dirty']!=pair['t2_is_dirty']):
		            # dirty_tuples.add(pair['t1_id'])
		            # dirty_tuples.add(pair['t2_id'])
		        # len_cc = len([x for x in res[k] if (x['t1_is_dirty']==False and x['t2_is_dirty']==False)])
		        # len_non_cc = len([x for x in res[k] if not (x['t1_is_dirty']==False and x['t2_is_dirty']==False)])
		        # len_dd = len([x for x in res[k] if (x['t1_is_dirty']==True and x['t2_is_dirty']==True)])
		        # len_dc = len([x for x in res[k] if ((x['t1_is_dirty']==True and x['t2_is_dirty']==False) or \
		        #             (x['t1_is_dirty']==False and x['t2_is_dirty']==True))])
		            rules_with_stats[r].add(pair['t1_id'])
		            rules_with_stats[r].add(pair['t2_id'])
		        # rules_with_stats[r].add(pair['t1_id'])
		        # print(f"number of ccs:{len_cc}, number of non ccs: {len_non_cc}, number of dds: {len_dd}, number of dcs:{len_dc}")
		#     break
		res = set()
		for k, v in rules_with_stats.items():
			for val in v:
				res.add(val)

		return res

