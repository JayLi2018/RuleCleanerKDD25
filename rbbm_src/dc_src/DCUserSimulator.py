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

# dc_tuple_violation_template_targeted_t1=\
# Template("SELECT  t1.id as t1_tid FROM $table t1 where EXISTS (select 1 from $table t2 WHERE $dc_desc)")
# dc_tuple_violation_template_targeted_t2=\
# Template("SELECT  t2.id as t2_tid FROM $table t2 where EXISTS (select 1 from $table t1 WHERE $dc_desc)")

dc_tuple_violation_template_targeted_t1=\
Template("SELECT DISTINCT $t1_desc,$t2_desc FROM $table t1, $table t2 WHERE $dc_desc")
dc_tuple_violation_template_targeted_t2=\
Template("SELECT DISTINCT $t1_desc,$t2_desc FROM $table t1, $table t2 WHERE $dc_desc")

class DCUser:

	def __init__(self, conn, tablename):
		self.conn = conn
		self.tablename = tablename

	def select_dc(self, violation_threshold, dc_file_dir, predicate_max_threshold, predicate_min_threshold):
		rule_texts = []

		cur = self.conn.cursor()
		cur.execute(f"select count(*) from {self.tablename}")
		db_size = int(cur.fetchone()[0])
		print(f"db_size: {db_size}")
		cols_q = f"select * from {tablename} limit 1"
		cols=list(pd.read_sql(cols_q, conn))

		total_num_rules = 0

		with open(dc_file_dir, "r") as file:
			for line in file:
				rule=line.strip('\n')
				if(re.search(r'\.id', rule)):
					continue
				total_num_rules+=1
				predicates = rule.split('&')
				pred_cnt = len(predicates[2:])
				if(pred_cnt>predicate_max_threshold or pred_cnt<predicate_min_threshold):
					continue
				rule_texts.append(rule)

		l_after_filter = len(rule_texts)

		print(f'we had {total_num_rules}, after predicate filter, we have {len(rule_texts)} rules')
		rules_with_stats={r:set() for r in rule_texts}
		i = 1
		for r in rule_texts:
			print(f"on {i}/{l_after_filter}... {r}")
			res = find_tuples_in_violation(conn, r, self.tablename, cols)
			for k in res:
				for pair in res[k]:
					# print(pair)
					rules_with_stats[r].add(pair['t1__tid_'])
					rules_with_stats[r].add(pair['t2__tid_'])
			print('\n')
			i+=1
		res = []

		for k,v in rules_with_stats.items():
			if(len(v)/db_size<=violation_threshold):
				res.append(k)

		return res, rules_with_stats


def find_tuples_in_violation(conn, dc_text, target_table, cols, targeted=False):
    if(non_symetric_op.search(dc_text)):
        q1= construct_query_for_violation('t1', dc_text, target_table, targeted, cols)
        q2 = construct_query_for_violation('t2', dc_text, target_table, targeted, cols)
        res = {'t1': pd.read_sql(q1, conn).to_dict('records'), 
        't2': pd.read_sql(q2, conn).to_dict('records')}
    else:
        q1= construct_query_for_violation('t1', dc_text, target_table, targeted, cols)
        # print(q1)
        res = {'t1':pd.read_sql(q1, conn).to_dict('records')}
        # print(res)
    return res


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
        r_q  = template.substitute(table=target_table, dc_desc=parse_rule_to_where_clause(dc_text),
                                           tuple_desc=constants_clause,
                                           t1_desc=t1_cols, t2_desc=t2_cols,
                                           )

    else:
    	cond = parse_rule_to_where_clause(dc_text)
    	cond+=f" AND t1._tid_!=t2._tid_"
    	r_q  = template.substitute(t1_desc=t1_cols, t2_desc=t2_cols, table=target_table, dc_desc=cond)

    return r_q



if __name__ == '__main__':
	conn=psycopg2.connect('dbname=cr user=postgres')

	# tablename='tax_sample'
	# dc_file = '/home/perm/chenjie/RBBM/rbbm_src/dc_src/tax_precise_dcs.txt'

	# tablename='adult_sample'
	# dc_file = '/home/opc/chenjie/RBBM/rbbm_src/dc_src/adult_dcs.txt'

	# tablename='airport_sample'
	# dc_file = '/home/opc/chenjie/RBBM/rbbm_src/dc_src/airport_dcs.txt'

	tablename='hospital_sample'
	dc_file = '/home/opc/chenjie/RBBM/rbbm_src/dc_src/hospital_dcs.txt'

	# tablename='tax'
	# dc_file = '/home/opc/chenjie/RBBM/rbbm_src/dc_src/tax_dcs.txt'

	du = DCUser(conn=conn, tablename=tablename)
	res, res_dict = du.select_dc(violation_threshold=0.2, dc_file_dir=dc_file, predicate_max_threshold=3, predicate_min_threshold=3)
	print(res)
	for k,v in res_dict.items():
		print(f"{k}: {len(v)}")