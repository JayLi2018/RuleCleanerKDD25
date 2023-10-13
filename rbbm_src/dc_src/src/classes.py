from dataclasses import dataclass
import networkx as nx
import re
from itertools import combinations
from collections import defaultdict
from string import Template
import re
import psycopg2
import pandas as pd
import scipy.cluster.hierarchy as shc 
from scipy.spatial.distance import squareform, pdist

dc_violation_template=Template("SELECT t2.* FROM $table t1, $table t2 WHERE $dc_desc;")

ops = re.compile(r'IQ|EQ|LTE|GTE|GT|LT')
eq_op = re.compile(r'EQ')
non_symetric_op = re.compile(r'LTE|GTE|GT|LT')
const_detect = re.compile(r'([\'|\"])')


def get_operator(predicate):
    predicate_sign=ops.search(predicate).group()
    # print(predicate_sign)
    if(predicate_sign=='EQ'):
        sign='='
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

def parse_rule_to_where_clause(rule):
	res = []
	for xl in rule.split('&'):
		if(ops.search(xl)):
			sign=get_operator(xl)
			bracket_content = re.findall(r'\((.*)\)', xl)[0]
			# res.append(sign.join(re.sub(r'(t[1|2]\.)([-\w]+)', r'\1"\2"', bracket_content).split(',')))
			res.append(sign.join(bracket_content.split(',')))

	# t1&t2&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.native-country,t2.native-country)&IQ(t1.income,t2.income)&EQ(t1.relationship,t2.relationship)&IQ(t1.workclass,t2.workclass)&IQ(t1.income,'<=50k')

	return ' AND '.join(res)


@dataclass
class Complaint:
	"""
	complaint_type: 
		DC or LF
	complaint_instance: 
		if DC then should be cell attribute name
		if LF then should be sentence text
	"""

	complain_type:str 
	attr_name:str='foo' # DC only
	tid:int=-1 # DC only
        
class RulePruner:
    """
    Given a set of rules (DCs or LFs), and a usercomplaint
    prune the input rules based on the following principle:
        1.DCs: produce a graph where each edge represents
        a pair of attributes appearing in the DC, prune out
        DCs that are not connected to the attriute which is
        the attribute that the complaint comes from
        2.LFs: prune out LFs that gives the user complaint
        trivial label (ABSTAINs) 
    """

    def __init__(self, ):
        pass 

    def prune_and_return(self, complaint, rules):
        if(complaint.complain_type=='DC'):
            # rule_graph = hnx.Hypergraph()
            # undirected graph
            rule_dict = {}
            res=[]
            nodes=set([])
            i=0
            for r in rules:
                rule_nodes=set(re.findall(r't[12]\.([-\w]+)', r))
            #         print(rule_nodes)
                rule_dict[r]=rule_nodes
            for k,v in rule_dict.items():
                if(complaint.attr_name in v):
                    res.append(k)
                    nodes=nodes.union(v)
            return len(res),res        
# 			rule_graph = nx.Graph()
# 			# undirected graph
# 			rule_dict = {}
# 			for r in rules:
# 				rule_nodes=list(set(re.findall(r't[12]\.(\w+)', r)))
# 				rule_edges = list(combinations(rule_nodes, 2))
# 				rule_graph.add_edges_from(rule_edges)
# 				rule_dict[r]=rule_nodes
# 			start_node=complaint.attr_name
# 			useful_nodes=[start_node] + [v for u, v in nx.bfs_edges(rule_graph, start_node)]
# 			set_useful_nodes=set(useful_nodes)
# 			print(f"set_useful_nodes: {set_useful_nodes}")
# 			useless_rules = [k for (k,v) in rule_dict.items() if not set(v).intersection(set_useful_nodes)]
# 			print(f'useless rules: {useless_rules}')
# 			return rule_graph, [k for (k,v) in rule_dict.items() if set(v).intersection(set_useful_nodes)]
# 		else:
# 			pass

class DataPruner:
	"""
	Given a set of rules, prune data(tuples/sentences) based on if 
	the data points have any effect on the rules
	"""
	def __init__(self,):
		pass 

	def dc_prune_and_return(self, db_conn, target_table, pruned_rules):
		drop_if_exist_q = f"drop table if exists {target_table}_pruned"
		drop_if_exist_intermediate_q = f"drop table if exists {target_table}_pruned_intermediate"
		create_q=f"create table {target_table}_pruned as select * from {target_table} limit 0"
		create_intermediate_q=f"create table {target_table}_pruned_intermediate as select * from {target_table} limit 0"
		cur = db_conn.cursor()
		cur.execute(drop_if_exist_q)
		cur.execute(drop_if_exist_intermediate_q)
		cur.execute(create_q)
		cur.execute(create_intermediate_q)
		for r in pruned_rules:
			r_q  = dc_violation_template.substitute(table=target_table, dc_desc=parse_rule_to_where_clause(r))
			cur.execute(f"INSERT INTO {target_table}_pruned_intermediate {r_q}")
			print(f"INSERT INTO {target_table}_pruned_intermediate {r_q}")
		cur.execute(f"SELECT COLUMN_NAME from information_schema.columns WHERE table_schema = 'public' AND table_name = '{target_table}'");
		cols = [f'"{x[0]}"' for x in cur.fetchall()]
		print(cols)
		col_names = ', '.join(cols)
		# q_insert_distinct = f"""
		# WITH distincts AS (SELECT COUNT(*) AS cnt, {col_names} from {target_table}_pruned_intermediate
		# GROUP BY {col_names}) INSERT INTO {target_table}_pruned SELECT {col_names} FROM distincts
		# """
		q_insert_distinct = f"""
		INSERT INTO {target_table}_pruned SELECT DISTINCT {col_names} FROM {target_table}_pruned_intermediate
		"""
		cur.execute(q_insert_distinct)
		q_cnt_pruned=f"select count(*) from {target_table}_pruned"
		q_cnt_before_pruned=f"select count(*) from {target_table}"
		cur.execute(q_cnt_before_pruned)
		cnt_before = cur.fetchone()[0]
		cur.execute(q_cnt_pruned)
		cnt_after = cur.fetchone()[0]
		print(f"before pruning data: {target_table} has {cnt_before} rows")
		print(f"after pruning data: {target_table}_pruned has {cnt_after} rows")
		cur.execute(f"SELECT _tid_ FROM {target_table}_pruned")
		unique_tids = [t[0] for t in cur.fetchall()]
		# cols.remove('"_tid_"')
		return unique_tids, pd.read_sql(f"SELECT {','.join(cols)} FROM {target_table}_pruned order by _tid_", db_conn)

	def lf_prune_and_return(self, ):
		pass



class RuleCluster:
	def __init__(self, conn):
		self.conn = conn
		self.cur = self.conn.cursor()
	def gen_cluster_break_points(self, filename, table_name):
		with open(filename, 'r') as file:
		    rules = file.read().split('\n')
		rules = [r for r in rules if r!='']
		violation_dict = {r:None for r in rules}
		for r in rules:
		    r_q  = dc_violation_template.substitute(table=table_name, dc_desc=parse_rule_to_where_clause(r))
		    self.cur.execute(r_q)
		    violation_dict[r]=[t[0] for t in self.cur.fetchall()]
		vector_dict = {r:None for r in rules}
		for r in rules:
		    vector_dict[r] = [1 if i in violation_dict[r] else 0  for i in range(0, 500)]

		vector_list = []

		for k in vector_dict:
		    l = [k]
		    l.extend(vector_dict[k])
		    vector_list.append(l)
		columns = ['rule']
		columns.extend(list(range(0,500)))
		rule_violation_vector_df = pd.DataFrame(vector_list, columns=columns)
		# plt.figure(figsize=(12,5))
		# plt.title(f"Dendrogram with Single inkage: {filename.split('/')[-1]}")  
		# dend = shc.dendrogram(shc.linkage(rule_violation_vector_df[list(range(0,500))], method='average', metric='hamming'), labels=rule_violation_vector_df.index)
		#     print(dend)
		z=shc.linkage(rule_violation_vector_df[list(range(0,500))], method='average', metric='hamming')
		return [t[-2] for t in z], z



if __name__=='__main__':

	test = "t1&t2&EQ(t1.hours-per-week,t2.hours-per-week)&IQ(t1.native-country,t2.native-country)&IQ(t1.income,t2.income)&EQ(t1.relationship,t2.relationship)&IQ(t1.workclass,t2.workclass)&IQ(t1.income,'<=50k')"
	print(parse_rule_to_where_clause(test))

	# test_rules = """
	# t1&t2&IQ(t1.Sex,t2.Sex)&EQ(t1.Occupation,t2.Occupation)&IQ(t1.Country,t2.Country)
	# t1&t2&EQ(t1.Relationship,t2.Relationship)&EQ(t1.Education,t2.Education)&IQ(t1.Maritalstatus,t2.Maritalstatus)
	# """
	# test_rules = test_rules.split('\n')
	# print(test_rules)
	# c=Complaint(complain_type='DC', complaint_instance='Country')
	# rp=RulePruner()
	# dp=DataPruner()
	# graph, res = rp.prune_and_return(complaint=c,rules=test_rules)
	# print(f"before pruning: we have {len(test_rules)} rules ")
	# print(f"after pruning: we have {len(res)} rules ")
	# conn = psycopg2.connect(dbname="holo", user="holocleanuser", password="abcd1234")
	# conn.autocommit=True
	# dp.dc_prune_and_return(db_conn=conn,target_table='adult',pruned_rules=res)

	# c=Complaint(complain_type='DC', complaint_instance='Race')
	# rules="""
	# t1&t2&IQ(t1.Workclass,t2.Workclass)&IQ(t1.Race,t2.Race)&EQ(t1.HoursPerWeek,t2.HoursPerWeek)&IQ(t1.Country,t2.Country)
	# t1&t2&EQ(t1.Education,t2.Education)&EQ(t1.Maritalstatus,t2.Maritalstatus)&IQ(t1.Race,t2.Race)&IQ(t1.Country,t2.Country)
	# t1&t2&IQ(t1.Sex,t2.Sex)&EQ(t1.Occupation,t2.Occupation)&IQ(t1.Country,t2.Country)
	# t1&t2&EQ(t1.Relationship,t2.Relationship)&EQ(t1.Education,t2.Education)&IQ(t1.Maritalstatus,t2.Maritalstatus)
	# t1&t2&IQ(t1.Income,t2.Income)&IQ(t1.Race,t2.Race)&EQ(t1.HoursPerWeek,t2.HoursPerWeek)&IQ(t1.Country,t2.Country)
	# t1&t2&IQ(t1.Sex,t2.Sex)&EQ(t1.Education,t2.Education)&EQ(t1.Maritalstatus,t2.Maritalstatus)&IQ(t1.Race,t2.Race)
	# t1&t2&IQ(t1.Sex,t2.Sex)&EQ(t1.Age,t2.Age)&EQ(t1.HoursPerWeek,t2.HoursPerWeek)&IQ(t1.Country,t2.Country)
	# t1&t2&EQ(t1.Relationship,t2.Relationship)&IQ(t1.Income,t2.Income)&IQ(t1.Race,t2.Race)&EQ(t1.HoursPerWeek,t2.HoursPerWeek)
	# t1&t2&EQ(t1.Relationship,t2.Relationship)&IQ(t1.Workclass,t2.Workclass)&IQ(t1.Maritalstatus,t2.Maritalstatus)&IQ(t1.Race,t2.Race)
	# t1&t2&IQ(t1.Workclass,t2.Workclass)&EQ(t1.Education,t2.Education)&EQ(t1.Maritalstatus,t2.Maritalstatus)&IQ(t1.Country,t2.Country)
	# t1&t2&IQ(t1.Sex,t2.Sex)&EQ(t1.Maritalstatus,t2.Maritalstatus)&IQ(t1.Race,t2.Race)&EQ(t1.HoursPerWeek,t2.HoursPerWeek)
	# t1&t2&IQ(t1.Sex,t2.Sex)&EQ(t1.Relationship,t2.Relationship)&EQ(t1.Age,t2.Age)&EQ(t1.HoursPerWeek,t2.HoursPerWeek)
	# t1&t2&IQ(t1.Sex,t2.Sex)&IQ(t1.Income,t2.Income)&EQ(t1.Maritalstatus,t2.Maritalstatus)&IQ(t1.Country,t2.Country)
	# t1&t2&EQ(t1.Age,t2.Age)&IQ(t1.Race,t2.Race)&EQ(t1.Occupation,t2.Occupation)
	# t1&t2&EQ(t1.Age,t2.Age)&EQ(t1.Maritalstatus,t2.Maritalstatus)&IQ(t1.Relationship,t2.Relationship)&IQ(t1.Country,t2.Country)
	# t1&t2&EQ(t1.Education,t2.Education)&EQ(t1.Age,t2.Age)&EQ(t1.Occupation,t2.Occupation)
	# t1&t2&IQ(t1.Income,t2.Income)&EQ(t1.Education,t2.Education)&EQ(t1.Maritalstatus,t2.Maritalstatus)&IQ(t1.Relationship,t2.Relationship)
	# t1&t2&IQ(t1.Income,t2.Income)&EQ(t1.Maritalstatus,t2.Maritalstatus)&IQ(t1.Relationship,t2.Relationship)&IQ(t1.Race,t2.Race)
	# t1&t2&EQ(t1.Relationship,t2.Relationship)&EQ(t1.Age,t2.Age)&IQ(t1.Race,t2.Race)&EQ(t1.HoursPerWeek,t2.HoursPerWeek)
	# t1&t2&IQ(t1.Income,t2.Income)&EQ(t1.Maritalstatus,t2.Maritalstatus)&IQ(t1.Race,t2.Race)&IQ(t1.Country,t2.Country)
	# t1&t2&EQ(t1.Sex,t2.Sex)&IQ(t1.Income,t2.Income)&EQ(t1.Maritalstatus,t2.Maritalstatus)&IQ(t1.Relationship,t2.Relationship)
	# t1&t2&IQ(t1.Income,t2.Income)&IQ(t1.Workclass,t2.Workclass)&EQ(t1.Maritalstatus,t2.Maritalstatus)&EQ(t1.Occupation,t2.Occupation)
	# t1&t2&IQ(t1.Income,t2.Income)&IQ(t1.Workclass,t2.Workclass)&EQ(t1.Education,t2.Education)&IQ(t1.Race,t2.Race)
	# t1&t2&EQ(t1.Education,t2.Education)&EQ(t1.Age,t2.Age)&IQ(t1.Race,t2.Race)&IQ(t1.Country,t2.Country)
	# t1&t2&EQ(t1.Age,t2.Age)&EQ(t1.HoursPerWeek,t2.HoursPerWeek)&EQ(t1.Occupation,t2.Occupation)
	# t1&t2&IQ(t1.Income,t2.Income)&EQ(t1.Age,t2.Age)&IQ(t1.Race,t2.Race)&IQ(t1.Country,t2.Country)
	# t1&t2&EQ(t1.Relationship,t2.Relationship)&IQ(t1.Income,t2.Income)&EQ(t1.HoursPerWeek,t2.HoursPerWeek)&IQ(t1.Country,t2.Country)
	# t1&t2&IQ(t1.Sex,t2.Sex)&IQ(t1.Income,t2.Income)&EQ(t1.Age,t2.Age)&EQ(t1.Occupation,t2.Occupation)
	# t1&t2&IQ(t1.Sex,t2.Sex)&EQ(t1.Education,t2.Education)&EQ(t1.Occupation,t2.Occupation)

	# """
	# rule_input = rules.split('\n')
	# print(f'before prunning: rule_input has {len(rule_input)}')
	# rp = DataPruner()
	# res = rp.prune_and_return(c, rule_input)
	# print(f'after prunning: rule_input has {len(res)}')


	# print(dc_violation_template.substitute(table='adult', dc_desc='t1."Age"=t2."Age" and t1."Workclass"!=t2."Workclass"'))

	# x = 't1&t2&EQ(t1.Age,t2.Age)&IQ(t1.Race,t2.Race)&EQ(t1.HoursPerWeek,t2.HoursPerWeek)&IQ(t1.Country,t2.Country)' 



	# c=Complaint(complain_type='DC', complaint_instance='Race')
