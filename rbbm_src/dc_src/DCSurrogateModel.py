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
import logging
import random

logger = logging.getLogger(__name__)

dc_tuple_violation_template_targeted_t1=\
Template("SELECT DISTINCT $t1_desc,$t2_desc FROM $table t1, $table t2 WHERE $dc_desc")
dc_tuple_violation_template_targeted_t2=\
Template("SELECT DISTINCT $t1_desc,$t2_desc FROM $table t1, $table t2 WHERE $dc_desc")

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

        r_q+=f" AND t1._tid_!=t2._tid_"
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
        # print(q1)
        res = {'t1':pd.read_sql(q1, conn).to_dict('records')}
        # print(res)
    return res 


class DCSurrogateModel:
    def __init__(self, conn):
        self.conn = conn

    def delete_tuples(self, table_name, dcs):
        ts_to_be_deleted = set([])
        # return the tuples that needs to be deleted
        # in order to hold for the dcs
        cols_q = f"select * from {table_name} limit 1"
        df_row = pd.read_sql(cols_q, self.conn)
        cols=list(df_row)
        for r in dcs:
            res = find_tuples_in_violation(self.conn, r, table_name, cols)
            print(f"rule: {r}")
            for k in res:
                for pair in res[k]:
                    t1_id,t2_id = pair['t1__tid_'], pair['t2__tid_']
                    if((t1_id not in ts_to_be_deleted) and (t2_id not in ts_to_be_deleted)):
                        random_number = random.randint(1, 2)
                        if(random_number==1):
                            ts_to_be_deleted.add(t1_id)
                        else:
                            ts_to_be_deleted.add(t2_id)
        print(f"ts_to_be_deleted: {ts_to_be_deleted}")
        return ts_to_be_deleted 
        # return res



if __name__ == '__main__':
    conn=psycopg2.connect('dbname=cr user=postgres port=5432')
    ds = DCSurrogateModel(conn)
    dc_file='/home/opc/author/RBBM/rbbm_src/muse/data/mas/tax_rules.txt'
    dcs = []
    try:
        with open(dc_file, "r") as file:
            for line in file:
                # rule_texts.append(line.strip())
                # rules_from_line = [(table_name, x) for x in convert_dc_to_muse_rule(line, 'adult', 't1')]
                rule,gt=line.strip('\n').split(":")
                dcs.append(rule)
        logger.debug(dcs)
        # exit()
    except FileNotFoundError:
        logger.debug("File not found.")
    except IOError:
        logger.debug("Error reading the file.")
    deleted = ds.delete_tuples('tax', dcs)

    print(f"deleted:")
    print(deleted)

    print(f"len(deleted) = {len(deleted)}")