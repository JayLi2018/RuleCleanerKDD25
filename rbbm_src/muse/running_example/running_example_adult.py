from rbbm_src.muse.database_generator.dba import DatabaseEngine
from rbbm_src.muse.Semantics.end_sem import EndSemantics
from rbbm_src.muse.Semantics.stage_sem import StageSemantics
from rbbm_src.muse.Semantics.step_sem import StepSemantics
from rbbm_src.muse.Semantics.independent_sem import IndependentSemantics

# specify the schema
# mas_schema = {"author": ('aid',
#                          'name',
#                          'oid'),
#               "publication": ('pid',
#                               'title',
#                               'year'),

#               "writes": ('aid', 'pid'),

#               "organization": ('oid',
#                                'name'),

#               "conference": ('cid',
#                              'mas_id',
#                              'name',
#                              'full_name',
#                              'homepage',
#                              'paper_count',
#                              'citation_count',
#                              'importance'),

#               "domain_conference" : ('cid', 'did'),

#               "domain" : ('did',
#                           'name',
#                           'paper_count',
#                           'importance'),

#               "cite" : ('citing', 'cited')
#               }

mas_schema = {
"adult":(
    '_tid_',
    'age',
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'hours-per-week',
    'native-country',
    'income'),
"flights_new":(
    'DAY_OF_WEEK',
 'FL_DATE',
 'OP_UNIQUE_CARRIER',
 'OP_CARRIER_FL_NUM',
 'ORIGIN_AIRPORT_ID',
 'ORIGIN',
 'ORIGIN_CITY_NAME',
 'ORIGIN_STATE_ABR',
 'ORIGIN_STATE_NM',
 'DEST_AIRPORT_ID',
 'DEST',
 'DEST_CITY_NAME',
 'DEST_STATE_ABR',
 'DEST_STATE_NM',
 'CRS_DEP_TIME',
 'DEP_TIME',
 'CRS_ARR_TIME',
 'ARR_TIME',
 'ACTUAL_ELAPSED_TIME',
 'DISTANCE',
 '_tid_',
 'is_dirty'),

"tax":(
"FName",
"LName",
"Gender",
"AreaCode",
"Phone",
"City",
"State",
"Zip",
"MaritalStatus",
"HasChild",
"Salary",
"Rate",
"SingleExemp",
"MarriedExemp",
"ChildExemp",
"_tid_",
"is_dirty"),

"hospital": {"providernumber",
 "HospitalName",
 "City",
 "State",
 "ZipCode",
 "CountyName",
 "PhoneNumber",
 "HospitalType",
 "HospitalOwner",
 "EmergencyService",
 "Condition",
 "MeasureCode",
 "MeasureName",
 "Sample",
 "Stateavg",
 "_tid_",
 "is_dirty"}
}

def read_rules(rule_file):
    """read programs from txt file"""
    all_programs = []
    with open(rule_file) as f:
        rules = []
        for line in f:
            if line.strip():
                tbl, r = line.split("|")
                rules.append((tbl, r[:-2]))
            else:
                all_programs.append([r for r in rules])
                rules = []
        all_programs.append(rules)
    return all_programs

# load delta programs
# programs = read_rules("../data/mas/join_programs.txt")

# programs = [
# ("test", "SELECT t1.* from test as t1, test as t2 where t1.job=t2.job AND t1.salary!=t2.salary"),
# ("test", "SELECT t1.* from test as t1, test as t2 where t1.ssn=t2.ssn AND t1.name!=t2.name AND t1.country!='south africa'")
# ]

# "SELECT hauthor1.* FROM hauthor AS hauthor1, hauthor AS hauthor2 WHERE hauthor1.aid = hauthor2.aid AND lower(hauthor1.name) <> lower(hauthor2.name);"

# start the database


def database_reset(db, tbl_names):
    """reset the database"""
    # print("executing query: select current_database();")
    # res = db.execute_query("select current_database();")
    # db_name = res[0][0]
    # print(f"dbname: {db_name}")
    db.delete_tables(tbl_names)
    print(f"deleted tables")
    # db.close_connection()
    # db = DatabaseEngine(db_name)
    db.load_database_tables(tbl_names)

# choose a delta program from the programs file


# database_reset(db)

# end_sem = EndSemantics(db, rules, tbl_names)
# end_semantics_result = end_sem.find_mss()
# print("result for end semantics:", end_semantics_result)

# # reset the database between runs
# database_reset(db)

# stage_sem = StageSemantics(db, rules, tbl_names)
# stage_semantics_result = stage_sem.find_mss()
# print("result for stage semantics:", stage_semantics_result)

# # reset the database between runs
# database_reset(db)

# step_sem = StepSemantics(db, rules, tbl_names)
# step_semantics_result = step_sem.find_mss(mas_schema)
# print("result for step semantics:", step_semantics_result)

# # reset the database between runs
# database_reset(db)

def muse_find_mss(rules, semantic_version='ind'):

    db = DatabaseEngine("cr")
    print("engine_created")
    tbl_names = ['adult','flights_new', 'tax']
    database_reset(db, tbl_names)
    print("database_reseted")
    # subfolder = "mas/"
    print("delta program:", rules)

    # end_sem = EndSemantics(db, rules, tbl_names)
    # end_semantics_result = end_sem.find_mss()
    # print("result for end semantics:", end_semantics_result)

    print("rules/programs:")
    print(rules)
    print(f"len rules: {len(rules)}")
    res = None 

    if(semantic_version=='ind'):
        ind_sem = IndependentSemantics(db, rules, tbl_names)
        res = ind_sem.find_mss(mas_schema)
        # print("result for independent semantics:", res)
    elif(semantic_version=='stage'):
        stage_sem = StageSemantics(db, rules, tbl_names)
        res = stage_sem.find_mss()
        # print("result for stage semantics:", res)
    elif(semantic_version=='end'):
        end_sem = EndSemantics(db, rules, tbl_names)
        res = end_sem.find_mss()
        # print("result for end semantics:", res)
    elif(semantic_version=='step'):
        step_sem = StepSemantics(db, rules, tbl_names)
        res = step_sem.find_mss(mas_schema)
        # print("result for end semantics:", res)
    else:
        print('not a valid semantic option')
        exit()

    db.close_connection()
    print("session_closed")
    
    return res
