#!/usr/bin/env python
# coding: utf-8

# In[1]:


# given a list of DCs, and a wrong repair(hand-picked), return the responsibility of the DCs


# ### 1. run holoclean and return wrongly repaired cells 

# In[2]:


from rbbm_src.holoclean.examples.holoclean_repair_example_adults import main
import psycopg2
import pandas as pd
import copy
pd.set_option('display.max_rows', None)


# In[3]:


# connection to db 
conn = psycopg2.connect(dbname="holo", user="holocleanuser", password="abcd1234")
cols = ["Age","Workclass","Education","Maritalstatus","Occupation","Relationship",
        "Race","Sex","HoursPerWeek","Country","Income"]
cur = conn.cursor()
conn.autocommit=True


# In[4]:


# drop preexisted repaired records 
select_old_repairs_q = """
SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_NAME LIKE 'adult_repaired_%' AND TABLE_TYPE = 'BASE TABLE'
"""
cur.execute(select_old_repairs_q)


# In[5]:


for records in cur.fetchall():
    drop_q = f"drop table if exists {records[0]}"
    cur.execute(drop_q)


# In[6]:


import time
start_time = time.time()
main('/home/opc/chenjie/holoclean/testdata/dc_finder_adult_rules', initial_training=True)
print("--- %s seconds ---" % (time.time() - start_time))


# In[7]:


union_sql = """
SELECT 'before_clean' AS type, * from  adult
union all 
SELECT 'after_clean' AS type, * from adult_repaired
order by _tid_
"""
df_union_before_and_after = pd.read_sql(union_sql, conn) 


# In[8]:


df_union_before_and_after


# In[9]:


j = 0
for i in range(0,500):
    if(j%100==0):
        print(j)
    j+=1
    df_row = pd.DataFrame(columns=['type']+cols)
    df_row.loc[0,'type'] = 'ground_truth'
    df_row.loc[0,'_tid_'] = i
    q = f"""
    SELECT * FROM adult_clean WHERE _tid_={i} 
    """
    df_for_one_row = pd.read_sql(q,conn)
    for index, row in df_for_one_row.iterrows():
        df_row.loc[0, f"{row['_attribute_']}"] = row['_value_']
    df_union_before_and_after = df_union_before_and_after.append(df_row, ignore_index=True)


# In[10]:


repaired_dict = {x:[] for x in cols}


# In[11]:


# iterate this dataframe to find all wrong predictions and list them to choose
grouped = df_union_before_and_after.groupby('_tid_')
k = 0

for name, group in grouped:
    tid = pd.to_numeric(group.iloc[0]['_tid_'], downcast="integer")
    if(k%100==0):
        print(k)
    for c in cols:
        if(group[group['type']=='after_clean'][c].to_string(index=False)           != group[group['type']=='before_clean'][c].to_string(index=False)):
            repaired_dict[c].append(tid)
    k+=1


# In[12]:


repaired_dict


# In[13]:


for d,v in repaired_dict.items():
    print(f"{d}: {len(repaired_dict[d])}")


# In[14]:


wrong_dict = {x:[] for x in cols}


# In[15]:


# iterate this dataframe to find all wrong predictions and list them to choose
grouped = df_union_before_and_after.groupby('_tid_')
k = 0

for name, group in grouped:
    tid = pd.to_numeric(group.iloc[0]['_tid_'], downcast="integer")
    if(k%100==0):
        print(k)
    for c in cols:
        if((group[group['type']=='after_clean'][c].to_string(index=False)           != group[group['type']=='ground_truth'][c].to_string(index=False)           ) and (group[group['type']=='after_clean'][c].to_string(index=False)           != group[group['type']=='before_clean'][c].to_string(index=False))):
            wrong_dict[c].append(tid)
    k+=1


# In[16]:


wrong_dict


# In[17]:


wrongs=df_union_before_and_after[df_union_before_and_after['_tid_'].isin(wrong_dict['Relationship'])]


# In[18]:


wrongs


# In[19]:


conn.close()


# ### given an attribute and a tid (row id), find responsible rule(s)

# In[20]:


import hc_responsibility
hc_responsibility.rule_responsibility(attr_name='Relationship', tid=309,
                      all_rules_file_name='/home/opc/chenjie/holoclean/testdata/dc_finder_adult_rules',
                      rule_subset_file_name='/home/opc/chenjie/holoclean/testdata/dc_finder_adult_rules_test')
# def rule_responsibility(attr_name,tid,all_rules_file_name, rule_subset_file_name):


# In[ ]:





# In[ ]:


from itertools import combinations


# In[ ]:


x = list(range(0,30))
res = 0
for i in range(0, 3):
    print(f"i={i}, combs={len(list(combinations(x,i)))}")
    res+=len(list(combinations(x,i)))*len(x)


# In[ ]:


res/60/24


# In[ ]:




