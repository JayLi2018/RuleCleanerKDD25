#!/usr/bin/env python
# coding: utf-8

# In[1]:


# given a list of DCs, and a wrong repair(hand-picked), return the responsibility of the DCs


# ### 1. run holoclean and return wrongly repaired cells 

# In[1]:


from rbbm_src.holoclean.examples.holoclean_repair_example import main
import psycopg2
import pandas as pd
import copy
pd.set_option('display.max_rows', None)


# In[2]:


# connection to db 
for s in ['10','18']:
    conn = psycopg2.connect(dbname="holo", user="holocleanuser", password="abcd1234")
    cols = ["ProviderNumber","HospitalName","Address1","Address2","Address3","City","State","ZipCode","CountyName","PhoneNumber","HospitalType","HospitalOwner",
            "EmergencyService","Condition","MeasureCode","MeasureName","Score","Sample","Stateavg"]
    cur = conn.cursor()
    conn.autocommit=True

    # drop preexisted repaired records 
    select_old_repairs_q = """
    SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME LIKE 'hospital_repaired_%' AND TABLE_TYPE = 'BASE TABLE'
    """
    cur.execute(select_old_repairs_q)

    for records in cur.fetchall():
        drop_q = f"drop table if exists {records[0]}"
        cur.execute(drop_q)


    main(f'/home/opc/chenjie/holoclean/testdata/dc_finder_hospital_rules_{s}',initial_training=True)


    union_sql = """
    SELECT 'before_clean' AS type, * from  hospital
    union all 
    SELECT 'after_clean' AS type, * from hospital_repaired
    order by _tid_
    """
    df_union_before_and_after = pd.read_sql(union_sql, conn) 


    j = 0
    for i in range(0,1000):
        if(j%100==0):
            print(j)
        j+=1
        df_row = pd.DataFrame(columns=['type']+cols)
        df_row.loc[0,'type'] = 'ground_truth'
        df_row.loc[0,'_tid_'] = i
        q = f"""
        SELECT * FROM hospital_clean WHERE _tid_={i} 
        """
        df_for_one_row = pd.read_sql(q,conn)
        for index, row in df_for_one_row.iterrows():
            df_row.loc[0, f"{row['_attribute_']}"] = row['_value_']
        df_union_before_and_after = df_union_before_and_after.append(df_row, ignore_index=True)


    repaired_dict = {x:[] for x in cols}


    # iterate this dataframe to find all wrong predictions and list them to choose
    grouped = df_union_before_and_after.groupby('_tid_')
    k = 0

    for name, group in grouped:
        tid = pd.to_numeric(group.iloc[0]['_tid_'], downcast="integer")
        if(k%100==0):
            print(k)
        for c in cols:
            if(group[group['type']=='after_clean'][c].to_string(index=False)!= group[group['type']=='before_clean'][c].to_string(index=False)):
                repaired_dict[c].append(tid)
        k+=1

    for d,v in repaired_dict.items():
        print(f"{d}: {len(repaired_dict[d])}")


    wrong_dict = {x:[] for x in cols}

    # iterate this dataframe to find all wrong predictions and list them to choose
    grouped = df_union_before_and_after.groupby('_tid_')
    k = 0

    for name, group in grouped:
        tid = pd.to_numeric(group.iloc[0]['_tid_'], downcast="integer")
        if(k%100==0):
            print(k)
        for c in cols:
            if((group[group['type']=='after_clean'][c].to_string(index=False)!= group[group['type']=='ground_truth'][c].to_string(index=False)) \
                and (group[group['type']=='after_clean'][c].to_string(index=False)!= group[group['type']=='before_clean'][c].to_string(index=False))):
                wrong_dict[c].append(tid)
        k+=1


    wrong_cities=df_union_before_and_after[df_union_before_and_after['_tid_'].isin(wrong_dict['Stateavg'])]

    conn.close()

    # ### given an attribute and a tid (row id), find responsible rule(s)

    # In[ ]:


    import hc_responsibility
    hc_responsibility.rule_responsibility(attr_name='Stateavg', tid=432, size=s)
