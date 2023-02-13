# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 
# # from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.firefox.options import Options as FirefoxOptions
# import pickle
# import time 
# import os.path

# options = FirefoxOptions()
# # options.add_argument("--headless");
# driver = webdriver.Firefox(options=options)
# # driver.maximize_window()

# driver.get("http://localhost:8080/#/result/14?count=false&cached=true&load=true&ind=false&fd=false&md=false&cfd=false&ucc=false&cucc=false&od=false&mvdfalse&basicStat=false&dc=true")
# # driver.switch_to.frame('fevo-purchase-flow-frame')
# print(driver.page_source.encode("utf-8"))
# # rule_blks = driver.find_element(By.XPATH, "//tr[1]")
# # WebDriverWait(driver, 1000000).until(EC.element_to_be_clickable((By.XPATH, '//tbody/tr[4]'))).click()
# # # print(rule_blks)
# # rule_blks.click()

import requests
import json

#this is pretty print, it just makes JSON more human-readable in the console:
from pprint import pprint

# options = dict(
#     page_no = 1,
#     checkin = "07/15/2016",
#     checkout = "07/16/2016",
#     sw_lat = "40.83397847641101",
#     sw_lng = "-74.0845568169126",
#     ne_lat = "40.88991628064286",
#     ne_lng = "-73.86380028615088"
# )

json_url = 'http://localhost:8081/api/result-store/get-from-to/Denial%20Constraint/Predicates/true/0/3000'
# download the raw JSON
raw = requests.get(json_url).text

# parse it into a dict
data = json.loads(raw)

# pretty-print some cool data about the 0th listing
pprint(data)

# [{'result': {'predicates': [{'column1': {'columnIdentifier': 'Address1',
#                                          'tableIdentifier': 'hospital.csv'},
#                              'column2': {'columnIdentifier': 'Address1',
#                                          'tableIdentifier': 'hospital.csv'},
#                              'index1': 0,
#                              'index2': 1,
#                              'op': 'EQUAL',
#                              'type': 'de.metanome.algorithm_integration.PredicateVariable'},
#                             {'column1': {'columnIdentifier': 'MeasureCode',
#                                          'tableIdentifier': 'hospital.csv'},
#                              'column2': {'columnIdentifier': 'MeasureCode',
#                                          'tableIdentifier': 'hospital.csv'},
#                              'index1': 0,
#                              'index2': 1,
#                              'op': 'EQUAL',
#                              'type': 'de.metanome.algorithm_integration.PredicateVariable'}],
#              'type': 'DenialConstraint'}},


# ∀t0∈hospital.csv,t1∈hospital.csv:
# ¬[t0.Address1=t1.Address1∧
#  t0.MeasureCode=t1.MeasureCode]



# t1&t2&EQ(t1.HospitalName,t2.HospitalName)&IQ(t1.PhoneNumber,t2.PhoneNumber)

with open('test.txt', 'a') as f:
	for d in data:
		conditions = d['result']['predicates']
		tokens =['t1&t2']
		for c in conditions:
			t1_col = c['column1']['columnIdentifier']
			t2_col = c['column2']['columnIdentifier']
			if(c['op']=='EQUAL'):
				sign = 'EQ'
			if(c['op']=='UNEQUAL'):
				sign='IQ'
			tokens.append(f"{sign}(t1.{t1_col},t2.{t2_col})")
		f.write('&'.join(tokens)+'\n')