import pandas as pd 
import glob

df = pd.DataFrame(columns=['text','label'])

ham_files = [f for f in glob.glob("/home/jayli/labelling_explanation/data/enron/*/ham/*")]
spam_files = [f for f in glob.glob("/home/jayli/labelling_explanation/data/enron/*/spam/*")]

i=1
for f in ham_files:
	print(f"one file {i}/{len(ham_files)}")
	with open(f,encoding = "ISO-8859-1") as inputfile:
	    drow = {"text":' '.join([line for line in inputfile]), 
	    "label":'ham'}
	    df = df.append(drow, ignore_index = True)
	i+=1
i=1
for f in spam_files:
	print(f"one file {i}/{len(spam_files)}")
	with open(f,encoding = "ISO-8859-1") as inputfile:
	    drow = {"text":' '.join([line for line in inputfile]), 
	    "label":'spam'}
	    df = df.append(drow, ignore_index = True)
	i+=1

df.to_csv('enrons.csv', index=False)