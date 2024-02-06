import pandas as pd
import pickle
import json
import nltk
from nltk.tokenize import word_tokenize
import random
import string
try: 
    nltk.data.find('tokenizers/punkt')
except:
    print("did not find punkt, downloading it now.")
    nltk.download('punkt')

import os


# for each input dataset to datasculpt, we need to provide at least 
# one example per class, this script is used to create such examples and append them to
# example.json under LLMDP (datasculpt folder)

def map_label_to_integer(record, label_json_dict):
    for k,v in label_json_dict.items():
        if(record['label']==v):
            record['label']=int(k)
            return int(k)
    print("no valid key mappings!")
    exit()

def remove_files(directory_path):
    # remove existing files under the dir
    files = os.listdir(directory_path)

    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed file: {file_path}")
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")

def preprocess_data_to_wrench_format(dataset_name, dataset_path, label_dict,
                                     num_class, seed=42, example_sample_cnt=5, 
                                     train_sample_per_class=1000, test_sample_per_class=1000, 
                                     valid_sample_per_class=1000):


    # outputdir = '/Users/chenjieli/Desktop/LLMDP_chenjie/data/wrench_data/sato/'
    outputdir = f'../../LLMDP/data/wrench_data/{dataset_name}/'
    if not os.path.exists(outputdir):
            os.makedirs(outputdir)
    remove_files(outputdir)

    df = pd.read_csv(dataset_path+dataset_name+'.csv')

    df.rename(columns={'class': 'label', 'content':'data'}, inplace=True)

    grouped_df = df.groupby('label')
    counts = grouped_df.size().sort_values(ascending=True)
    # print(grouped_df.size())
    # print("counts ascending")
    # print(counts)
    # exit()
    # top_x_groups = grouped_df.size().nsmallest(num_class).index
    bottom_x_groups = grouped_df.size().nsmallest(num_class).index
    # classes_with_enough_instances = list(top_x_groups)
    classes_with_enough_instances = list(bottom_x_groups)

    # df_filtered_by_instances = df[df['label'].isin(top_x_groups)]
    df_filtered_by_instances = df[df['label'].isin(bottom_x_groups)]

    # create examples under examples.json
    sampled_example = df_filtered_by_instances.groupby('label').apply(lambda x: \
        x.sample(n=min(example_sample_cnt, x.shape[0]), random_state=seed)).reset_index(drop=True)
    sampled_example_json_format = sampled_example[['label', 'data']].to_json(orient='records')
    example_records = json.loads(sampled_example_json_format)
    # print('example_records')
    # print(example_records)
    for r in example_records:
        text = r['data']
        tokens = word_tokenize(text)
        filtered_tokens = set([token for token in tokens if token not in string.punctuation])
        sampled_keywords = random.sample(filtered_tokens, min(3, len(filtered_tokens)))
        r['keywords'] = ', '.join(sampled_keywords)
        r['explanation'] = f"The sequence has some commonly used {r['label']} keywords so the label should be {r['label']}"
    # for i in range(len(example_records)):
    #     example_records[i]['label'] = map_label_to_integer(example_records[i],label_dict)
    #     example_records[i]['label'] = example_records[i]['label']

    with open('../../LLMDP/examples.json', 'r') as file:
        existing_data = json.load(file)

    new_item = {dataset_name: example_records}                                                                                                                                   
    existing_data.update(new_item)

    # with open('/Users/chenjieli/Desktop/LLMDP_chenjie/examples.json', 'w') as file:
    with open('../../LLMDP/examples.json', 'w') as file:
        json.dump(existing_data, file, indent=2)

    # create train.json, test.json, valid.json
    train_df = df_filtered_by_instances.groupby('label').apply(lambda x: \
        x.sample(n=min(train_sample_per_class, x.shape[0]), random_state=seed)).reset_index(drop=True)
    remaining_df_after_sample_train = pd.merge(df_filtered_by_instances, train_df, how='outer', \
                            indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    valid_df = remaining_df_after_sample_train.groupby('label').apply(lambda x: \
        x.sample(n=min(valid_sample_per_class, x.shape[0]), random_state=seed)).reset_index(drop=True)
    remaining_df_after_sample_valid = pd.merge(remaining_df_after_sample_train, \
        valid_df, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    test_df = remaining_df_after_sample_valid.groupby('label').apply(lambda x: x.sample(n=min(test_sample_per_class, x.shape[0]), \
        random_state=seed)).reset_index(drop=True)

    dfs = [train_df, test_df, valid_df]
    jnames = ['train.json', 'test.json', 'valid.json']

    # Convert dictionary to JSON
    json_labels = json.dumps(label_dict, indent=2) 


    # Write JSON data to the file
    with open(outputdir+'label.json', "w") as json_file:
        json_file.write(json_labels)

    for x in list(zip(dfs, jnames)):
        res_path = outputdir + x[1]
        res_json_data  = x[0].to_dict(orient='index')
        for key in res_json_data:
            res_json_data[key]['data'] = {'text': res_json_data[key]['data']}
            res_json_data[key]['label'] = res_json_data[key]['label']
        # Convert the modified dictionary to a JSON string
        res_json_data = json.dumps(res_json_data, indent=2)
        with open(res_path, 'w') as res_json_file:
            res_json_file.write(res_json_data)

    # printout the task description, should copy them into gpt_util.py

    task = f"LF creation for {dataset_name} dataset"
    task_info = "In each iteration, the user will provide a sequence and a label indicating which class the sequence belongs to"
    class_info = [f'"{k} for {v}."' for k,v in label_dict.items()]

    print(f"task='{task}'")
    print(f"task_info='{task_info}'")
    print("class_info=")
    for cf in class_info:
        print(cf + ' \\')
if __name__ == '__main__':
    # preprocess_doduo_data(num_class=20)
    # preprocess_doduo_data(num_class=30)
    preprocess_data_to_wrench_format(num_class=78)