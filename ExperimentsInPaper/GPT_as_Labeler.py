#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import psycopg2
import logging
import pandas as pd
import re
import pickle 
from tqdm import tqdm

# Configure logging to write to a file
logging.basicConfig(
    filename="output.log",  # Name of the log file
    level=logging.INFO,     # Logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)

from datetime import datetime


# In[2]:


# Amazon 200000 68.9 positive/negative 15
# AGnews 60000 37.7 business/technology 9
# PP 54476 55.8 physician/professor 18
# IMDB 50000 230.7 positive/negative 7
# FNews 44898 405.9 true/false 11
# Yelp 38000 133.6 negative/positive 8
# PT 24588 62.2 professor/teacher 7
# PA 12236 62.6 painter/architect 10
# Tweets 11541 18.5 positive/negative 16
# SMS 5572 15.6 spam/ham 17
# MGenre 1945 26.5 action/romance 10


# In[3]:


used_datasets = [
    'amazon',
#     'agnews',
    'physician_professor',
    'imdb',
    'fakenews',
    'yelp',
#     'professor_teacher',
#     'painter_architect',
#     'tweets',
#     'spam',
#     'plots'
]


# In[4]:


conn = psycopg2.connect(
    dbname="label",
    user="postgres",
)


# In[5]:


mapping_dicts = {}


# In[6]:


for ud in used_datasets:
    print(ud)
    info_df = pd.read_sql(f'select count(*), class, label from {ud} group by class, label', conn)
    print(info_df)
    result = dict(zip(info_df['label'], info_df['class']))
    mapping_dicts[ud]= result
    print('\n')


# In[7]:


mapping_dicts


# In[8]:


import openai
import pandas as pd
from openai import OpenAI

# Set your API key
openai.api_key = "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"


# In[9]:


def extract_labels(response):
    # Use a regular expression to extract integers from the response
    labels = re.findall(r'\b\d+\b', response)
    
    # Convert to integers
    return [int(label) for label in labels]


# In[10]:


def classify_texts(batch, class_mapping, customized_content=None):
    # Create the system's instruction and user prompt
    
    class_mapping_string = '\n'.join([f" - {k} (use {v} to represent this class)" for k, v in class_mapping.items()])
    label_string = f"{' or '.join([str(v) for v in list(class_mapping.values())])}"
    if(not customized_content):
        messages = [
            {"role": "system", "content": "You are a sentiment analysis assistant."},
            {
                "role": "user",
                "content": (
                    "You are an annotator that classifies the following text into one of the two classes:"
                    + f"\n {class_mapping_string}"
                    + f"\n Classify the text below and respond with only the label ({label_string}):"
#                     + '\n'.join([f"{text}" for text in batch])
                     + "\n".join([f" text {idx}: \n{text}\n" for idx, text in enumerate(batch, 1)]) 
                    + "\n\n Provide the classifications as a comma-separated list."
                ),
            },
        ]

    else:
        messages = [
            {"role": "system", "content": "You are a sentiment analysis assistant."},
            {
                "role": "user",
                "content": (
                  customized_content
                ),
            },
        ]
    
    print(messages)
        
    client = OpenAI(api_key=openai.api_key)
    
    # Call the ChatCompletion endpoint
    
    response = client.chat.completions.create(
        model="gpt-4o",  # Use "gpt-4" for more advanced tasks if needed
        messages=messages,
        max_tokens=200,
        temperature=0,
    )
    
    print("raw response")
    print(response)
        
    res = response.choices[0].message.content
    logging.info(res)
    processed_res = extract_labels(res)
#     processed_res = [int(cls.strip()) for cls in res.split(',')]
#     print(processed_res)
    
    return processed_res


# In[11]:


def get_gpt_response_df(input_df, class_mapping, batch_size):
    texts = input_df['content'].tolist()
    batch_size = batch_size
    max_retry = 3
    preds = []
    
    update_interval = 1000  # Number of API calls after which tqdm updates
    total_batches = len(texts) // batch_size
    progress_bar = tqdm(total=total_batches, desc="Processing texts", unit="batch")

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logging.info(f"batch : {i}:{i + batch_size}")
        
        predicted_labels = classify_texts(batch, class_mapping)
        print(f"predicted_labels: {predicted_labels}")
        logging.info(f"predicted_labels: {predicted_labels}")
        cur_retry=0
        while(len(predicted_labels) != batch_size):
            if(i+batch_size>=len(texts)):
                last_batch_size = len(texts)-i
                if(last_batch_size == len(predicted_labels)):
                   print("this was the last batch and dont need to get response size = batch size")
                   logging.info("this was the last batch and dont need to get response size = batch size")
                   break
            predicted_labels = classify_texts(batch, class_mapping)
            print(f"retry {cur_retry} predicted_labels: {predicted_labels}")
            logging.info(f"retry {cur_retry} predicted_labels: {predicted_labels}")
            cur_retry+=1
            if(cur_retry>max_retry):
                print(f"instance still failed after {max_retry} times")
                logging.info(f"instance still failed after {max_retry} times")
                predicted_labels = [-100]*batch_size
        if(cur_retry>1):
            print(f"succeeded after {cur_retry} times!")
            
        preds.extend(predicted_labels)

        # Update tqdm every 1000 API calls
        if (i // batch_size + 1) % update_interval == 0:
            progress_bar.update(update_interval)

    input_df['preds_by_gpt'] = preds
    return input_df


# In[12]:


runtimes = {}


# In[13]:


for k, v in mapping_dicts.items():
    start_time = datetime.now()
    with open(f'{k}_results_from_gpt.pkl', 'wb') as f:
        df_input = pd.read_sql(f'select * from {k}', conn)
        print(f"running dataset {k}, with size = {len(df_input)}")
        logging.info(f"running dataset {k}, with size = {len(df_input)}")
        df_input_with_preds = get_gpt_response_df(input_df=df_input, 
                                                  class_mapping=v, 
                                                  batch_size=20)
        end_time = datetime.now()
        kruntime = (end_time - start_time).total_seconds()
        df_input_with_preds['runtime'] = kruntime
        pickle.dump(df_input_with_preds, f)
    print('\n')
#     break

    runtimes[k]=kruntime


# In[14]:


df_input_with_preds


# In[15]:


# # Load your dataset
# file_path = "../tweet_for_gpt_label.csv"
# data = pd.read_csv(file_path).head(20)

# # Prepare the dataset for processing
# texts = data['content'].tolist()
# batch_size = 10  # Define the batch size




    
#     print(response)
#     print(response.choices[0].message.content)
    
#     return [x for x in response.choices[0].message.content.split(', ')]

# #     # Extract and parse the response text
# #     classifications = response['choices'][0]['message']['content'].strip().split(",")
# #     return [int(cls.strip()) for cls in classifications]

# # Process the texts in batches
# predictions = []
# for i in range(0, len(texts), batch_size):
#     batch = texts[i:i + batch_size]
#     classify_texts(batch, class_mapping)
#     predictions.extend(batch_predictions)

# # Add predictions to the dataset
# data['predicted class'] = predictions

# # Save the results
# output_path = "tweets_with_gpt_classifications.csv"
# data.to_csv(output_path, index=False)

# print(f"Classification complete. Results saved to {output_path}.")

