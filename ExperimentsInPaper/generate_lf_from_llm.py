from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import openai
import pandas as pd
from openai import OpenAI
import logging
import psycopg2
import math 
import pickle


used_datasets = [
    # 'plots',
    'spam',
    'tweets',
    'painter_architect',
    'professor_teacher',
    'yelp',
    'fakenews',
    'imdb',
    'physician_professor',
    'agnews',
    'amazon'
]

conn = psycopg2.connect(
    dbname="label",
    user="postgres",
)



mapping_dicts = {}


for ud in used_datasets:
    print(ud)
    info_df = pd.read_sql(f'select count(*), class, label from {ud} group by class, label', conn)
    print(info_df)
    result = dict(zip(info_df['label'], info_df['class']))
    mapping_dicts[ud]= result
    print('\n')


# Configure logging to write to a file
logging.basicConfig(
    filename="output.log",  # Name of the log file
    level=logging.INFO,     # Logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)
# functions 

TEMPLATE_KEYWORD = """
def keyword_[label_name][label_number](x):
ABSTAIN = -1
 keywords = [list of identified keywords]
return [label_number] if any(keyword in x for keyword in keywords)
else ABSTAIN
"""

TEMPLATE_REGEX = """
def regex_[label_name][label_number](x):
 ABSTAIN = -1
 return [label_number] if [regular expression related condition]
else ABSTAIN
"""

def classify_texts(batch, class_mapping, customized_content=None):
    # Create the system's instruction and user prompt
    
    class_mapping_string = '\n'.join([f" - {k} (use {v} to represent this class)" for k, v in class_mapping.items()])
    label_string = f"{' or '.join([str(v) for v in list(class_mapping.values())])}"
    if(not customized_content):
        messages = [
            {"role": "system", "content": "You are a labelling function generator"},
            {
                "role": "user",
                "content": (
                    "You are helping me to generate labelling functions from a group of sentences provided with groud truth labels. "
                    + "\n below are the labels and their label number: "
                    + f"\n {class_mapping_string}"
                    + f"\n I will provide you the sentences and its ground truth label. Your task is to come up with the "
                    + f"labeling functions based on the given sentences"
                    + f"The labelling function can come from 2 types of templates. Keyword and Regular Expression." 
                    + f" No more than 10 keywords/regular expression is allowed per labelling function so try to select the best"
                    + f"keywords/regular expressions in terms of generality. Try to comeup with regular expressions that cannot be captured with keyword labeling functions."
                    + f"\n The 2 templates are show below:" 
                    + f"\n" + TEMPLATE_KEYWORD + "\n" + TEMPLATE_REGEX 
                    + f"\n The sentences are show in the format 'Sentence [setence id]: [Sentence content]. Label: [label_text]" 
                    + f"\n Below are the sentences: " 
                    + f"\n".join([f" Sentence {idx}: {b['content']}. Label: {b['label']}" for idx, b in enumerate(batch, 1)])
                    + f"\n come up with the labelling functions based on the given templates"
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
        max_tokens=1024,
        temperature=0,
    )
    
    print("raw response")
    print(response)
        
    res = response.choices[0].message.content
    logging.info(res)
    print(res)
    
    return res

df_lf_number_requirement = pd.read_csv('dataset_lf_meta_data.csv')


for k, v in mapping_dicts.items():
    responses = []
    df_input = pd.read_csv(f'clustered_{k}.csv')
    print(f"running dataset {k}, with size = {len(df_input)}")
    clusters = list(df_input['cluster'].unique())
    lf_count = df_lf_number_requirement.loc[df_lf_number_requirement["Dataset"] == k, "LFCount"].values[0]
    lf_per_cluster = math.floor(lf_count/len(clusters))
    for c in clusters:
        for i in range(lf_per_cluster):
            batch_test = df_input[df_input['cluster']==c].sample(30).to_dict(orient='records')
            response = classify_texts(batch=batch_test, class_mapping=v)
            responses.append(response)
    with open(f'lf_raw_response_datase_{k}_new.pkl', 'wb') as f:
        pickle.dump(responses, f)


# # Step 1: Embed the tex
# embeddings = embed_text(texts)  # Use a pre-trained embedding model

# # Step 2: Find the optimal number of clusters
# silhouette_scores = []
# for k in range(2, 20):
#     kmeans = KMeans(n_clusters=k)
#     labels = kmeans.fit_predict(embeddings)
#     silhouette_scores.append(silhouette_score(embeddings, labels))
# optimal_k = np.argmax(silhouette_scores) + 2  # Add 2 because range starts at 2

# # Step 3: Distribute LFs across clusters
# kmeans = KMeans(n_clusters=optimal_k)
# cluster_labels = kmeans.fit_predict(embeddings)
# cluster_sizes = np.bincount(cluster_labels)
# total_lfs = 100  # Example: Generate 100 LFs
# lfs_per_cluster = (cluster_sizes / len(texts)) * total_lfs

# # Step 4: Generate LFs using ChatGPT
# for cluster_id in range(optimal_k):
#     cluster_examples = [texts[i] for i in range(len(texts)) if cluster_labels[i] == cluster_id]
#     num_lfs = int(lfs_per_cluster[cluster_id])
#     for _ in range(num_lfs):
#         prompt = f"Generate a keyword-based labeling function for the following examples: {cluster_examples[:5]}"
#         response = openai.Completion.create(prompt=prompt, model="gpt-4")
#         lf = response.choices[0].text
#         save_labeling_function(lf)  # Save the generated LF
