import glob
import os
import subprocess
from collections import OrderedDict
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pulp as lp
import pandas as pd
from snorkel.classification.data import DictDataset, DictDataLoader


def load_spam_dataset(load_train_labels: bool = False, split_dev_valid: bool = False):

    try:
        cwd = os.getcwd()
        if(not os.path.isdir(f'{cwd}/data')):
            subprocess.run(["bash", "download_data.sh"], check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode())
        raise e
    filenames = sorted(glob.glob("data/Youtube*.csv"))
    filenames.sort()

    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename)
        # Lowercase column names
        df.columns = map(str.lower, df.columns)
        # Remove comment_id field
        df = df.drop("comment_id", axis=1)
        # Add field indicating source video
        df["video"] = [i] * len(df)
        # Rename fields
        df = df.rename(columns={"class": "label", "content": "text"})
        # Shuffle order
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.reset_index(drop=True)

    return df

    # df_train = pd.concat(dfs[:4])
    # df_dev = df_train.sample(100, random_state=123)

    # if not load_train_labels:
    #     df_train["label"] = np.ones(len(df_train["label"])) * -1
    # df_valid_test = dfs[4]
    # df_valid, df_test = train_test_split(
    #     df_valid_test, test_size=250, random_state=123, stratify=df_valid_test.label
    # )

    # if split_dev_valid:
    #     return df_train, df_dev, df_valid, df_test
    # else:
    #     return df_train, df_test


def get_keras_logreg(input_dim, output_dim=2):
    model = tf.keras.Sequential()
    if output_dim == 1:
        loss = "binary_crossentropy"
        activation = tf.nn.sigmoid
    else:
        loss = "categorical_crossentropy"
        activation = tf.nn.softmax
    dense = tf.keras.layers.Dense(
        units=output_dim,
        input_dim=input_dim,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
    )
    model.add(dense)
    opt = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model


def get_keras_lstm(num_buckets, embed_dim=16, rnn_state_size=64):
    lstm_model = tf.keras.Sequential()
    lstm_model.add(tf.keras.layers.Embedding(num_buckets, embed_dim))
    lstm_model.add(tf.keras.layers.LSTM(rnn_state_size, activation=tf.nn.relu))
    lstm_model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    lstm_model.compile("Adagrad", "binary_crossentropy", metrics=["accuracy"])
    return lstm_model


def get_keras_early_stopping(patience=10, monitor="val_acc"):
    """Stops training if monitor value doesn't exceed the current max value after patience num of epochs"""
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor, patience=patience, verbose=1, restore_best_weights=True
    )


def map_pad_or_truncate(string, max_length=30, num_buckets=30000):
    """Tokenize text, pad or truncate to get max_length, and hash tokens."""
    ids = tf.keras.preprocessing.text.hashing_trick(
        string, n=num_buckets, hash_function="md5"
    )
    return ids[:max_length] + [0] * (max_length - len(ids))


def featurize_df_tokens(df):
    return np.array(list(map(map_pad_or_truncate, df.text)))


def preview_tfs(df, tfs):
    transformed_examples = []
    for f in tfs:
        for i, row in df.sample(frac=1, random_state=2).iterrows():
            transformed_or_none = f(row)
            # If TF returned a transformed example, record it in dict and move to next TF.
            if transformed_or_none is not None:
                transformed_examples.append(
                    OrderedDict(
                        {
                            "TF Name": f.name,
                            "Original Text": row.text,
                            "Transformed Text": transformed_or_none.text,
                        }
                    )
                )
                break
    return pd.DataFrame(transformed_examples)


def df_to_features(vectorizer, df, split):
    """Convert pandas DataFrame containing spam data to bag-of-words PyTorch features."""
    words = [row.text for i, row in df.iterrows()]

    if split == "train":
        feats = vectorizer.fit_transform(words)
    else:
        feats = vectorizer.transform(words)
    X = feats.todense()
    Y = df["label"].values
    return X, Y


def create_dict_dataloader(X, Y, split, **kwargs):
    """Create a DictDataLoader for bag-of-words features."""
    ds = DictDataset.from_tensors(torch.FloatTensor(X), torch.LongTensor(Y), split)
    return DictDataLoader(ds, **kwargs)


def get_pytorch_mlp(hidden_dim, num_layers):
    layers = []
    for _ in range(num_layers):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    return nn.Sequential(*layers)


def lf_constraint_solve(input_df, lf_acc_thresh=0.5, instance_acc_thresh=0.5, new_lf_names=[]):
    # Problem initialization
    prob = pulp.LpProblem("Label_Flip_Minimization", pulp.LpMinimize)

    # Parameters
    labeling_functions = input_df.columns[:-1]  
    num_instances = len(input_df)
    M = 1000 
    x_nlf1 = pulp.LpVariable("x_nlf1", cat='Binary')

    P_vars = pulp.LpVariable.dicts("P", (range(num_instances), labeling_functions), 
                                lowBound=-1, upBound=1, cat='Integer')
    new_lf_weight = 100

    # Binary variables for each type of flip
    flip_1_to_0 = pulp.LpVariable.dicts("flip_1_to_0", (range(num_instances), labeling_functions), cat='Binary')
    flip_1_to_neg1 = pulp.LpVariable.dicts("flip_1_to_neg1", (range(num_instances), labeling_functions), cat='Binary')
    flip_0_to_1 = pulp.LpVariable.dicts("flip_0_to_1", (range(num_instances), labeling_functions), cat='Binary')
    flip_0_to_neg1 = pulp.LpVariable.dicts("flip_0_to_neg1", (range(num_instances), labeling_functions), cat='Binary')
    flip_neg1_to_1 = pulp.LpVariable.dicts("flip_neg1_to_1", (range(num_instances), labeling_functions), cat='Binary')
    flip_neg1_to_0 = pulp.LpVariable.dicts("flip_neg1_to_0", (range(num_instances), labeling_functions), cat='Binary')

    # Binary variables to track correctness of predictions (1 if correct, 0 if not)
    correctness_vars = pulp.LpVariable.dicts("correct", (range(num_instances), labeling_functions), cat='Binary')

    # Objective: Minimize the number of flips
    flip_cost = pulp.lpSum([flip_1_to_0[i][lf] + flip_1_to_neg1[i][lf] + 
                            flip_0_to_1[i][lf] + flip_0_to_neg1[i][lf] + 
                            flip_neg1_to_1[i][lf] + flip_neg1_to_0[i][lf] 
                            for i in range(num_instances) for lf in labeling_functions])

    prob += flip_cost + new_lf_weight*x_nlf1, "Minimize_Flips"

    # Mutual exclusivity
    for i in range(num_instances):
        for lf in labeling_functions:
            prob += (flip_1_to_0[i][lf] + flip_1_to_neg1[i][lf] + 
                    flip_0_to_1[i][lf] + flip_0_to_neg1[i][lf] + 
                    flip_neg1_to_1[i][lf] + flip_neg1_to_0[i][lf]) <= 1, f"Flip_Exclusivity_{i}_{lf}"

    for i in range(num_instances):
        for lf in labeling_functions:
            original_val = input_df.loc[i, lf]

            if original_val == 1:
                prob += P_vars[i][lf] == 0 * flip_1_to_0[i][lf] + (-1) * flip_1_to_neg1[i][lf] + 1 * (1 - flip_1_to_0[i][lf] - flip_1_to_neg1[i][lf]), f"Flip_From_1_{i}_{lf}"
            elif original_val == 0:
                prob += P_vars[i][lf] == 1 * flip_0_to_1[i][lf] + (-1) * flip_0_to_neg1[i][lf] + 0 * (1 - flip_0_to_1[i][lf] - flip_0_to_neg1[i][lf]), f"Flip_From_0_{i}_{lf}"
            elif original_val == -1:
                prob += P_vars[i][lf] == 1 * flip_neg1_to_1[i][lf] + 0 * flip_neg1_to_0[i][lf] + (-1) * (1 - flip_neg1_to_1[i][lf] - flip_neg1_to_0[i][lf]), f"Flip_From_neg1_{i}_{lf}"

    # Accuracy constraint for each labeling function (except nlf1)
    for lf in labeling_functions:
        if lf == 'nlf1':
            lf_correct_predictions = pulp.lpSum([correctness_vars[i][lf] for i in range(num_instances)])
            prob += lf_correct_predictions >= lf_acc_thresh * num_instances - M * (1 - x_nlf1), f"LF_nlf1_Accuracy"
        else:
            lf_correct_predictions = pulp.lpSum([correctness_vars[i][lf] for i in range(num_instances)])
            prob += lf_correct_predictions >= lf_acc_thresh * num_instances, f"LF_{lf}_Accuracy"

    # Instance accuracy constraint (nlf1 is optional, conditional on x_nlf1)
    for i in range(num_instances):
        # Big-M method applied to conditional inclusion of nlf1
        # Ensure that nlf1's correctness is only counted if x_nlf1 == 1
        prob += correctness_vars[i]['nlf1'] <= M * x_nlf1, f"nlf1_active_{i}"
        
        correct_predictions_per_instance = pulp.lpSum([correctness_vars[i][lf] for lf in labeling_functions if lf != 'nlf1']) \
                                        + correctness_vars[i]['nlf1']
        num_labeling_functions_used = len(labeling_functions) - 1 + x_nlf1  # Adjust number of LFs based on nlf1
        prob += correct_predictions_per_instance >= instance_acc_thresh * num_labeling_functions_used, f"Instance_{i}_Accuracy"

    # Ensure correctness tracking between P_vars and true labels
    for i in range(num_instances):
        for lf in labeling_functions:
            true_label = input_df['tlabel'][i]
            
            # Ensure that correctness_vars[i][lf] is 1 if P_vars[i][lf] equals true_label, else 0
            prob += P_vars[i][lf] - true_label <= M * (1 - correctness_vars[i][lf]), f"Correctness_UpperBound_{i}_{lf}"
            prob += true_label - P_vars[i][lf] <= M * (1 - correctness_vars[i][lf]), f"Correctness_LowerBound_{i}_{lf}"


    # Solve the integer program
    prob.solve()

    p_vars_solution = pd.DataFrame(index=input_df.index, columns=labeling_functions)

    for i in range(num_instances):
        for lf in labeling_functions:
            p_vars_solution.loc[i, lf] = pulp.value(P_vars[i][lf])