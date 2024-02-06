#!/bin/bash

# List of dataset names

dataset_names=('youtube_new' 'amazon' 'pt')
# dataset_names=('youtube_new' 'amazon' 'pa' 'pt')

# 'amazon' 'amazon05' 'pa' 'pt'
# Other parameters
lf_llm_model="gpt-3.5-turbo-1106"
runs=1
num_query=50
train_iter=50
return_explanation='--return-explanation'
sample_instance_per_class=25
example_per_class=5

# Loop over each dataset name
for dataset_name in "${dataset_names[@]}"; do
    python gpt_driver.py --dataset-name "$dataset_name" \
        --lf-llm-model "$lf_llm_model" \
        --runs "$runs" \
        --display \
        --num-query "$num_query" \
        --train-iter "$train_iter" \
        --use-rc-flavor \
        "$return_explanation" \
        --sample-instance-per-class "$sample_instance_per_class" \
        --example-per-class "$example_per_class"

    echo "python gpt_driver.py --dataset-name \"$dataset_name\" \\
    --lf-llm-model \"$lf_llm_model\" \\
    --runs \"$runs\" \\
    --display \\
    --num-query \"$num_query\" \\
    --train-iter \"$train_iter\" \\
    \"$use_rc_rule\" \\
    --sample-instance-per-class \"$sample_instance_per_class\" \\
    --example-per-class \"$example_per_class\""
done