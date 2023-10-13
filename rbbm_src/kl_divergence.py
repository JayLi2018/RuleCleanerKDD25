import sys
import numpy as np
import os 

cols_to_compare = ['rbbm_runtime','bbox_runtime','avg_tree_size_increase','fix_rate','confirm_preserve_rate','new_global_accuracy']

def kl_divergence(p, q):
    """
    Calculate the Kullback-Leibler (KL) divergence between two probability distributions p and q.

    Parameters:
    p (list): The first probability distribution.
    q (list): The second probability distribution.

    Returns:
    float: The KL divergence value.
    """
    # Ensure that p and q have the same length and are non-negative
    if len(p) != len(q):
        raise ValueError("Input distributions must have the same length")
    if any(x < 0 for x in p) or any(x < 0 for x in q):
        raise ValueError("Probabilities must be non-negative")

    # Avoid division by zero by adding a small epsilon value
    epsilon = 1e-10
    p = [x + epsilon for x in p]
    q = [x + epsilon for x in q]

    # Calculate KL divergence
    kl = sum(x * (np.log(x / y)) for x, y in zip(p, q))

    return kl


if __name__ == '__main__':
    # Start from the first argument (script name is sys.argv[0])
    args = sys.argv[1:]
    run_kl = args[0]
    result_file = args[1]
    stats_file = args[2]
    num_per_run = args[3]
    stats_from_current_run = args[4:]

    if(not os.path.exists(stats_file)):
        with open(stats_file, 'w') as file:
            # Write some text to the file
            file.write('rbbm_runtime','bbox_runtime','avg_tree_size_increase','fix_rate','confirm_preserve_rate','new_global_accuracy\n')

    stats_from_this_run = pd.concat([f for f in stats_from_current_run])
    stats_df = pd.read_csv(stats_file)

    if(run_kl=='yes'):
        if(not os.path.exists(result_file)):
            with open(result_file, 'w') as file:
                # Write some text to the file
                file.write('kl_rbbm_runtime','kl_bbox_runtime','kl_avg_tree_size_increase','kl_fix_rate','kl_confirm_preserve_rate','kl_new_global_accuracy\n')

        stats_from_last_run = stats_df.tail(num_per_run)
        kl_divs = []
        for c in cols_to_compare:
            kl_value = kl_divergence(stats_from_last_run[c], stats_from_this_run[c])
            kl_divs.append(kl_value)
            print(f"KL Divergence for {c}: {kl_value}")
        with open(result_file, 'a') as file:
            file.write(f"{','.join(kl_divs)}\n")


    pd.concat([stats_df,stats_from_this_run]).to_csv(stats_file, index=False)
