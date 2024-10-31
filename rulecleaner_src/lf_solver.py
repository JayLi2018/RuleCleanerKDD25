import pulp 
import pandas as pd 

def lf_constraint_solve_no_new_lf(df, lf_acc_thresh=0.5, 
                        instance_acc_thresh=0.5,
                        min_non_abstain_thresh=0.8,
                        expected_label_col='expected_label',
                        instance_acc_on_valid=False,
                        use_non_abstain=True
                       ):
    
    # Problem initialization
    prob = pulp.LpProblem("Label_Flip_Minimization", pulp.LpMinimize)

    # Parameters
    labeling_functions = [lf_name for lf_name in df.columns if lf_name!=expected_label_col]
    print(f"lf_acc: {lf_acc_thresh}, ins_acc:{instance_acc_thresh}, min_non_abstain_thresh:{min_non_abstain_thresh}")
    print(f"labeling_functions: {labeling_functions}")
    num_instances = len(df)
    print(f"num_instances: {num_instances}")
    M = 5

    P_vars = pulp.LpVariable.dicts("P", (range(num_instances), labeling_functions), 
                                   lowBound=-1, upBound=1, cat='Integer')
    
    is_abstain = pulp.LpVariable.dicts("is_abstain", 
                               (range(num_instances), labeling_functions), 
                               cat='Binary')

    flip_1_to_0 = pulp.LpVariable.dicts("flip_1_to_0", 
                                        (range(num_instances), labeling_functions), cat='Binary')
    flip_1_to_neg1 = pulp.LpVariable.dicts("flip_1_to_neg1", 
                                           (range(num_instances), labeling_functions), cat='Binary')
    flip_0_to_1 = pulp.LpVariable.dicts("flip_0_to_1", 
                                        (range(num_instances), labeling_functions), cat='Binary')
    flip_0_to_neg1 = pulp.LpVariable.dicts("flip_0_to_neg1", 
                                           (range(num_instances), labeling_functions), cat='Binary')
    flip_neg1_to_1 = pulp.LpVariable.dicts("flip_neg1_to_1", 
                                           (range(num_instances), labeling_functions), cat='Binary')
    flip_neg1_to_0 = pulp.LpVariable.dicts("flip_neg1_to_0", 
                                           (range(num_instances), labeling_functions), cat='Binary')

    correctness_vars = pulp.LpVariable.dicts("correct", 
                                             (range(num_instances), labeling_functions), cat='Binary')


    # Objective: Minimize the number of flips
    flip_cost = pulp.lpSum([flip_1_to_0[i][lf] + flip_1_to_neg1[i][lf] + 
                            flip_0_to_1[i][lf] + flip_0_to_neg1[i][lf] + 
                            flip_neg1_to_1[i][lf] + flip_neg1_to_0[i][lf] 
                            for i in range(num_instances) for lf in labeling_functions])

    prob += flip_cost, "Minimize_Flips"


    # Mutual exclusivity
    for i in range(num_instances):
        for lf in labeling_functions:
            prob += (flip_1_to_0[i][lf] + flip_1_to_neg1[i][lf] + 
                     flip_0_to_1[i][lf] + flip_0_to_neg1[i][lf] + 
                     flip_neg1_to_1[i][lf] + flip_neg1_to_0[i][lf]) <= 1, f"Flip_Exclusivity_{i}_{lf}"

    for i in range(num_instances):
        for lf in labeling_functions:
            original_val = df.loc[i, lf]
            if original_val == 1:
                prob += P_vars[i][lf] == 0 * flip_1_to_0[i][lf] + \
                (-1) * flip_1_to_neg1[i][lf] + 1 * (1 - flip_1_to_0[i][lf] - flip_1_to_neg1[i][lf]), f"Flip_From_1_{i}_{lf}"
                
            elif original_val == 0:                
                prob += P_vars[i][lf] == 1 * flip_0_to_1[i][lf] + \
                (-1) * flip_0_to_neg1[i][lf] + 0 * (1 - flip_0_to_1[i][lf] - flip_0_to_neg1[i][lf]), f"Flip_From_0_{i}_{lf}"
                
            elif original_val == -1:
                prob += P_vars[i][lf] == 1 * flip_neg1_to_1[i][lf] + 0 * flip_neg1_to_0[i][lf] + (-1) * (1 - flip_neg1_to_1[i][lf] - flip_neg1_to_0[i][lf]), f"Flip_From_neg1_{i}_{lf}"
    
    for i in range(num_instances):
        for lf in labeling_functions:
            prob += P_vars[i][lf] >= -1 - (1 - is_abstain[i][lf]) * M, f"Abstain_LowerBound_{i}_{lf}"
            prob += P_vars[i][lf] <= -1 + (1 - is_abstain[i][lf]) * M, f"Abstain_UpperBound_{i}_{lf}"

            # If is_abstain[i][lf] == 0, P_vars[i][lf] can only be 0 or 1
            prob += P_vars[i][lf] >= 0 - is_abstain[i][lf] * M, f"Non_Abstain_LowerBound_{i}_{lf}"
            prob += P_vars[i][lf] <= 1 + is_abstain[i][lf] * M, f"Non_Abstain_UpperBound_{i}_{lf}"
    
    # Set up the constraints for the auxiliary variables
    for lf in labeling_functions:
        num_instances_abstain = pulp.lpSum([is_abstain[i][lf] for i in range(num_instances)])
        lf_correct_predictions = pulp.lpSum([correctness_vars[i][lf] for i in range(num_instances)])
        prob += lf_correct_predictions >= lf_acc_thresh * (num_instances-num_instances_abstain), f"LF_{lf}_Accuracy"



    for i in range(num_instances):
        correct_predictions_per_instance = pulp.lpSum([correctness_vars[i][lf] for lf in labeling_functions])
        instance_abstain_count = pulp.lpSum([is_abstain[i][lf] for lf in labeling_functions])        
        num_labeling_functions_used = len(labeling_functions)
        if(instance_acc_on_valid):
            prob += correct_predictions_per_instance >= instance_acc_thresh * (num_labeling_functions_used-instance_abstain_count), f"Instance_{i}_Accuracy"
        else:
            prob += correct_predictions_per_instance >= instance_acc_thresh * (num_labeling_functions_used), f"Instance_{i}_Accuracy"
        if(use_non_abstain):
            prob += instance_abstain_count <= num_labeling_functions_used *(1- min_non_abstain_thresh), f"Instance_{i}_NonAbastain"

        
    for i in range(num_instances):
        for lf in labeling_functions:
            true_label = df[expected_label_col][i]
            prob += P_vars[i][lf] - true_label <= M * (1 - correctness_vars[i][lf]),\
                                     f"Correctness_UpperBound_{i}_{lf}"
            prob += true_label - P_vars[i][lf] <= M * (1 - correctness_vars[i][lf]), \
                                     f"Correctness_LowerBound_{i}_{lf}"


    # Solve the integer program
            

    solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=600)
    prob.solve(solver)
    # prob.solve()

    p_vars_solution = pd.DataFrame(index=df.index, columns=labeling_functions)
    active_abstain_df = pd.DataFrame(index=df.index, columns=labeling_functions)
    is_abstain_df = pd.DataFrame(index=df.index, columns=labeling_functions)
    
    for i in range(num_instances):
        for lf in labeling_functions:
            p_vars_solution.loc[i, lf] = int(pulp.value(P_vars[i][lf]))
    
    correctness_solution = pd.DataFrame(index=df.index, columns=labeling_functions)
    for i in range(num_instances):
        for lf in labeling_functions:
            correctness_solution.loc[i, lf] = int(pulp.value(correctness_vars[i][lf]))
    
#     x_nlfs_solution = {lf: pulp.value(x_nlfs[lf]) for lf in nlfs}
    
    print(f"Status: {pulp.LpStatus[prob.status]}")
    print(f"pulp.value(num_labeling_functions_used) : {pulp.value(num_labeling_functions_used)}")
    
#     for i in range(num_instances):
#         for lf in labeling_functions:
#             is_abstain_df.loc[i, lf] = int(pulp.value(is_abstain[i][lf]))
#     for i in range(num_instances):
#         for lf in nlfs:
#             active_abstain_df.loc[i, lf] = int(pulp.value(active_abstain[i][lf]))
    
#     return p_vars_solution, x_nlfs_solution, pulp, prob, active_abstain_df, is_abstain_df

    return p_vars_solution, pulp.value(flip_cost)