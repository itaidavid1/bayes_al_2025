# generate_sbatch_commands.py
import sys

import numpy as np

number_of_reps = 3
max_number_of_commands_per_job = 1


# datasets = ["cifar100"]
# datasets = ["half_moon", "scenario_a"]
datasets = ["cifar10"]

al_methods = ["bayes_misp"]
# eval_models = ["from_features"]
eval_models = ["from_images"]
EXP_NAME = "fixed_bayes_misp_full_test_rbf_1011"

alpha_values = [0.001, 0.01, 0.1, 0.5]
# alpha_values = [0.5, 1, 2.5]
# alpha_values = [0.001, 0.01, 0.1, 1]
# alpha_values = [0.000005, 0.00001, 0.00005, 0.0005, 0.005, 0.001, 0.005, 0.01,
#  0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25]

# diff_methods = ["max", "top2_weighted_max"]
# diff_methods = ["full_weighted_max"]
diff_methods = ["full_weighted_max", "top2_weighted_max"]
# cont_method = ["positive", "abs", "fusion", "reg_sum_positive", "reg_sum_abs"]
# cont_method = ["positive", "abs", "fusion", "reg_sum_positive", "reg_sum_abs"]
cont_method = ["reg_sum_positive", "positive"]
# cont_method = ["positive"]

# kernel_type = "tophat"
kernel_type = "rbf"

template = (
    "python ./train_al.py "
    "--cfg ../configs/{dataset}/al/RESNET18.yaml "
    "--al {al_method} "
    "--exp-name {exp_name} "
    "--initial_size 0 "
    "--budget {budget} "
    "--initial_delta {initial_delta} "
    "--eval_model {eval_model_flag} "
    "--kernel_type {kernel_type} "
    # "--seed {seed} "
    "--diff_method {diff_method} "
    "--alpha {alpha} "
    "--cont_method {cont_method} "
    # "--sparse_ds "
    # "--decrease_alpha "
    # "--max_iter 6"
)

# Map eval model to correct CLI flag
eval_flag_map = {
    "from_features": "from_features",
    "from_images": "from_images",
    "NN1": "NN1"
}

budget_map = {
    "cifar10": 10,
    "cifar100": 100,
    "half_moon": 2,
    "scenario_a": 5,

}

commands = []
seed = int(sys.argv[1]) - 1
for dataset in datasets:
    for eval_model in eval_models:
        flag = eval_flag_map[eval_model]
        for al_method in al_methods:
            for alpha in alpha_values:
                for diff_method in diff_methods:
                    for cont in cont_method:
                        if diff_method in["top2_weighted_max", "max"] and cont.startswith("reg_sum"):
                            continue
                        if kernel_type == 'tophat':
                            delta = 0.65 if dataset == "cifar100" else 0.75
                        elif kernel_type == 'rbf':
                            delta = 1.0
                        budget = budget_map[dataset]

                        cmd = template.format(
                            dataset=dataset,
                            exp_name=EXP_NAME,
                            al_method=al_method,
                            eval_model_flag=flag,
                            # seed=seed,
                            initial_delta=delta,
                            diff_method=diff_method,
                            alpha=alpha,
                            budget=budget,
                            kernel_type=kernel_type,
                            cont_method=cont
                        )
                        commands.append(cmd)
def calculate_total_jobs(commands, number_of_reps, max_number_of_commands_per_job):
    """
    Calculate the total number of jobs needed based on the commands,
    number of repetitions, and maximum commands per job.
    """
    total_commands = len(commands) * number_of_reps
    total_jobs = np.ceil(total_commands / max_number_of_commands_per_job).astype(int)

    seed_to_indices = {}
    for job_id in range(total_jobs):
        # For each job, select number_of_reps indices with wraparound
        indices = []
        for i in range(max_number_of_commands_per_job):
            overall_index = (job_id * max_number_of_commands_per_job + i)
            if overall_index >= total_commands:
                break
            index = overall_index % len(commands)
            indices.append(index)
        seed_to_indices[job_id] = indices


    return total_jobs, seed_to_indices


total_jobs, seed_to_indices = calculate_total_jobs(commands, number_of_reps, max_number_of_commands_per_job)




if seed in seed_to_indices:
    job_indices = np.arange(seed * max_number_of_commands_per_job, np.minimum((seed + 1) * max_number_of_commands_per_job, len(commands) * number_of_reps))
    ordered_seeds = job_indices // len(commands)

    selected_indices = seed_to_indices[seed]
    job_commands = [commands[i] for i in selected_indices]

    fixed_seed_commands = [job_commands[i] + f" --seed {ordered_seeds[i]} " for i in range(len(job_commands))]


for command in fixed_seed_commands:
    print(command)

