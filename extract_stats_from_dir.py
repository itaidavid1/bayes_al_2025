import os
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
def moving_average(data, window_size):
    """Calculates the moving average of a 1D array."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size



BASE_OUTPUT_PATH = "/cs/labs/daphna/itai.david/py_repos/TypiClust/visual_outputs/"
# Replace with your actual root directory path
# ROOT_DIRS = ['/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_9_21']# ,'/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_8_15',  '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_8_14',  '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_8_13']
# ROOT_DIRS = [  '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_9_16',
#  '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_9_21',]
              # '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_9_17']
# ROOT_DIRS = ['/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_9_22',
#                 '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_9_23',
#              '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_9_24',
#              '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_9_25',
#             '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_9_28',
#              '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_9_27']
#              # ]
# ROOT_DIRS = ['/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_10_16',
#              '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_10_19',
#              '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_10_20',
#              '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_10_22']
# ROOT_DIRS = [ '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_10_28',
#               '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_10_29',
#               '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_10_30',
#               '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_11_1',
#               '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_11_2']
# ROOT_DIRS = [
#                 '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_11_4',
#               '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_11_5',
#               '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_11_7',
#               '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_11_10']


ROOT_DIRS = [   '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_11_4',
                '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_11_5',
              '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_11_7',
              '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_11_10',
    '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_11_11',
             '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_11_12']

# ROOT_DIRS = [ '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_10_26',
#              '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR10/resnet18/2025_10_23',
#               '/cs/labs/daphna/itai.david/py_repos/TypiClust_clean/output/CIFAR10/resnet18/']


# ROOT_DIRS = ['/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_11_2',
#               '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_11_3']

# ROOT_DIRS = [ '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/HALF_MOON/resnet18/2025_10_20',
#              '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/SCENARIO_A/resnet18/2025_10_20',
#               '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/HALF_MOON/resnet18/2025_10_22',
#               '/cs/labs/daphna/itai.david/py_repos/TypiClust/output/SCENARIO_A/resnet18/2025_10_22'
#               ]

# Store all the data in a structured form
records = []
eval_models = ["from_features", "from_images", "NN1"]
al_methods = ["dcom", "margin", "random", "bayes_misp", "all_misp", "max_misp", "misp", "prob_cover", "max_herding", ]
datasets_names = ["scenraio_a", "half_moon","CIFAR100", "CIFAR10", 'tinyimage']
for root_dir in ROOT_DIRS:
    files_names = os.listdir(root_dir)

    for folder_name in files_names:
        # check within each folder if the folder episode_31 exists if not continue
        if not os.path.isdir(os.path.join(root_dir, folder_name, 'episode_50')):
            continue
        folder_path = os.path.join(root_dir, folder_name)
        config_path = os.path.join(folder_path, 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        sampling_fn = config['ACTIVE_LEARNING']['SAMPLING_FN']
        eval_model = config['EVAL_MODEL_TYPE'] if 'EVAL_MODEL_TYPE' in config else "from_features"
        dataset_name = config['DATASET']['NAME']
        if not sampling_fn or not eval_model:
            continue  # Skip folders that don't match expected patterns

        # Use everything after the model name as an experiment ID
        exp_id_start = folder_name.find(eval_model) + len(eval_model) + 1
        exp_id = folder_name# "2025_8_7_123812_251469"
        print(exp_id,exp_id_start)
        x_path = os.path.join(folder_path, 'plot_episode_xvalues.npy')
        y_path = os.path.join(folder_path, 'plot_episode_yvalues.npy')


        initial_delta = config['ACTIVE_LEARNING']['INITIAL_DELTA']
        kernel_type = config['KERNEL_TYPE'] if 'KERNEL_TYPE' in config else 'none'
        diff_method = config['DIFF_METHOD'] if 'DIFF_METHOD' in config else 'none'
        diff_method = diff_method if sampling_fn == "bayes_misp" else 'none'
        alpha = config['ALPHA'] if 'ALPHA' in config else -1
        alpha = alpha if sampling_fn == "bayes_misp" else -1
        cont_method = config['CONT_METHOD'] if 'CONT_METHOD' in config else 'none'
        cont_method = cont_method if sampling_fn == "bayes_misp" else 'none'
        alpha_decrease_bool = config['DECREASING_ALPHA'] if 'DECREASING_ALPHA' in config else False
        alpha_decrease = "True" if alpha_decrease_bool else "False"
        alpha_decrease = alpha_decrease if sampling_fn == "bayes_misp" else 'none'
        sparse_ds = config['SPARSE_DS'] if 'SPARSE_DS' in config else False
        seed = config['RNG_SEED'] if 'RNG_SEED' in config else -1

        if os.path.exists(x_path) and os.path.exists(y_path):
            x_vals = np.load(x_path)
            y_vals = np.load(y_path)


            lset = None
            lset_path = os.path.join(folder_path, f'episode_{y_vals.size - 1}', 'lSet.npy')
            if os.path.exists(lset_path):
                lset = np.load(lset_path, allow_pickle=True)
            records.append({
                'dataset': dataset_name,
                'sampling_fn': sampling_fn,
                'eval_model': eval_model,
                'id': exp_id,
                'x': x_vals,
                'y': y_vals,
                'delta': initial_delta,
                'kernel_type': kernel_type,
                "diff_method": diff_method,
                "alpha": alpha
                , "cont_method": cont_method,
                "alpha_decrease" :alpha_decrease,
                "sparse_ds": sparse_ds,
                "seed": seed,
                "lset": lset

            })
# Convert to DataFrame for easier plotting/grouping

base_path = "/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets"
parquet_name = "1211_bayes_misp_CIFAR10_from_images_tophat_HIGH_BUDGET_RESULTS.parquet"
full_path = os.path.join(base_path, parquet_name)
df = pd.DataFrame(records)


final_df = df[df['sampling_fn'] != "prob_cover"]
relevant_seeds_df[relevant_seeds_df.apply(lambda row: len(row['y'].values) >= 32)]


# final_df.to_parquet(full_path)
imgs_df = df[df['eval_model'] == 'from_images']
imgs_df["unique_id"] = imgs_df.apply(lambda row: f"{row['sampling_fn']}_{row['diff_method']}_{row['diff_method']}_alpha={row['alpha']}_cont={row['cont_method']}", axis=1)
unique_exps = imgs_df['unique_id'].unique()
for exp in unique_exps:
    exp_df = imgs_df[imgs_df['unique_id'] == exp]
    exp_seeds = exp_df['seed'].unique()
    if len(exp_seeds) < 3:
        print(f"Dropping {exp} due to insufficient seeds: {exp_seeds}")
        # imgs_df = imgs_df[imgs_df['unique_id'] != exp]
imgs_df = imgs_df.drop_duplicates(subset=['unique_id', 'seed'])
imgs_df.to_parquet(full_path)


random_vals = imgs_df[imgs_df['sampling_fn'] == 'random']['y'].values[0]
imgs_df = df
imgs_df["unique_id"] = imgs_df.apply(lambda row: f"{row['sampling_fn']}_{row['diff_method']}_{row['diff_method']}_alpha={row['alpha']}_cont={row['cont_method']}", axis=1)
plt.close()
x_range = np.arange(100, 300)

for idx, row in imgs_df.iterrows():
    if row['sampling_fn'] == 'random':
        continue
    if row['sampling_fn'] == 'bayes_misp':
        if row['diff_method'] == 'max':
            if row['alpha'] not in  [0.1, 0.001]:
                continue
        else:
            if row['alpha'] not in  [0.1, 1]:
                continue
    adjusted_y = row['y'][:300] - random_vals[:min(300, len(row['y']))]
    adjusted_y_smooth = moving_average(adjusted_y, 10)
    adjusted_y_smooth_full = np.full(700, fill_value=np.nan)
    adjusted_y_smooth_full[: len(adjusted_y_smooth)] = adjusted_y_smooth
    plt.plot(adjusted_y_smooth, label=row["unique_id"])
# make legend at right side outside of plot and incerease figure size
plt.legend()
#increase plot size
plt.gcf().set_size_inches(20, 12)
plt.show()





cifar100_features__df = df[df['eval_model'] == 'from_features']
cifar100_features__df["unique_id"] = cifar100_features__df.apply(lambda row: f"{row['sampling_fn']}_{row['diff_method']}_{row['diff_method']}_alpha={row['alpha']}_cont={row['cont_method']}", axis=1)
unique_exps = cifar100_features__df['unique_id'].unique()
for exp in unique_exps:
    exp_df = cifar100_features__df[cifar100_features__df['unique_id'] == exp]
    exp_seeds = exp_df['seed'].unique()
    if len(exp_seeds) < 3:
        print(f"Dropping {exp} due to insufficient seeds: {exp_seeds}")
        # cifar100_features__df = cifar100_features__df[cifar100_features__df['unique_id'] != exp]
cifar100_features__df = cifar100_features__df.drop_duplicates(subset=['unique_id', 'seed'])
cifar100_features__df.to_parquet(full_path)






full_cifar10_df = df[df['sparse_ds'] == False]
full_cifar10_df["unique_id"] = full_cifar10_df.apply(lambda row: f"{row['sampling_fn']}_{row['diff_method']}_{row['diff_method']}_alpha={row['alpha']}_cont={row['cont_method']}", axis=1)
full_cifar10_df = full_cifar10_df[full_cifar10_df['eval_model'] == 'from_images']

relevant_seeds_df = full_cifar10_df[full_cifar10_df['seed'] <= 4]
relevant_seeds_df = relevant_seeds_df[relevant_seeds_df.apply(lambda row: len(row['y'].values) >=32)]


unique_exps = relevant_seeds_df['unique_id'].unique()
for exp in unique_exps:
    exp_df = relevant_seeds_df[relevant_seeds_df['unique_id'] == exp]
    exp_seeds = exp_df['seed'].unique()
    if len(exp_seeds) < 3:
        print(f"Dropping {exp} due to insufficient seeds: {exp_seeds}")
        relevant_seeds_df = relevant_seeds_df[relevant_seeds_df['unique_id'] != exp]

relevant_seeds_df = relevant_seeds_df.drop_duplicates(subset=['unique_id', 'seed'])

seed0_full_cifar_10_df = full_cifar10_df[full_cifar10_df['seed'] == 0]


seed0_full_cifar_10_df = seed0_full_cifar_10_df.drop_duplicates(subset=['unique_id', 'seed'])



random_vals = seed0_full_cifar_10_df[seed0_full_cifar_10_df['sampling_fn'] == 'random']['y'].values[0]

seed0_full_cifar_10_df["unique_id"] = seed0_full_cifar_10_df.apply(lambda row: f"{row['diff_method']}_{row['diff_method']}_alpha={row['alpha']}_cont={row['cont_method']}", axis=1)

for idx, row in seed0_full_cifar_10_df.iterrows():
    if row['sampling_fn'] == 'random':
        continue
    adjusted_y = row['y'][:6] - random_vals[:6]
    plt.plot(adjusted_y, label=row["unique_id"])
# make legend at right side outside of plot and incerease figure size
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#increase plot size
plt.gcf().set_size_inches(20, 12)
plt.show()
relevant_seeds_df.to_parquet(full_path)


