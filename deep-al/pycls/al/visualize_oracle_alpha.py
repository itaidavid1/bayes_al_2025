import sys
import os
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_arrays(series_of_arrays):
    """
    Takes a pandas Series where each element is a numpy array (or list).
    Stacks them vertically (after trimming to the minimum length) and computes
    the mean and standard deviation across the seed dimension.
    Returns (mean_array, std_array, trimmed_length).
    """
    arrays = [np.array(arr) for arr in series_of_arrays.values]
    min_len = min(len(arr) for arr in arrays)
    trimmed = [arr[:min_len] for arr in arrays]
    matrix = np.vstack(trimmed)
    mean_arr = np.mean(matrix, axis=0)
    std_arr = np.std(matrix, axis=0)
    return mean_arr, std_arr, min_len


def prepare_single_alpha_df(df, alpha_column='alpha'):
    """
    Prepare dataframe for single alpha plotting by aggregating over the alpha column.

    Args:
        df: DataFrame with alpha_column, 'x', 'y' columns (and optionally 'weight_method', 'sampling_fn')
        alpha_column: Name of the alpha column to group by

    Returns:
        Aggregated dataframe ready for plotting
    """
    def aggregate_group(group):
        mean_y, std_y, min_len = aggregate_arrays(group['y'])
        x_vals = np.array(group['x'].iloc[0])[:min_len]
        result = {'y': mean_y, 'y_std': std_y, 'x': x_vals}
        # Preserve other columns if they exist
        for col in ['weight_method', 'sampling_fn']:
            if col in group.columns:
                result[col] = group[col].iloc[0]
        return pd.Series(result)

    return df.groupby([alpha_column]).apply(aggregate_group).reset_index()

def basic_single_alphas_plot(dfs, df_names, ref_df, target_steps, alpha_column='alpha'):
    """
    Plot accuracy vs alpha for multiple dataframes across different steps.

    Args:
        dfs: List of DataFrames with raw data (must have alpha_column, 'x', 'y' columns)
             Multiple rows per alpha value (one per seed) will be aggregated
        df_names: List of names/labels for each dataframe (must match length of dfs)
        ref_df: Reference dataframe with same structure
        target_steps: List of step values to plot
        alpha_column: Name of the column containing alpha values (default: 'alpha')
    """
    if len(dfs) != len(df_names):
        raise ValueError(f"Length of dfs ({len(dfs)}) must match length of df_names ({len(df_names)})")
    
    # Aggregate data using aggregate_arrays to compute mean and std across seeds
    dfs_agg = [prepare_single_alpha_df(df, alpha_column) for df in dfs]
    
    if ref_df.empty:
        ref_agg = pd.DataFrame()
    else:
        ref_agg = prepare_single_alpha_df(ref_df, alpha_column)

    # Extract reference method name
    ref_method_name = 'prob_cover'

    # Define colors and markers for plotting (cycle through if more dfs than colors)
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

    # Create subplots - each step in a subplot
    fig, axes = plt.subplots(len(target_steps), 1, figsize=(10, 5 * len(target_steps)))
    if len(target_steps) == 1:
        axes = [axes]  # Make it iterable

    for i, step in enumerate(target_steps):
        # Collect data for all dataframes at this step
        all_df_data = []
        all_alpha_vals_numeric_set = set()
        
        for df_idx, df_agg in enumerate(dfs_agg):
            alpha_vals = []
            y_vals = []
            y_std_vals = []
            
            for _, row in df_agg.iterrows():
                x_array = np.array(row['x'])
                indices = np.where(x_array == step)[0]
                if len(indices) > 0:
                    idx = indices[0]
                    alpha_vals.append(row[alpha_column])
                    y_vals.append(row['y'][idx])
                    y_std_vals.append(row['y_std'][idx])
            
            all_df_data.append({
                'alpha_vals': alpha_vals,
                'y_vals': y_vals,
                'y_std_vals': y_std_vals
            })
            all_alpha_vals_numeric_set.update(alpha_vals)

        # Extract reference value at this step (horizontal line)
        ref_y_val = None
        ref_y_std_val = None
        # Take the first row's value at this step (should be constant across alpha for prob_cover)
        if not ref_agg.empty:
            row = ref_agg.iloc[0]
            x_array = np.array(row['x'])
            indices = np.where(x_array == step)[0]
            if len(indices) > 0:
                idx = indices[0]
                ref_y_val = row['y'][idx]
                ref_y_std_val = row['y_std'][idx]

        # Combine all unique alpha values and sort them
        all_alpha_vals_numeric = sorted(all_alpha_vals_numeric_set)
        # Convert to string labels
        alpha_labels = [str(alpha) for alpha in all_alpha_vals_numeric]
        # Create mapping for categorical positions
        alpha_to_position = {alpha: i for i, alpha in enumerate(all_alpha_vals_numeric)}

        # Plot each dataframe with error bars
        for df_idx, data in enumerate(all_df_data):
            if data['alpha_vals']:
                x_positions = [alpha_to_position[alpha] for alpha in data['alpha_vals']]
                color = colors[df_idx % len(colors)]
                marker = markers[df_idx % len(markers)]
                
                axes[i].errorbar(x_positions, data['y_vals'], yerr=data['y_std_vals'],
                                 marker=marker, linestyle='-', linewidth=2, markersize=8,
                                 capsize=5, capthick=2, label=df_names[df_idx], color=color)

        # Plot reference as horizontal line spanning the full width
        if ref_y_val is not None and alpha_labels:
            # Span from first to last categorical position
            axes[i].axhline(y=ref_y_val, linestyle='--', linewidth=2.5,
                            color='red', alpha=0.8, label=ref_method_name,
                            xmin=0, xmax=1)  # Full width in axes coordinates
            # Add shaded region for std spanning full width
            if ref_y_std_val is not None:
                axes[i].fill_between(range(len(alpha_labels)),
                                     ref_y_val - ref_y_std_val,
                                     ref_y_val + ref_y_std_val,
                                     color='red', alpha=0.15)

        # Set categorical x-axis labels
        if alpha_labels:
            axes[i].set_xticks(range(len(alpha_labels)))
            axes[i].set_xticklabels(alpha_labels, rotation=45, ha='right')

        axes[i].legend(loc='best')
        axes[i].set_title(f'Step {step}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(alpha_column, fontsize=10)
        axes[i].set_ylabel('Accuracy', fontsize=10)
        axes[i].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


high_budget_ref_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/0701_CIFAR100_HIGH_BUDGET_all_exps.parquet'
hb_df = pd.read_parquet(high_budget_ref_path)
hb_ref_df = hb_df[(hb_df['eval_model'] == 'from_images') & (hb_df['sampling_fn'].isin(['dcom']))]
df1 = hb_df[hb_df['alpha_upper_bound'] == hb_df['alpha_lower_bound']]
df1['alpha'] = df1['alpha_lower_bound']
dfs_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/1201_CIFAR100_IMAGES_high_budget_WEIGHTED_EQUAL_FIXED.parquet'
hb_dfs = pd.read_parquet(dfs_path)
df1 = hb_dfs[hb_dfs['weight_method']=='weighted']
df2 = hb_dfs[hb_dfs['weight_method']=='equal']
# df2 = hb_dfs[hb_dfs['weight_method']=='equal']
# basic_single_alphas_plot([df1, df2], ['weighted', 'equal'], hb_ref_df, [0, 30, 90, 120, 150, 180, 210, 300])

### images

df_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/1201_CIFAR100_IMAGES_low_budget_WEIGHTED_EQUAL_FIXED.parquet'
# df1_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/2612_CIFAR100_from_IMAGES_sparsity_alphas_Exp_LOW_BUDGET_ORACLE_ENTROPY_comparison_FIX.parquet'
# df2_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/0601_CIFAR100_from_IMAGES_wide_alphas_Exp_LOW_BUDGET_EQUAL_CONT_comparison_FIX.parquet'
ref_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/2312_CIFAR100_from_IMAGES_sparsity_Exp_LOW_BUDGET_comparison_FIX.parquet'

df = pd.read_parquet(df_path)
df1 = df[df['weight_method']=='weighted']
df2 = df[df['weight_method']=='equal']
ref_df = pd.read_parquet(ref_path)
ref_df = ref_df[(ref_df['eval_model'] == 'from_images') & (ref_df['sampling_fn'].isin(['prob_cover']))]
# basic_single_alphas_plot([df1, df2], ['weighted', 'equal'], ref_df, [0, 3, 9, 15, 21, 30])

### features
ref_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/2512_CIFAR100_from_FEATURES_sparsity_Alphas_Exp_LOW_BUDGET_comparison_FIX.parquet'
df1_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/0601_CIFAR100_from_FEATURES_more_wider_alphas_Exp_LOW_BUDGET_WEIGHTED_CONT_comparison_FIX.parquet'
df1_1_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/0601_CIFAR100_from_FEATURES_wider_alphas_Exp_LOW_BUDGET_WEIGHTED_CONT_comparison_FIX.parquet'
df1_2_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/0601_CIFAR100_from_FEATURES_wider_alphas_different_sigmas_Exp_LOW_BUDGET_WEIGHTED_CONT_comparison_FIX.parquet'
df1_3_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/1201_CIFAR100_FEATURES_low_budget_different_class_weights_EQUAL_FIXED.parquet'
df2_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/0601_CIFAR100_from_FEATURES_wide_alphas_Exp_LOW_BUDGET_EQUAL_CONT_comparison_FIX.parquet'
df3_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/1201_CIFAR100_low_budget_class_weights_methods.parquet'
df3_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/1201_CIFAR100_FEATURES_low_budget_different_class_weights_EQUAL_FIXED.parquet'


df1_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/1401_CIFAR100_FEATURES_LOW_budget_tophat_local_alpha_Exp.parquet'


# df1 = pd.concat(pd.read_parquet(df1_2_path)), ignore_index=True)
df1 = pd.read_parquet(df1_path)
# df1 = df1[df1['sigma'] == 1.0]
#
# df1 = df1[df1['alpha_upper_bound'] == df1['alpha_lower_bound']]
# df1['alpha'] = df1['alpha_lower_bound']
df2 = pd.read_parquet(df2_path)

df3 = pd.read_parquet(df3_path)
unique_class_weights = sorted(df3['class_weighting_func'].unique())
df3_list = [df3[df3['class_weighting_func'] == class_func] for class_func in unique_class_weights]
ref_df = pd.read_parquet(ref_path)
ref_df = ref_df[(ref_df['eval_model'] == 'from_features') & (ref_df['sampling_fn'].isin(['prob_cover']))& (ref_df['delta'] ==0.65)]
# basic_single_alphas_plot([df1, df2] + df3_list, ['weighted', 'equal'] + unique_class_weights, ref_df, [0, 3, 9, 15, 21, 30])


eval_mode = 'images'  # 'features' or 'images'
budget = 'low'
ref_sampling_functions = []


if eval_mode == 'features':
    # ref_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/ref_exps/cifar100_features_ref_df.parquet'
    ref_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/2512_CIFAR100_from_FEATURES_sparsity_Alphas_Exp_LOW_BUDGET_comparison_FIX.parquet'
    # df_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/1112_CIFAR100_from_features_oracle_local_sparsity_comparison_v2.parquet'
    # df_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/2512_CIFAR100_from_FEATURES_sparsity_Alphas_Exp_LOW_BUDGET_comparison_FIX.parquet'

    # df_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/2612_CIFAR100_from_FEATURES_sparsity_alphas_Exp_LOW_BUDGET_ORACLE_MAX_comparison_FIX.parquet'
    # df_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/0601_CIFAR100_from_FEATURES_more_wider_alphas_Exp_LOW_BUDGET_WEIGHTED_CONT_comparison_FIX.parquet'
    df_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/1401_CIFAR100_FEATURES_LOW_budget_tophat_local_alpha_Exp.parquet'
    df = pd.read_parquet(df_path)
    df = df[df['sampling_fn'] == 'bayes_misp']
    ref_df = pd.read_parquet(ref_path)
    target_steps = [0, 3, 9, 15, 21, 30]
    ref_sampling_functions = ['prob_cover', "max_herding"]
    ref_df = ref_df[(ref_df['eval_model'] == 'from_features') & (ref_df['sampling_fn'].isin(ref_sampling_functions))& (ref_df['delta'] ==0.65)]
elif eval_mode == 'images':
    if budget =='low':
        # ref_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/ref_exps/cifar100_images_ref_df.parquet'
        ref_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/2312_CIFAR100_from_IMAGES_sparsity_Exp_LOW_BUDGET_comparison_FIX.parquet'

        ref_df = pd.read_parquet(ref_path)
        # df_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/1112_CIFAR100_from_IAMGES_oracle_local_sparsity_comparison_v2.parquet'
        # df_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/1512_CIFAR100_from_IMAGES_oracle_local_sparsity_LOW_BUDGET_comparison.parquet'
        # df_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/2412_CIFAR100_from_IMAGES_sparsity_Alphas_Exp_LOW_BUDGET_comparison_FIX.parquet'
        # df_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/2612_CIFAR100_from_IMAGES_sparsity_Alphas_Exp_LOW_BUDGET_ORACLE_MAX_comparison_FIX.parquet'


        # df_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/2612_CIFAR100_from_IMAGES_sparsity_alphas_Exp_LOW_BUDGET_ORACLE_ENTROPY_comparison_FIX.parquet'
        df_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/1401_CIFAR100_IMAGES_LOW_budget_tophat_local_alpha_Exp.parquet'
        df = pd.read_parquet(df_path)

        unique_sparsity_vals = sorted(df['sparse_K_threshold'].unique())
        sparse_val = unique_sparsity_vals[0]
        df = df[df['sparse_K_threshold'] == sparse_val]

        target_steps = [0, 3, 9, 15, 21, 30]
        ref_sampling_functions = ['prob_cover', 'max_herding']
        ref_df = ref_df[(ref_df['eval_model'] == 'from_images') & (ref_df['sampling_fn'].isin(ref_sampling_functions))]
    elif budget =='high':
        # exp_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/1712_CIFAR100_from_IMAGES_oracle_local_sparsity_HIGH_BUDGET_comparison_v2.parquet'
        exp_path = '/cs/labs/daphna/itai.david/py_repos/TypiClust/stats_parquets/1912_CIFAR100_from_IMAGES_oracle_local_sparsity_HIGH_BUDGET_comparison_v3_FIX.parquet'
        df = pd.read_parquet(exp_path)
        ref_sampling_functions = ['dcom']
        ref_df = df[df['sampling_fn'].isin(ref_sampling_functions)]
        df = df[df['sampling_fn']=='bayes_misp']
        target_steps = [50, 100, 150, 200, 250, 300, 350, 400]
        ref_df = ref_df[(ref_df['eval_model'] == 'from_images') & (ref_df['sampling_fn'].isin(ref_sampling_functions))]


def compute_reference_step_stats(ref_df, sampling_functions, steps):
    """
    Build mapping per sampling function to mean/std accuracy for each target step.
    """
    ref_stats = {}
    for sampling_fn in sampling_functions:
        subset = ref_df[ref_df['sampling_fn'] == sampling_fn]
        if subset.empty or subset['x'].empty:
            continue
        mean_curve, std_curve, ref_len = aggregate_arrays(subset['y'])
        ref_steps = np.array(subset['x'].iloc[0])[:ref_len]
        step_stats = {}
        for step in steps:
            indices = np.where(ref_steps == step)[0]
            if len(indices) > 0:
                idx = indices[0]
                step_stats[step] = (mean_curve[idx], std_curve[idx])
            else:
                step_stats[step] = (np.nan, np.nan)
        ref_stats[sampling_fn] = step_stats
    return ref_stats


def aggregate_group(group):
    mean_y, std_y, min_len = aggregate_arrays(group['y'])
    x_vals = np.array(group['x'].iloc[0])[:min_len]
    return pd.Series({'y': mean_y, 'y_std': std_y, 'x': x_vals})


# Group by the parameters and aggregate both y (trimmed mean) and x (aligned steps)
df_agg = df.groupby(['alpha_lower_bound', 'alpha_upper_bound']).apply(aggregate_group).reset_index()


df = df_agg
default_missing_value = np.nan  # Fill for missing alpha combinations

# All alpha grid values (used to enforce a complete grid per plot)
all_alpha_lower = sorted(df['alpha_lower_bound'].unique())
all_alpha_upper = sorted(df['alpha_upper_bound'].unique())

# Precompute reference mean accuracy per target step for each sampling function
ref_step_stats = compute_reference_step_stats(ref_df, ref_sampling_functions, target_steps)


plt.close()
# Setup the figure: grid size adapts to target_steps length
n_steps = len(target_steps)
cols = int(np.ceil(np.sqrt(n_steps)))
rows = int(np.ceil(n_steps / cols))
fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
axes = np.array(axes).reshape(-1)

# Helper list to collect all values for global vmin/vmax calculation
all_extracted_values = []

# First pass: Collect all data needed for plotting to determine color scale
plot_data_map = {}  # Store extracted dataframes to avoid re-processing

for step in target_steps:
    extracted_rows = []
    for _, row in df.iterrows():
        # Find the index in array 'x' that matches the current 'step'
        # np.where returns indices where the condition is true
        indices = np.where(np.array(row['x']) == step)[0]

        if len(indices) > 0:
            idx = indices[0]
            acc_val = row['y'][idx]
            acc_std = row['y_std'][idx]
            extracted_rows.append({
                'alpha_lower_bound': row['alpha_lower_bound'],
                'alpha_upper_bound': row['alpha_upper_bound'],
                'y': acc_val,
                'y_std': acc_std
            })
            all_extracted_values.append(acc_val)

    plot_data_map[step] = pd.DataFrame(extracted_rows)

# Determine shared color limits
vmin = min(all_extracted_values)
vmax = max(all_extracted_values)
# Second pass: Generate the plots
for i, step in enumerate(target_steps):
    step_df = plot_data_map[step]

    if not step_df.empty:
        # Create matrix for heatmap
        heatmap_data = step_df.pivot(index="alpha_lower_bound", columns="alpha_upper_bound", values="y")
        std_data = step_df.pivot(index="alpha_lower_bound", columns="alpha_upper_bound", values="y_std")
        # Reindex to ensure full grid; fill missing combinations
        heatmap_data = heatmap_data.reindex(index=all_alpha_lower, columns=all_alpha_upper)
        std_data = std_data.reindex(index=all_alpha_lower, columns=all_alpha_upper)
        heatmap_data = heatmap_data.fillna(default_missing_value)
        std_data = std_data.fillna(default_missing_value)

        # Build annotation text with mean ± std
        annot = heatmap_data.copy()
        for r in heatmap_data.index:
            for c in heatmap_data.columns:
                mean_val = heatmap_data.loc[r, c]
                std_val = std_data.loc[r, c]
                annot.loc[r, c] = "" if pd.isna(mean_val) else f"{mean_val:.2f}±{std_val:.2f}"

        # Plot with shared color scale
        sns.heatmap(
            heatmap_data,
            ax=axes[i],
            annot=annot,
            fmt="",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            annot_kws={"fontsize": 8},
        )

        axes[i].set_title(f'Iteration (Step): {step}')
        axes[i].set_xlabel('alpha_upper_bound')
        axes[i].set_ylabel('alpha_lower_bound')
        ref_text_lines = []
        for ref_name, step_stats in ref_step_stats.items():
            ref_val, ref_std = step_stats.get(step, (np.nan, np.nan))
            if not np.isnan(ref_val):
                ref_text_lines.append(f'{ref_name} mean @ step {step}: {ref_val:.3f}±{ref_std:.3f}')
        if ref_text_lines:
            axes[i].text(0.5, 1.1,
                         "\n".join(ref_text_lines),
                         transform=axes[i].transAxes,
                         ha='center', va='center', fontsize=10, color='darkred')

# Hide any unused subplots
for j in range(len(target_steps), len(axes)):
    axes[j].axis('off')

plt.suptitle('Accuracy Evolution Over Training Steps', fontsize=16)
plt.tight_layout()
plt.show()

