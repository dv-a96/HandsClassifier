import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def create_template(list_of_dfs, axis='y_sg', target_length=200):
    """
    Normalizes the signals in the list of dataframes to a common length and computes their average to create a template signal.
    """
    resampled_signals = []
    
    for df in list_of_dfs:
        signal = df[axis].values
        # Interpolation to a fixed length
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, target_length)
        f = interp1d(x_old, signal, kind='linear')
        resampled_signals.append(f(x_new))
    
    # Return the average of all normalized signalss
    return np.mean(resampled_signals, axis=0)


def extract_correlation_features(df, accel_df, left_template, right_template, hand, n_left, n_right, target_length=200):
    """Extracts correlation features from the given dataframe using the provided templates.
    Args:   df (pd.DataFrame): The input dataframe containing the gyro data.
            accel_df (pd.DataFrame): The dataframe containing the accelerometer data.
            left_template (np.array): The template signal for the left class.
            right_template (np.array): The template signal for the right class.
            target_length (int): The length to which signals will be resampled for correlation with the templates.
            hand (str): The label for the hand ('Left' or 'Right') to be used in feature extraction.
            n_left (int): The number of samples in the left class (used for template adjustment).
            n_right (int): The number of samples in the right class (used for template adjustment
    Returns:
            dict: A dictionary containing the extracted correlation features:
                - 'gyro_accel_corr': Correlation between gyro_y and accel_x.
                - 'gyro_gyro_corr': Correlation between gyro_y and gyro_x.
                - 'corr_with_right_template': Correlation of gyro_y with the right template.
                - 'corr_with_left_template': Correlation of gyro_y with the left template.
    """
    features = {}
    
    # Helper function to resample a signal to the target length
    def resample_signal(sig, length):
        if len(sig) < 2: 
            return np.zeros(length)
        x_old = np.linspace(0, 1, len(sig))
        x_new = np.linspace(0, 1, length)
        return interp1d(x_old, sig, kind='linear', fill_value="extrapolate")(x_new)

    # Preprocess gyro_y,gyro_x and accel_x by resampling to the target length
    gyro_y_resampled = resample_signal(df['y_sg'].values, target_length)
    gyro_x_resampled = resample_signal(df['x_sg'].values, target_length)
    accel_x_resampled = resample_signal(accel_df['x_sg'].values, target_length)

    # 1. gyro_accel_correlation (gyro_y vs accel_x)
    features['gyro_accel_corr'] = np.corrcoef(gyro_y_resampled, accel_x_resampled)[0, 1]

    # 2. gyro_gyro_correlation (gyro_y vs gyro_x)
    features['gyro_gyro_corr'] = np.corrcoef(gyro_y_resampled, gyro_x_resampled)[0, 1]

    # Resample gyro_y to the target length for template correlation
    y_gyro_resampled = resample_signal(df['y_sg'].values, target_length)
    
    if hand == 'Left':
        left_template = left_template * n_left
        left_template -= y_gyro_resampled
        left_template /= (n_left-1)
    elif hand == 'Right':
        right_template = right_template * n_right
        right_template -= y_gyro_resampled
        right_template /= (n_right-1)

    # 3. Correlation with Right template
    features['corr_with_right_template'] = np.corrcoef(y_gyro_resampled, right_template)[0, 1]

    # 4. Correlation with Left template
    features['corr_with_left_template'] = np.corrcoef(y_gyro_resampled, left_template)[0, 1]

    return features


def get_paired_files(directory):
    '''
    Scans the specified directory for CSV files and identifies pairs of files that correspond to the same base name but different types (accel and gyro).
    Args:    directory (str): The path to the directory containing the CSV files.
    Returns:    dict: A dictionary where each key is a base name (common part of the filename) and the value is another dictionary with keys 'accel' and 'gyro' containing the respective file paths.
    '''

    pairs = {}
    for filename in os.listdir(directory):
        if not filename.endswith('.csv'):
            continue
            
        # Extract the base name by removing the specific suffixes
        base_name = filename.replace('accel.csv', '').replace('gyro.csv', '')
        
        if base_name not in pairs:
            pairs[base_name] = {'accel': None, 'gyro': None}
        
        if 'accel' in filename:
            pairs[base_name]['accel'] = os.path.join(directory, filename)
        elif 'gyro' in filename:
            pairs[base_name]['gyro'] = os.path.join(directory, filename)
            
    # Return only those pairs that have both accel and gyro files
    return {k: v for k, v in pairs.items() if v['accel'] and v['gyro']}

# left_gyro_list = [pd.read_csv(f'New/Smoothed/Left/{path}') for path in os.listdir('New/Smoothed/Left') if path.endswith('gyro.csv')]
# n_left = len(left_gyro_list)
# right_gyro_list = [pd.read_csv(f'New/Smoothed/Right/{path}') for path in os.listdir('New/Smoothed/Right') if path.endswith('gyro.csv')]
# n_right = len(right_gyro_list)
# left_accel_list = [pd.read_csv(f'New/Smoothed/Left/{path}') for path in os.listdir('New/Smoothed/Left') if path.endswith('accel.csv')]
# right_accel_list = [pd.read_csv(f'New/Smoothed/Right/{path}') for path in os.listdir('New/Smoothed/Right') if path.endswith('accel.csv')]
# left_template = create_template(left_gyro_list, axis='y_sg', target_length=750)
# right_template = create_template(right_gyro_list, axis='y_sg', target_length=750)



# # Plot right vs left template
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.plot(left_template, label='Left Template', color='blue')
# plt.plot(right_template, label='Right Template', color='orange')
# plt.title('Left vs Right Template Signals')
# plt.xlabel('Time (normalized)')
# plt.ylabel('Gyro Signal')
# plt.legend()
# plt.show()

# left_pairs = get_paired_files('New/Smoothed/Left')
# right_pairs = get_paired_files('New/Smoothed/Right')
# # בתוך cross_corr.py - בסוף הקובץ
# all_correlation_data = []

# # עיבוד כל הזוגות (Left ו-Right)
# for hand, pairs in [('Left', left_pairs), ('Right', right_pairs)]:
#     for base_name, paths in pairs.items():
#         accel_df = pd.read_csv(paths['accel'])
#         gyro_df = pd.read_csv(paths['gyro'])
        
#         # חילוץ הפיצ'רים
#         features = extract_correlation_features(gyro_df, accel_df, left_template, right_template, hand, n_left, n_right, target_length=750)
        
#         # הוספת מזהים לאיחוד
#         features['filename_clean'] = base_name + '.csv'
#         features['label_from_corr'] = hand
#         all_correlation_data.append(features)

# בתוך cross_corr.py

def save_correlation_stats(left_pairs, right_pairs, left_template, right_template, n_left, n_right, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    for hand, pairs in [('left', left_pairs), ('right', right_pairs)]:
        data = []
        for base_name, paths in pairs.items():
            accel_df = pd.read_csv(paths['accel'])
            gyro_df = pd.read_csv(paths['gyro'])
            
            # Features extraction
            features = extract_correlation_features(gyro_df, accel_df, left_template, right_template, hand, n_left, n_right, target_length=750)
            
            # Create a row for the stats file
            row = {
                'filename': base_name,
                'axis': 'sync', # Imaginary axis for stats file, since we are not storing the raw signals here
                **features
            }
            data.append(row)
        
        # Save the stats to a CSV file
        df = pd.DataFrame(data)
        out_path = os.path.join(save_dir, f"{hand}_correlation_stats.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved correlation stats to: {out_path}")

# # קריאה לפונקציה בסוף הקובץ
# save_correlation_stats(left_pairs, right_pairs, left_template, right_template, 'New/Stats')

# # שמירה לקובץ ביניים
# corr_df = pd.DataFrame(all_correlation_data)
# corr_df.to_csv('New/correlation_features.csv', index=False)
# print("Correlation features saved to New/correlation_features.csv")

def run_permutation_test(left_pairs, right_pairs, n_permutations=100):
    # Load all data into memory
    print("Pre-loading data into memory...")
    data_cache = {}
    all_pairs_list = list(left_pairs.items()) + list(right_pairs.items())
    
    for base_name, paths in tqdm(all_pairs_list, desc="Loading CSVs"):
        data_cache[base_name] = {
            'accel': pd.read_csv(paths['accel']),
            'gyro': pd.read_csv(paths['gyro'])
        }

    # Permutation test
    all_data = []
    n_left = len(left_pairs)
    combined_items = list(left_pairs.items()) + list(right_pairs.items())
    
    pbar = tqdm(range(n_permutations + 1), desc="Permutation Progress")

    for i in pbar:
        # Decide which pairs to use for this iteration
        if i == 0:
            current_left = list(left_pairs.items())
            current_right = list(right_pairs.items())
            perm_type = "original"
        else:
            shuffled = combined_items.copy()
            random.shuffle(shuffled)
            current_left = shuffled[:n_left]
            current_right = shuffled[n_left:]
            perm_type = f"perm_{i}"

        # Template creation for the current permutation
        pbar.set_postfix({"task": "templates"})
        temp_left = create_template([data_cache[p[0]]['gyro'] for p in current_left], 'y_sg', target_length=750)
        temp_right = create_template([data_cache[p[0]]['gyro'] for p in current_right], 'y_sg', target_length=750)

        # Feature extraction for the current permutation
        pbar.set_postfix({"task": "extracting"})
        for group_name, group_data in [('left', current_left), ('right', current_right)]:
            for base_name, _ in group_data:
                accel_df = data_cache[base_name]['accel']
                gyro_df = data_cache[base_name]['gyro']
                
                features = extract_correlation_features(
                    gyro_df, accel_df, temp_left, temp_right, 
                    group_name, n_left, len(right_pairs), target_length=750
                )
                
                all_data.append({
                    'iteration': i,
                    'type': perm_type,
                    'filename': base_name,
                    'assigned_as': group_name,
                    **features
                })
                
    return pd.DataFrame(all_data)

def analyze_permutation_with_std(perm_df: pd.DataFrame):
    stat_cols = ['gyro_accel_corr', 'gyro_gyro_corr', 'corr_with_right_template', 'corr_with_left_template']
    
    # Aggregate mean and std for each iteration, type, and assigned_as
    iter_stats = perm_df.groupby(['iteration', 'type', 'assigned_as'])[stat_cols].agg(['mean', 'std'])
    
    # Define new column names for the multi-index columns
    iter_stats.columns = [f"{c[0]}_{c[1]}" for c in iter_stats.columns]
    iter_stats = iter_stats.reset_index()

    # Separate original and permutation data for comparison
    original = iter_stats[iter_stats['type'] == 'original']
    perms = iter_stats[iter_stats['type'] != 'original']
    
    final_comparison = []

    for hand in ['left', 'right']:
        for col in stat_cols:
            # The original mean and std for the current hand and feature
            orig_row = original[original['assigned_as'] == hand]
            orig_mean = orig_row[f"{col}_mean"].values[0]
            orig_std = orig_row[f"{col}_std"].values[0]
            
            # The permutation means for the current hand and feature
            perm_hand_vals = perms[perms['assigned_as'] == hand]
            perm_means = perm_hand_vals[f"{col}_mean"].values
            
            # Calculate p-value: how many permutation means are greater than or equal to the original mean?
            p_val = np.sum(perm_means >= orig_mean) / len(perm_means)
            
            final_comparison.append({
                'hand': hand,
                'feature': col,
                'orig_avg': orig_mean,      # Original average for the hand and feature
                'orig_std': orig_std,       # How homogeneous the original group was
                'perm_avg_mean': np.mean(perm_means), # Mean of the random guesses
                'p_value': p_val,
                'significant': p_val < 0.05
            })
            
    return pd.DataFrame(final_comparison)

def calculate_cohens_d(group1_data, group2_data):
    """חישוב עוצמת אפקט (Cohen's d) תוך התחשבות בסטיית התקן"""
    n1, n2 = len(group1_data), len(group2_data)
    var1, var2 = np.var(group1_data, ddof=1), np.var(group2_data, ddof=1)
    mu1, mu2 = np.mean(group1_data), np.mean(group2_data)
    
    # Pooled standard deviation calculation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0: return 0
    return (mu1 - mu2) / pooled_std

def analyze_effect_size_permutation(perm_df: pd.DataFrame):
    stat_cols = ['gyro_accel_corr', 'gyro_gyro_corr', 'corr_with_right_template', 'corr_with_left_template']
    iterations = perm_df['iteration'].unique()
    
    effect_sizes = []

    for i in iterations:
        iter_data = perm_df[perm_df['iteration'] == i]
        perm_type = iter_data['type'].iloc[0]
        
        iteration_results = {'iteration': i, 'type': perm_type}
        
        for col in stat_cols:
            right_vals = iter_data[iter_data['assigned_as'] == 'right'][col].values
            left_vals = iter_data[iter_data['assigned_as'] == 'left'][col].values
            
            # Calculate Cohen's d for the current feature and iteration
            d = calculate_cohens_d(right_vals, left_vals)
            iteration_results[col] = d
            
        effect_sizes.append(iteration_results)
    
    df_effect = pd.DataFrame(effect_sizes)
    
    # Summery
    summary = []
    for col in stat_cols:
        orig_d = df_effect[df_effect['type'] == 'original'][col].values[0]
        perm_ds = df_effect[df_effect['type'] != 'original'][col].values
        
        # Calculate the p-value based on the proportion of permuted Cohen's d values that are as extreme as or more extreme than the original Cohen's d
        p_val = np.sum(np.abs(perm_ds) >= np.abs(orig_d)) / len(perm_ds)
        
        summary.append({
            'feature': col,
            'original_cohens_d': orig_d,
            'p_value': p_val,
            'interpretation': 'Large' if abs(orig_d) > 0.8 else 'Medium' if abs(orig_d) > 0.5 else 'Small'
        })
        
    return pd.DataFrame(summary), df_effect


def plot_permutation_d_dist(df_effect, summary_results, save_path=None):
    """
    Plots the distribution of Cohen's d values from the permutation test, and annotating with p-values.
    """
    stat_cols = summary_results['feature'].tolist()
    n_features = len(stat_cols)
    
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 5 * n_features))
    if n_features == 1: axes = [axes]
    
    for i, col in enumerate(stat_cols):
        ax = axes[i]
        
        # Data for plotting
        orig_d = df_effect[df_effect['type'] == 'original'][col].values[0]
        perm_ds = df_effect[df_effect['type'] != 'original'][col]
        p_val = summary_results.loc[summary_results['feature'] == col, 'p_value'].values[0]
        interpretation = summary_results.loc[summary_results['feature'] == col, 'interpretation'].values[0]
        
        # Plotting the distribution of Cohen's d from permutations
        sns.kdeplot(perm_ds, fill=True, color="gray", alpha=0.3, ax=ax, label='Null Distribution (Shuffled)')
        sns.rugplot(perm_ds, color="gray", alpha=0.5, ax=ax) # Add rug plot for individual points
        
        # Mark the original Cohen's d
        ax.axvline(orig_d, color='red', linestyle='--', linewidth=3, 
                   label=f'Original d = {orig_d:.2f} (p={p_val:.3f})')
        
        # Desidn
        ax.set_title(f'Permutation Test for Cohen\'s d: {col.replace("_", " ").title()}', fontsize=15)
        ax.set_xlabel('Cohen\'s d Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        # Legend
        ax.legend(loc='upper right')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Permutation plot saved to: {save_path}")
    else:
        plt.show()

# permute_data = run_permutation_test(left_pairs, right_pairs, n_permutations=10000)
# print(permute_data.head())
# print(permute_data.columns)
# print(f"Permutation test completed with {len(permute_data)} rows of data.")

# summary_results, df_effect = analyze_effect_size_permutation(permute_data)
# summary_results.to_csv('New/permutation_summary.csv', index=False)
# print("Permutation summary saved to New/permutation_summary.csv")
# df_effect.to_csv('New/permutation_effect_sizes.csv', index=False) 
# print("Permutation effect sizes saved to New/permutation_effect_sizes.csv")
df_effect = pd.read_csv('New/permutation_effect_sizes.csv')
summary_results = pd.read_csv('New/permutation_summary.csv')
plot_permutation_d_dist(df_effect, summary_results, save_path='New/permutation_effect_size_distribution.png')