import os
import numpy as np
import pandas as pd
from scipy import stats
import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from cross_corr import create_template

def run_permutation_test(left_gyro_list, right_gyro_list, n_permutations=1000, target_length=750):
    """
    מבצע Permutation Test כדי לבדוק מובהקות של ההפרדה בין יד ימין ליד שמאל.
    """
    # 1. הכנת הדאטה המקורי (איחוד כל הדגימות)
    all_samples = left_gyro_list + right_gyro_list
    n_left = len(left_gyro_list)
    
    def get_t_stat(group_a, group_b):
        # יצירת תבניות
        temp_a = create_template(group_a, target_length=target_length)
        temp_b = create_template(group_b, target_length=target_length)
        
        # חישוב קורלציות עבור כל דגימה מול שתי התבניות
        # נבדוק כאן את ה"הפרש" בקורלציות כמדד ליכולת הפרדה
        corrs_a = [np.corrcoef(resample_signal(df['y_sg'].values, target_length), temp_a)[0,1] for df in group_a]
        corrs_b = [np.corrcoef(resample_signal(df['y_sg'].values, target_length), temp_b)[0,1] for df in group_b]
        
        # חישוב t-test בין שתי קבוצת הקורלציות
        t_stat, _ = stats.ttest_ind(corrs_a, corrs_b, equal_var=False)
        return t_stat

    # חישוב ה-t-statistic המקורי (האמיתי)
    print("Calculating original t-statistic...")
    t_original = get_t_stat(left_gyro_list, right_gyro_list)
    
    t_distribution = []
    
    print(f"Starting {n_permutations} permutations...")
    for i in range(n_permutations):
        # שלב 1: ערבוב תיוגים (Shuffle)
        shuffled = all_samples.copy()
        random.shuffle(shuffled)
        
        # פיצול לקבוצות אקראיות באותו גודל
        perm_left = shuffled[:n_left]
        perm_right = shuffled[n_left:]
        
        # שלבים 2-5: חישוב t-score לקבוצות המעורבבות
        t_perm = get_t_stat(perm_left, perm_right)
        t_distribution.append(t_perm)
        
        if (i+1) % 100 == 0:
            print(f"Iteration {i+1}/{n_permutations} completed.")

    # שלב 6: חישוב P-value
    # כמה פעמים קיבלנו בטעות t-score גבוה יותר מהמקורי?
    p_value = np.sum(np.abs(t_distribution) >= np.abs(t_original)) / n_permutations
    
    return t_original, t_distribution, p_value



def run_permutation_test_b(left_gyro_list, right_gyro_list, n_permutations=1000, target_length=750):
    """
    מבצע Permutation Test כדי לבדוק מובהקות של ההפרדה בין יד ימין ליד שמאל.
    """
    # 1. הכנת הדאטה המקורי (איחוד כל הדגימות)
    all_samples = left_gyro_list + right_gyro_list
    n_left = len(left_gyro_list)
    
    def get_t_stat(group_a, group_b):
        # יצירת תבניות
        temp_a = create_template(group_a, target_length=target_length)
        temp_b = create_template(group_b, target_length=target_length)
        
        # חישוב קורלציות עבור כל דגימה מול שתי התבניות
        # נבדוק כאן את ה"הפרש" בקורלציות כמדד ליכולת הפרדה
        corrs_a = [np.corrcoef(resample_signal(df['y_sg'].values, target_length), temp_a)[0,1] for df in right_gyro_list]
        corrs_b = [np.corrcoef(resample_signal(df['y_sg'].values, target_length), temp_b)[0,1] for df in left_gyro_list]
        
        # חישוב t-test בין שתי קבוצת הקורלציות
        t_stat, _ = stats.ttest_ind(corrs_a, corrs_b, equal_var=False)
        return t_stat

    # חישוב ה-t-statistic המקורי (האמיתי)
    print("Calculating original t-statistic...")
    t_original = get_t_stat(left_gyro_list, right_gyro_list)
    
    t_distribution = []
    
    print(f"Starting {n_permutations} permutations...")
    for i in range(n_permutations):
        # שלב 1: ערבוב תיוגים (Shuffle)
        shuffled = all_samples.copy()
        random.shuffle(shuffled)
        
        # פיצול לקבוצות אקראיות באותו גודל
        perm_left = shuffled[:n_left]
        perm_right = shuffled[n_left:]
        
        # שלבים 2-5: חישוב t-score לקבוצות המעורבבות
        t_perm = get_t_stat(perm_left, perm_right)
        t_distribution.append(t_perm)
        
        if (i+1) % 100 == 0:
            print(f"Iteration {i+1}/{n_permutations} completed.")

    # שלב 6: חישוב P-value
    # כמה פעמים קיבלנו בטעות t-score גבוה יותר מהמקורי?
    p_value = np.sum(np.abs(t_distribution) >= np.abs(t_original)) / n_permutations
    
    return t_original, t_distribution, p_value


# פונקציית עזר לעיבוד סיגנל (נדרשת עבור הלולאה)
def resample_signal(sig, length):
    if len(sig) < 2: return np.zeros(length)
    x_old = np.linspace(0, 1, len(sig))
    x_new = np.linspace(0, 1, length)
    return interp1d(x_old, sig, kind='linear', fill_value="extrapolate")(x_new)

# הרצה
left_gyro_list = [pd.read_csv(f'New/Smoothed/Left/{path}') for path in os.listdir('New/Smoothed/Left') if path.endswith('gyro.csv')]
right_gyro_list = [pd.read_csv(f'New/Smoothed/Right/{path}') for path in os.listdir('New/Smoothed/Right') if path.endswith('gyro.csv')]
t_orig, t_dist, p_val = run_permutation_test(left_gyro_list, right_gyro_list)

print(f"\nResults:")
print(f"Original T-Statistic: {t_orig:.4f}")
print(f"P-value: {p_val:.4f}")

# ויזואליזציה של התוצאה
plt.figure(figsize=(10, 6))
plt.hist(t_dist, bins=30, alpha=0.7, label='Null Hypothesis Distribution')
plt.axvline(t_orig, color='red', linestyle='--', label=f'Original T-stat (p={p_val:.4f})')
plt.title('Permutation Test Results')
plt.xlabel('T-statistic value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

t_orig_b, t_dist_b, p_val_b = run_permutation_test_b(left_gyro_list, right_gyro_list)

print(f"\nResults:")
print(f"Original T-Statistic: {t_orig_b:.4f}")
print(f"P-value: {p_val_b:.4f}")

# ויזואליזציה של התוצאה
plt.figure(figsize=(10, 6))
plt.hist(t_dist_b, bins=30, alpha=0.7, label='Null Hypothesis Distribution')
plt.axvline(t_orig_b, color='red', linestyle='--', label=f'Original T-stat (p={p_val_b:.4f})')
plt.title('Permutation Test Results')
plt.xlabel('T-statistic value')
plt.ylabel('Frequency')
plt.legend()
plt.show()