import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def reshape_sensor_data(df):
    """
    Combine features from all axes (x, y, z) into a single row per file.
    """
    # Define the statistical metrics to be extracted
    stats_cols = ['mean', 'variance', 'min', 'max', 'median', 
                  'delta_min_max', 'count_negative', 'count_positive', 'intensity', 'skewness',
                  'argmax', 'argmin', 'zcr']
    
    # Pivot the table: Transform axis values (rows) into new columns
    # This organizes the data so each unique file has exactly one row
    pivoted = df.pivot_table(index=['filename', 'label'], columns='axis', values=stats_cols)
    
    # Flatten MultiIndex columns: e.g., ('mean', 'x') becomes 'x_mean'
    pivoted.columns = [f"{axis}_{metric}" for metric, axis in pivoted.columns]
    
    return pivoted.reset_index()

def create_full_feature_matrix(accel_stats_df, gyro_stats_df):
    """
    Merge the processed Accel and Gyro DataFrames based on filename.
    """            
    # Reshape both DataFrames to a flat format
    accel_flat = reshape_sensor_data(accel_stats_df)
    gyro_flat = reshape_sensor_data(gyro_stats_df)
    
    # 1. Standardize filename strings for the Accelerometer table
    # Removing the 'accel' suffix to allow matching with Gyro files
    accel_flat['filename_clean'] = accel_flat['filename'].str.replace('accel', '', case=False)

    # 2. Standardize filename strings for the Gyroscope table
    gyro_flat['filename_clean'] = gyro_flat['filename'].str.replace('gyro', '', case=False)

    # 3. Merge the two sensors into a single master DataFrame
    # Using suffixes to distinguish between same-named metrics (e.g., intensity_accel vs intensity_gyro)
    final_df = pd.merge(accel_flat, gyro_flat, on='filename_clean', suffixes=('_accel', '_gyro'))
    
    # 4. Define the Target Label (Hand Classification)
    # Extracts the hand side from the filename for supervised learning
    final_df['label'] = final_df['label_accel'].apply(
        lambda x: 1 if 'right' in x.lower() else 0 # 1 represents Right Hand, 0 represents Left Hand
    )
    
    return final_df

def load_feture_matrix(features_path, save_path=None):
    """Load the different sensors and axes features from the CSV files."""

    left_accel_df = pd.read_csv(f"{features_path}/left_accel_stats.csv")
    right_accel_df = pd.read_csv(f"{features_path}/right_accel_stats.csv")
    left_gyro_df = pd.read_csv(f"{features_path}/left_gyro_stats.csv")
    right_gyro_df = pd.read_csv(f"{features_path}/right_gyro_stats.csv")

    # Add a 'label' column to each DataFrame
    left_accel_df['label'] = 'Left'
    right_accel_df['label'] = 'Right'
    left_gyro_df['label'] = 'Left'
    right_gyro_df['label'] = 'Right'

    left_features = create_full_feature_matrix(left_accel_df, left_gyro_df)
    right_features = create_full_feature_matrix(right_accel_df, right_gyro_df)

    # Combine Left and Right hand features into a single DataFrame
    full_features = pd.concat([left_features, right_features], ignore_index=True)
    if save_path:
        full_features.to_csv(save_path, index=False)
    return full_features



def plot_feature_correlation(df):
    # 1. נחשב את המטריצה רק עבור עמודות מספריות (ללא ה-filename)
    # נניח ש-df הוא ה-final_df שלך אחרי האיחוד
    # convert label to numeric for correlation calculation
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()

    # 2. נצייר Heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    
    plt.title('Feature Correlation Matrix (Redundancy Check)')
    plt.show()
    
    return corr_matrix


def get_top_correlations_with_target(corr_matrix, target_col='label'):
    # מוציאים את הטור של הלייבל וממיינים לפי ערך מוחלט
    target_corr = corr_matrix[target_col].drop(target_col) # נוריד את הקורלציה של הלייבל עם עצמו
    target_corr_sorted = target_corr.abs().sort_values(ascending=False)
    
    print("Top Features Correlated with Target (Hand):")
    print(target_corr_sorted.head(10))
    
    return target_corr_sorted

import pandas as pd
import numpy as np

def smart_feature_selection(df, target_col='label', threshold=0.95):
    """
    Performs feature selection by removing redundant features.
    If two features are highly correlated, the one with the lower 
    correlation to the target (label) is dropped.
    
    Parameters:
    - df (pd.DataFrame): The full feature matrix.
    - target_col (str): The name of the label column.
    - threshold (float): Correlation threshold (e.g., 0.95).
    """
    # 1. Ensure the target column is numeric (0 and 1)
    # This prevents errors during correlation calculation
    if df[target_col].dtype == 'object':
        mapping = {val: i for i, val in enumerate(df[target_col].unique())}
        df[target_col] = df[target_col].map(mapping)
        print(f"Mapped {target_col} values: {mapping}")

    # 2. Select only numeric columns to avoid "ValueError: could not convert string to float"
    # This automatically excludes columns like 'filename'
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 3. Calculate the absolute correlation matrix
    corr_matrix = numeric_df.corr().abs()
    
    # 4. Get correlation of all features with the target (label)
    # We drop the target itself to avoid comparing it to itself
    target_corr = corr_matrix[target_col].drop(target_col)
    
    # 5. Mask the upper triangle of the correlation matrix
    # This ensures we only check each pair once and ignore the diagonal (1.0)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    
    # 6. Iterate through the matrix to find highly correlated pairs
    for col in upper.columns:
        if col == target_col:
            continue
            
        # Find features that correlate with the current 'col' above the threshold
        correlated_features = upper.index[upper[col] > threshold].tolist()
        
        for feature in correlated_features:
            if feature == target_col:
                continue
                
            # Compare which feature has a higher correlation with the target
            # Keep the "stronger" one, drop the "weaker" one
            if target_corr[col] > target_corr[feature]:
                to_drop.add(feature)
            else:
                to_drop.add(col)
    
    # 7. Drop the identified columns from the ORIGINAL dataframe
    final_df = df.drop(columns=list(to_drop))
    
    print(f"Smart Selection Complete.")
    print(f"Removed {len(to_drop)} redundant features.")
    print(f"Final feature count: {final_df.shape[1] - 2}") # Subtracting label and filename
    
    return final_df
df = load_feture_matrix('New/Stats', save_path='New/full_features.csv')
cor_matrix = plot_feature_correlation(df)
top_features = get_top_correlations_with_target(cor_matrix)
selected_df = smart_feature_selection(df, target_col='label', threshold=0.80)
selected_df.to_csv('New/selected_features.csv', index=False)
print(selected_df['label'].value_counts())