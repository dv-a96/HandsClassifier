import pandas as pd

def reshape_sensor_data(df):
    """
    Combine features from all axes (x, y, z) into a single row per file.
    """
    # Define the statistical metrics to be extracted
    stats_cols = ['mean', 'std', 'variance', 'min', 'max', 'median', 
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
    final_df['label'] = final_df['filename_clean'].apply(
        lambda x: 1 if 'right' in x.lower() else 0 # 1 represents Right Hand, 0 represents Left Hand
    )
    
    return final_df

def load_feture_matrix(features_path):
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
    return full_features