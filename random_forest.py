import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

def train_hand_classifier(features_path):
    # 1. Load the dataset
    # Expects a table containing feature columns (mean, variance, intensity, etc.)
    # and a 'label' column (Target variable)
    df = pd.read_csv(features_path)
    
    # Remove identification columns that should not be used as features for the model
    X = df.drop(columns=['label_accel', 'label_gyro', 'filename_clean', 'filename_accel', 'filename_gyro'], errors='ignore')
    y = df['label_accel']
    
    # 2. Encode the Target (e.g., 'Left'/'Right' -> 0/1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 3. Split into Training and Testing sets
    # 'stratify' ensures the ratio of Left/Right hands remains consistent in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 4. Model Initialization and Training
    # n_estimators: Number of trees in the forest (100 is a standard baseline)
    # n_jobs=-1: Uses all available CPU cores for faster training
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # 5. Model Evaluation
    y_pred = rf_model.predict(X_test)
    
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Display the Confusion Matrix
    # Helps visualize where the model confuses 'Left' with 'Right'
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Confusion Matrix: Left vs Right')
    plt.show()
    
    # 6. Feature Importance Visualization
    # Analyzes which features (e.g., Intensity, Mean) had the most impact on the classification
    importances = rf_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15))
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.show()
    
    return rf_model, le


def reshape_sensor_data(df):
    """
    Step 1: Combine features from all axes (x, y, z) into a single row per file.
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

def create_final_feature_matrix(accel_stats_df, gyro_stats_df):
    """
    Step 2: Merge the processed Accel and Gyro DataFrames based on filename.
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

# Execute the training process
# train_hand_classifier('master_features.csv')