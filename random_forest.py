import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

def train_hand_classifier(features_path):
    # 1. טעינת הנתונים
    # נניח שהטבלה מכילה עמודות של פיצ'רים (mean_avg, variance_avg וכו') 
    # ועמודה של 'hand' (המטרה שלנו - Target)
    df = pd.read_csv(features_path)
    
    # הסרת עמודות מזהות שלא אמורות להיכנס למודל
    X = df.drop(columns=['hand', 'filename'], errors='ignore')
    y = df['label']  # עמודת המטרה (0/1) שהוספנו בשלב יצירת הטבלה הסופית
    
    # 2. קידוד המטרה (Left/Right -> 0/1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 3. פיצול לסט אימון וסט מבחן
    # מומלץ להשתמש ב-stratify כדי לשמור על יחס שווה של ידיים בשני הסטים
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 4. יצירת המודל ואימון
    # n_estimators: מספר העצים (100 זה סטנדרט טוב להתחלה)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # 5. הערכת המודל
    y_pred = rf_model.predict(X_test)
    
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # הצגת מטריצת בלבול (Confusion Matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Confusion Matrix: Left vs Right')
    plt.show()
    
    # 6. הצגת חשיבות הפיצ'רים (Feature Importance)
    # זה השלב הכי מעניין למחקר שלך - איזה פיצ'ר הכי משפיע על הזיהוי?
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
    # Define the statistical columns we want to keep
    stats_cols = ['mean', 'std', 'variance', 'min', 'max', 'median', 
                  'delta_min_max', 'count_negative', 'count_positive', 'intensity']
    
    # Pivot the table: Index stays as filename, Axis values become new columns
    # We use pivot_table to handle the reorganization efficiently
    pivoted = df.pivot_table(index='filename', columns='axis', values=stats_cols)
    
    # Flatten MultiIndex columns: ('mean', 'x') -> 'x_mean'
    pivoted.columns = [f"{axis}_{metric}" for metric, axis in pivoted.columns]
    
    return pivoted.reset_index()

def create_final_feature_matrix(accel_stats_df, gyro_stats_df):
    """
    Step 2: Merge the processed Accel and Gyro DataFrames based on filename.
    """
    # 1. Reshape both DataFrames using Step 1
    accel_flat = reshape_sensor_data(accel_stats_df)
    gyro_flat = reshape_sensor_data(gyro_stats_df)
    
    # 1. ניקוי שמות הקבצים בטבלת האקסלרומטר
    accel_flat['filename_clean'] = accel_flat['filename'].str.replace('accel', '', case=False)

    # 2. ניקוי שמות הקבצים בטבלת הג'ירוסקופ
    gyro_flat['filename_clean'] = gyro_flat['filename'].str.replace('gyro', '', case=False)

    # 2. Merge them on the 'filename' column
    # We add suffixes to distinguish between same metrics from different sensors
    final_df = pd.merge(accel_flat, gyro_flat, on='filename_clean', suffixes=('_accel', '_gyro'))
    
    # 3. Add the Target Label (Hand)
    # Extracting 'Left' or 'Right' from the filename for supervised learning
    final_df['label'] = final_df['filename_clean'].apply(
        lambda x: 1 if 'right' in x.lower() else 0 # 1 for Right, 0 for Left
    )
    
    return final_df

# Usage Example:
# final_data = create_final_feature_matrix(left_accel_df, left_gyro_df)
# final_data.to_csv("master_features.csv", index=False)

# הרצה לדוגמה (וודא שסיפקת את הנתיב הנכון לקובץ הפיצ'רים שלך)
# model, encoder = train_hand_classifier('path_to_your_features.csv')


train_hand_classifier('master_features.csv')