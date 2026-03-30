import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from analyze_data import create_stats_dfs
import cross_corr

def split_train_test(data_path, test_size=0.2, random_state=42):
    # Split the dataset into training and testing sets
    # test_size: Proportion of the dataset to include in the test split (20% here)
    # random_state: Ensures reproducibility of the split
    files_id_set = set()
    for hand in ['left', 'right']:
        hand_dir = os.path.join(data_path, hand)
        for filename in os.listdir(hand_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(hand_dir, filename)
                file_id = filename.replace('accel_', '').replace('gyro_', '').replace('.csv', '')
                # Store the file ID along with its path for later use in feature extraction
                file_path_acc, file_path_gyro = None, None
                if 'accel' in filename:
                    file_path_acc = file_path
                elif 'gyro' in filename:
                    file_path_gyro = file_path
                if file_id in files_id_set and file_path_acc:
                    files_id_set[file_id] = (file_id, file_path_acc, files_id_set[file_id][2])
                elif file_id in files_id_set and file_path_gyro:
                    files_id_set[file_id] = (file_id, files_id_set[file_id][1], file_path_gyro)
                else:
                    files_id_set.add((file_id, file_path_acc, file_path_gyro))
    files_id_list = list(files_id_set)
    train_ids, test_ids = train_test_split(files_id_list, test_size=test_size, random_state=random_state)
    return train_ids, test_ids

def extract_train_features(train_ids):
    os.makedirs('Temp', exist_ok=True)
    os.makedirs('Temp/Left', exist_ok=True)
    os.makedirs('Temp/Right', exist_ok=True)
    # save the train samples in a the correct temporary directory for feature extraction
    for file_id, file_path_acc, file_path_gyro in train_ids:
        if file_path_acc and file_path_gyro:
            hand = 'Left' if 'left' in file_path_acc else 'Right'
            temp_dir = os.path.join('Temp', hand)
            # Save the accel and gyro files in the temporary directory for feature extraction
            temp_acc_path = os.path.join(temp_dir, f'{file_id}accel.csv')
            temp_gyro_path = os.path.join(temp_dir, f'{file_id}gyro.csv')
            pd.read_csv(file_path_acc).to_csv(temp_acc_path, index=False)
            pd.read_csv(file_path_gyro).to_csv(temp_gyro_path, index=False)
    # Call the feature extraction function on the temporary directory
    create_stats_dfs('Temp', 'Temp/Stats')
    left_pairs = cross_corr.get_paired_files('Temp/Left')
    right_pairs = cross_corr.get_paired_files('Temp/Right')
    left_gyro_list = [pd.read_csv(f'Temp/Left/{path}') for path in os.listdir('Temp/Left') if path.endswith('gyro.csv')]
    right_gyro_list = [pd.read_csv(f'Temp/Right/{path}') for path in os.listdir('Temp/Right') if path.endswith('gyro.csv')]
    template_left = cross_corr.create_template(left_gyro_list, 'y_sg', 750)
    template_right = cross_corr.create_template(right_gyro_list, 'y_sg', 750)
    n_left = len(left_pairs)
    n_right = len(right_pairs)
    cross_corr.save_correlation_stats(left_pairs, right_pairs, template_left, template_right, n_left, n_right, 'Temp/Stats')
    # Unit corr data to one DF
    # Use load_feature_matrix to combine all features to single DF
    # Optional: selsct features
    os.rmdir('Temp')  # Clean up the temporary directory after feature extraction
    return fearues_df, template_left, template_right

        

def extract_test_features(test_samples, left_template, right_template):
    pass

def train_hand_classifier(train_features_path):
    # 1. Load the dataset
    # Expects a table containing feature columns (mean, variance, intensity, etc.)
    # and a 'label' column (Target variable)
    df = pd.read_csv(train_features_path)

    # Remove identification columns that should not be used as features for the model
    X = df.drop(columns=['label_accel', 'label_gyro', 'label', 'filename_clean', 'filename_accel', 'filename_gyro'], errors='ignore')
    y = df['label']
    
    # 2. Encode the Target (e.g., 'Left'/'Right' -> 0/1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    
    # 4. Model Initialization and Training
    # n_estimators: Number of trees in the forest (100 is a standard baseline)
    # n_jobs=-1: Uses all available CPU cores for faster training
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X, y_encoded)

    return X, rf_model, le

def predict_and_evaluate(X, rf_model, le, test_samples, test_labels):
    X_test = extract_test_features(test_samples)
    y_test = le.transform(test_labels)
    # 5. Model Evaluation
    y_pred = rf_model.predict(X_test)
    print(f"DEBUG: type(y_test) = {type(y_test)}, shape = {getattr(y_test, 'shape', 'no shape')}")
    print(f"DEBUG: type(y_pred) = {type(y_pred)}, shape = {getattr(y_pred, 'shape', 'no shape')}")
    print(f"DEBUG: target_names = {le.classes_}")
        
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_.astype(str)))
    
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
    
    return y_pred

X, model , le = train_hand_classifier('New/selected_features.csv')
y_pred = predict_and_evaluate(X, model, le, X_test, y_test)
