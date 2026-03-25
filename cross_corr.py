import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os

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


def extract_correlation_features(df, accel_df, left_template, right_template, target_length=200):
    """Extracts correlation features from the given dataframe using the provided templates.
    Args:   df (pd.DataFrame): The input dataframe containing the gyro data.
            accel_df (pd.DataFrame): The dataframe containing the accelerometer data.
            left_template (np.array): The template signal for the left class.
            right_template (np.array): The template signal for the right class.
            target_length (int): The length to which signals will be resampled for correlation with the templates.
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

    # 3. Correlation with Right template
    features['corr_with_right_template'] = np.corrcoef(y_gyro_resampled, right_template)[0, 1]

    # 4. Correlation with Left template
    features['corr_with_left_template'] = np.corrcoef(y_gyro_resampled, left_template)[0, 1]

    return features


def get_paired_files(directory):
    pairs = {}
    for filename in os.listdir(directory):
        if not filename.endswith('.csv'):
            continue
            
        # נניח שהקובץ הוא "20260325_1400_accel.csv"
        # אנחנו לוקחים את הכל חוץ מהסיומת '_accel.csv' או '_gyro.csv'
        base_name = filename.replace('accel.csv', '').replace('gyro.csv', '')
        
        if base_name not in pairs:
            pairs[base_name] = {'accel': None, 'gyro': None}
        
        if 'accel' in filename:
            pairs[base_name]['accel'] = os.path.join(directory, filename)
        elif 'gyro' in filename:
            pairs[base_name]['gyro'] = os.path.join(directory, filename)
            
    # מחזירים רק זוגות מלאים (שיש להם גם וגם)
    return {k: v for k, v in pairs.items() if v['accel'] and v['gyro']}

left_gyro_list = [pd.read_csv(f'New/Smoothed/Left/{path}') for path in os.listdir('New/Smoothed/Left') if path.endswith('gyro.csv')]
right_gyro_list = [pd.read_csv(f'New/Smoothed/Right/{path}') for path in os.listdir('New/Smoothed/Right') if path.endswith('gyro.csv')]
left_accel_list = [pd.read_csv(f'New/Smoothed/Left/{path}') for path in os.listdir('New/Smoothed/Left') if path.endswith('accel.csv')]
right_accel_list = [pd.read_csv(f'New/Smoothed/Right/{path}') for path in os.listdir('New/Smoothed/Right') if path.endswith('accel.csv')]
left_template = create_template(left_gyro_list, axis='y_sg', target_length=750)
right_template = create_template(right_gyro_list, axis='y_sg', target_length=750)



# Plot right vs left template
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(left_template, label='Left Template', color='blue')
plt.plot(right_template, label='Right Template', color='orange')
plt.title('Left vs Right Template Signals')
plt.xlabel('Time (normalized)')
plt.ylabel('Gyro Signal')
plt.legend()
plt.show()

left_pairs = get_paired_files('New/Smoothed/Left')
right_pairs = get_paired_files('New/Smoothed/Right')

print("-"*20,"Left ", "-"*20)
for base_name, paths in left_pairs.items():
    accel_path = paths['accel']
    gyro_path = paths['gyro']
    accel_df = pd.read_csv(accel_path)
    gyro_df = pd.read_csv(gyro_path)
    features = extract_correlation_features(gyro_df, accel_df, left_template, right_template, target_length=750)
    print(f"Features for {base_name}:")
    print(f"\tCorrelation with Right Template: {features['corr_with_right_template']}")
    print(f"\tCorrelation with Left Template: {features['corr_with_left_template']}")

print("-"*20,"Right ", "-"*20)
for base_name, paths in right_pairs.items():
    accel_path = paths['accel']
    gyro_path = paths['gyro']
    accel_df = pd.read_csv(accel_path)
    gyro_df = pd.read_csv(gyro_path)
    features = extract_correlation_features(gyro_df, accel_df, left_template, right_template, target_length=750)
    print(f"Features for {base_name}:")
    print(f"\tCorrelation with Right Template: {features['corr_with_right_template']}")
    print(f"\tCorrelation with Left Template: {features['corr_with_left_template']}")