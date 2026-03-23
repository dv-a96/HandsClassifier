import os
import glob
import pandas as pd
import files_prepro
import analyze_data


def load_data(root_dir) -> pd.DataFrame:
    """Load all accelerometer and gyroscope data from Left and Right subdirectories."""
    data_frames = []
    
    for hand in ['Left', 'Right']:
        hand_dir = os.path.join(root_dir, hand)
        if not os.path.isdir(hand_dir):
            continue
        
        for subdir in os.listdir(hand_dir):
            subpath = os.path.join(hand_dir, subdir)
            if not os.path.isdir(subpath):
                continue
            
            # Find accel and gyro files
            accel_files = glob.glob(os.path.join(subpath, '*accel.csv'))
            gyro_files = glob.glob(os.path.join(subpath, '*gyro.csv'))
            
            if accel_files and gyro_files:
                try:
                    # Load accel data
                    df_accel = pd.read_csv(accel_files[0], header=None, 
                                          names=['accel_x', 'accel_y', 'accel_z', 'timestamp'])
                    
                    # Load gyro data
                    df_gyro = pd.read_csv(gyro_files[0], header=None, 
                                        names=['gyro_x', 'gyro_y', 'gyro_z', 'timestamp'])
                    
                    # Merge on timestamp
                    df = pd.merge(df_accel, df_gyro, on='timestamp', how='inner')
                    
                    # Add metadata
                    df['hand'] = hand
                    df['file_id'] = subdir
                    
                    # Reorder columns
                    df = df[['file_id', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'hand']]
                    
                    data_frames.append(df)
                except Exception as e:
                    print(f"Error loading data for {subpath}: {e}")
    
    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        return pd.DataFrame()


def normelized_data(df: pd.DataFrame) -> pd.DataFrame:
    "Normalize the data using z-score normalization"
    for col in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            df[col] = (df[col] - mean_val) / std_val
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    "Preprocess the data by normalizing"
    normelized_df = normelized_data(df)
    return normelized_df


def main():
    raw_data_path = 'New/Raw'

    # # Plot raw data for each hand, file type, and axis
    # for hand in ['Left', 'Right']:
    #     for file_type in ['accel', 'gyro']:
    #         for ax in ['x', 'y', 'z']:
    #             analyze_data.plot_hand_data(f'{raw_data_path}/{hand}', file_type=file_type, axis=ax, raw=True, save_path=f'{raw_data_path}/{hand.lower()}_{file_type}_{ax}_raw.png')
    
    # # Check raw data sample rate
    # accel_sampling_rates_list = files_prepro.add_timestamp_diff_column(raw_data_path, 'accel')
    # gyro_sampling_rates_list = files_prepro.add_timestamp_diff_column(raw_data_path, 'gyro')
    # files_prepro.plot_sampling_rate_histograms(accel_sampling_rates_list, save_path='New/Raw/accel_sampling_consistency_histograms.png')
    # files_prepro.plot_sampling_rate_histograms(gyro_sampling_rates_list, save_path='New/Raw/gyro_sampling_consistency_histograms.png')

    # # Resample data to a consistent rate and save to new CSV files
    # files_prepro.resample_and_interpolate_dataset(source_dir=raw_data_path, output_base_dir='New/Resampled/', target_interval_ns=2000000) # 500Hz target rate corresponds to 2ms interval
        
    # # Verify resampled data sample rate consistency
    # accel_sampling_rates_list = files_prepro.add_timestamp_diff_column('New/Resampled/', 'accel')
    # gyro_sampling_rates_list = files_prepro.add_timestamp_diff_column('New/Resampled/', 'gyro')
    # files_prepro.plot_sampling_rate_histograms(accel_sampling_rates_list, save_path='New/Resampled/accel_sampling_consistency_histograms.png')
    # files_prepro.plot_sampling_rate_histograms(gyro_sampling_rates_list, save_path='New/Resampled/gyro_sampling_consistency_histograms.png')

    # # Plot resampled data for each hand, file type, and axis
    # for hand in ['Left', 'Right']:
    #     for file_type in ['accel', 'gyro']:
    #         for ax in ['x', 'y', 'z']:
    #             analyze_data.plot_hand_data(f'New/Resampled/{hand}', file_type=file_type, axis=ax, raw=False, save_path=f'New/Resampled/{hand.lower()}_{file_type}_{ax}_res.png')
    

    # # Clean the data
    # files_prepro.apply_highpass_to_all_files(root_dir='New/Resampled', save_dir='New/Clean')

    # # Plot cleaned data for each hand, file type, and axis
    # for hand in ['Left', 'Right']:
    #     for file_type in ['accel', 'gyro']:
    #         for ax in ['x', 'y', 'z']:
    #             analyze_data.plot_hand_data(f'New/Clean/{hand}', file_type=file_type, axis=ax, raw=False, save_path=f'New/Clean/{hand.lower()}_{file_type}_{ax}_clean.png')


    # Smooth the cleaned data
    for hand in ['Left', 'Right']:
        for file_type in ['accel', 'gyro']:
            analyze_data.smooth_and_save_hand_data(hand_dir=f'New/Clean/{hand}', save_dir=f'New/Smoothed/{hand}', file_type=file_type, max_files=5)


    # Plot smoothed data for each hand, file type, and axis
    for hand in ['Left', 'Right']:
        for file_type in ['accel', 'gyro']:
            for ax in ['x', 'y', 'z']:
                analyze_data.plot_hand_data(f'New/Smoothed/{hand}', file_type=file_type,max_files=5, axis=ax, raw=False, save_path=f'New/Smoothed/{hand.lower()}_{file_type}_{ax}_smoothed.png')

if __name__ == "__main__":
    main()