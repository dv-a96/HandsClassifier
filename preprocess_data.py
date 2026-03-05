import os
import glob
import pandas as pd


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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    "Preprocess the data by normalizing"
    pass

def main():
    raw_data = load_data('./')
    print(raw_data.head(5))
    print(raw_data.tail(5))
    print(raw_data.shape)

if __name__ == "__main__":
    main()