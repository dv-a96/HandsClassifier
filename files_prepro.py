import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.signal import butter, filtfilt
from analyze_data import _load_sensor_csv

def add_timestamp_diff_column(base_path: str, file_type: str):
    """
    Iterates through Left and Right hand directories, adds a time difference column, 
    and returns processing statistics.
    """
    hands = ['Left', 'Right']
    processed_stats = []

    for hand in hands:
        hand_dir = os.path.join(base_path, hand)
        pattern = f"**/*{file_type}.csv"
        files = sorted(glob.glob(os.path.join(hand_dir, pattern), recursive=True))
        
        for fpath in files:
            # Resampled files already contain headers; raw files do not
            if "resampled" in fpath.lower():
                header = 0  
            else:
                header = None
            
            df = _load_sensor_csv(fpath, header=header)
            
            # Calculate time intervals (Sampling Interval)
            # This represents the gap between consecutive samples in nanoseconds
            df['ts_diff'] = pd.to_numeric(df['timestamp'], errors='coerce').diff()
            
            # Collect statistics for visualization
            valid_diffs = df['ts_diff'].dropna()
            processed_stats.append({
                'filename': os.path.basename(fpath),
                'hand': hand,
                'min': valid_diffs.min(),
                'max': valid_diffs.max(),
                'mean': valid_diffs.mean(),
                'all_diffs': valid_diffs.values # Storing all values for boxplot/histogram analysis
            })
            
    return processed_stats


def plot_sampling_consistency(stats_list: list, save_path: str=None):
    """
    Generates a Boxplot to visualize the consistency of sampling intervals across all files.
    """
    filenames = [s['filename'] for s in stats_list]
    all_data = [s['all_diffs'] for s in stats_list]
    colors = ['blue' if s['hand'] == 'Left' else 'orange' for s in stats_list]

    plt.figure(figsize=(15, 7))
    
    # Use Boxplot to show Median, Quartiles, and Outliers (which indicate dropped samples)
    bp = plt.boxplot(all_data, tick_labels=filenames, patch_artist=True)
    
    # Color coding by hand side
    for i in range(len(bp['boxes'])):
            # Set the box body color
            bp['boxes'][i].set_facecolor(colors[i])
            bp['boxes'][i].set_alpha(0.5)
            
            # Set median line color and width for better visibility
            bp['medians'][i].set_color(colors[i]) 
            bp['medians'][i].set_linewidth(2)

    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Timestamp Interval (ns)')
    plt.title('Sampling Consistency Check (Boxplot per File)\nBlue: Left Hand | Orange: Right Hand')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_sampling_rate_histograms(stats_list, bins=50, save_path=None):
    """
    Generates a histogram for each file to visualize sampling interval distribution.
    Includes vertical lines for median/quartiles and a summary box for metrics.
    """
    num_files = len(stats_list)
    cols = 2
    rows = (num_files + 1) // cols
    
    # Create subplots grid
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), squeeze=False)
    axes = axes.flatten()

    for i, s in enumerate(stats_list):
        ax = axes[i]
        data = np.array(s['all_diffs'])
        color = 'blue' if s['hand'] == 'Left' else 'orange'
        
        # 1. Calculate Descriptive Statistics
        median = np.median(data)
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        d_min, d_max = np.min(data), np.max(data)
        unique_counts = len(np.unique(data))
        
        # 2. Plot Histogram
        ax.hist(data, bins=bins, color=color, alpha=0.6, edgecolor='black', label='Interval Distribution')
        
        # 3. Add Vertical Indicator Lines (Median & Quartiles)
        ax.axvline(median, color='red', linestyle='--', linewidth=2)
        ax.axvline(q1, color='green', linestyle=':', linewidth=1.5)
        ax.axvline(q3, color='green', linestyle=':', linewidth=1.5)
        
        # 4. Annotate Lines with Values
        trans = ax.get_xaxis_transform() 
        ax.text(median, 0.95, f' Med: {median:,.0f}', color='red', transform=trans, fontweight='bold', ha='left')
        ax.text(q1, 0.88, f' Q1: {q1:,.0f}', color='green', transform=trans, ha='right')
        ax.text(q3, 0.88, f' Q3: {q3:,.0f}', color='green', transform=trans, ha='left')
        
        # 5. Create Summary Statistics Box
        stats_text = (f"Min: {d_min:,.0f} ns\n"
                      f"Max: {d_max:,.0f} ns\n"
                      f"Unique Diffs: {unique_counts}")
        
        # Positioning text box in upper right corner of the individual plot
        ax.text(0.95, 0.75, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        # Format axes
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.set_title(f"File: {s['filename']} ({s['hand']})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time Difference (ns)")
        ax.set_yscale('log')  # Log scale used to emphasize outliers/jitter
        ax.set_ylabel("Frequency (Log Scale)")
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Histogram plot saved successfully to: {save_path}")
    else:
        plt.show()

def resample_and_interpolate_file(df, target_interval_ns=2_000_000):
    """
    Aligns sensor data to a fixed time grid (e.g., every 2ms) using linear interpolation.
    This fixes irregular sampling rates before applying digital filters.
    """
    target_col = 'timestamp'
    if "timestamp" not in df.columns:
        target_col = 3
    # 1. Ensure numeric data types and clean timestamp duplicates
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    
    # Average values if two samples occur on the exact same nanosecond
    df = df.groupby(target_col).mean().reset_index()
    
    # 2. Set timestamp as index for time-based operations
    df = df.set_index(target_col)
    
    # 3. Create a "Perfect Grid"
    # Generates a sequence from start to end with a constant 2ms interval (500Hz)
    start_time = df.index.min()
    end_time = df.index.max()
    perfect_grid = np.arange(start_time, end_time, target_interval_ns)
    
    # 4. Perform Alignment:
    # Combine original indexes with the perfect grid to prevent data loss during calculation
    combined_index = np.unique(np.concatenate([df.index, perfect_grid]))
    df_aligned = df.reindex(combined_index)
    
    # 5. Linear Interpolation to fill the gaps between original samples
    df_interpolated = df_aligned.interpolate(method='linear')
    
    # 6. Extraction: Keep only the points that match our target grid exactly
    df_final = df_interpolated.loc[perfect_grid]
    
    return df_final.reset_index()


def resample_and_interpolate_dataset(source_dir, output_base_dir, target_interval_ns=2_000_000):
    """
    Iterates through all CSV files in the source directory, applies resampling,
    and saves the results into subdirectories organized by hand (Left/Right).
    
    Parameters:
    - source_dir (str): Path where the raw sensor CSV files are located.
    - output_base_dir (str): Root path for saving the resampled files.
    - target_interval_ns (int): The fixed time grid interval (default 2ms = 500Hz).
    """
    
    # 1. Find all CSV files in the source directory
    file_pattern = [os.path.join(source_dir, "Left/*accel.csv"), os.path.join(source_dir, "Right/*accel.csv"), os.path.join(source_dir, "Left/*gyro.csv"), os.path.join(source_dir, "Right/*gyro.csv")]
    all_files = []
    for pattern in file_pattern:
        all_files.extend(glob.glob(pattern, recursive=True))
    
    if not all_files:
        print(f"No CSV files found in {source_dir}")
        return

    print(f"Found {len(all_files)} files. Starting resampling process...")

    for file_path in all_files:
        try:
            # 2. Identify the hand (label) from the filename
            filename = os.path.basename(file_path)
            hand_label = "Left" if "left" in file_path.lower() else "Right"
            
            # 3. Create the output directory for the specific hand if it doesn't exist
            # e.g., output_base_dir/Left/
            save_dir = os.path.join(output_base_dir, hand_label)
            os.makedirs(save_dir, exist_ok=True)
            
            # 4. Load the raw data
            raw_df = pd.read_csv(file_path, header=None)
            
            # 5. Apply your resampling function
            # Note: Ensure the resample_and_interpolate_file function is defined in your script
            resampled_df = resample_and_interpolate_file(raw_df, target_interval_ns)
            resampled_df.rename(columns={3: 'timestamp', 0: 'x', 1: 'y', 2: 'z'}, inplace=True)
            
            # 6. Save the resampled file with a prefix to indicate it's processed
            save_path = os.path.join(save_dir, f"res_{filename}")
            resampled_df.to_csv(save_path, index=False)
            
            print(f"Processed: {filename} -> {hand_label} folder")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("\nResampling complete for all files.")



def apply_butterworth_highpass(data, cutoff_hz, fs, order=4):
    """
    Applies a High-pass Butterworth filter to remove DC components (like gravity).
    """
    # 1. Calculate Nyquist Frequency
    # Maximum measurable frequency, always half the sampling rate (fs/2).
    nyq = 0.5 * fs
    
    # 2. Normalize Cutoff Frequency
    # Digital filters expect a value between 0 and 1, where 1 is the Nyquist Frequency.
    normal_cutoff = cutoff_hz / nyq
    
    # 3. Filter Design
    # Returns 'b' (numerator) and 'a' (denominator) coefficients of the filter formula.
    # btype='high' creates a High-pass filter (blocks lows, passes highs).
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    # 4. Filter Execution
    # Using 'filtfilt' (Zero-phase filtering) to avoid time delays (Phase Shift).
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data



def apply_highpass_to_all_files(root_dir: str, save_dir: str):
    """
    Processes all files in the directory tree, applying high-pass filtering to each axis.
    """
    for hand in ['Left', 'Right']:
        for file_type in ['accel', 'gyro']:
            pattern = f"**/*{file_type}.csv"
            files = sorted(glob.glob(os.path.join(root_dir, f'{hand}', pattern), recursive=True))
            for file in files:
                df = _load_sensor_csv(file, header=0)  
                
                if file_type == 'accel':
                    cutoff = 0.6  # Optimal for gravity removal
                    order = 4
                    cols_to_filter = ['accel_x', 'accel_y', 'accel_z'] 
                else:
                    cutoff = 0.1  # Optimal for gyroscope bias drift cleaning
                    order = 2
                    cols_to_filter = ['gyro_x', 'gyro_y', 'gyro_z']
                
                for col in cols_to_filter:
                        data_col = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        df[col] = apply_butterworth_highpass(data_col, cutoff_hz=cutoff, fs=500, order=order)
                
                # Save processed data to the target directory
                df.to_csv(f'{save_dir}/{hand}/{file.split("/")[-1]}', index=False)

# Start Processing
# apply_highpass_to_all_files(root_dir='Resampled', save_dir='Clean')