import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def summarize_dataframe(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """Return a summary table with basic statistics for each column.
    
    Column names are assigned based on the pattern:
    - *accel.csv -> accel_x, accel_y, accel_z, timestamp
    - *gyro.csv -> gyro_x, gyro_y, gyro_z, timestamp
    """
    # Determine column names based on pattern
    if "accel" in pattern:
        col_names = ["accel_x", "accel_y", "accel_z", "timestamp"]
    elif "gyro" in pattern:
        col_names = ["gyro_x", "gyro_y", "gyro_z", "timestamp"]
    else:
        col_names = [f"col_{i}" for i in range(len(df.columns))]
    
    # Assign column names to dataframe
    df.columns = col_names[:len(df.columns)]
    
    # compute stats
    stats = {
        "count": df.count(),
        "mean": df.mean(),
        "median": df.median(),
        "std": df.std(),
        "min": df.min(),
        "max": df.max(),
    }
    summary = pd.DataFrame(stats).T
    return summary


def process_file(path: str, pattern: str) -> None:
    """Read a csv file and print statistics."""
    try:
        df = pd.read_csv(path, header=None)
    except Exception as e:
        print(f"Unable to read {path}: {e}")
        return

    print(f"\n--- {path} ---")
    print(f"Number of rows: {len(df)}")
    if df.empty:
        print("(empty dataframe)")
        return

    summary = summarize_dataframe(df.select_dtypes(include=["number"]), pattern)
    if summary.empty:
        print("No numeric columns to summarize.")
    else:
        print(summary.to_string())


def collect_data_for_root(root_dir: str) -> dict:
    """Collect statistics data for a root directory."""
    data = {"accel": {}, "gyro": {}}
    
    if not os.path.isdir(root_dir):
        return data
    
    for subdir in sorted(os.listdir(root_dir)):
        subpath = os.path.join(root_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        
        patterns = [("*accel.csv", "accel"), ("*gyro.csv", "gyro")]
        for pat, file_type in patterns:
            for filepath in glob.glob(os.path.join(subpath, pat)):
                try:
                    df = pd.read_csv(filepath, header=None)
                except Exception as e:
                    continue
                
                # Assign column names
                if file_type == "accel":
                    col_names = ["accel_x", "accel_y", "accel_z", "timestamp"]
                else:
                    col_names = ["gyro_x", "gyro_y", "gyro_z", "timestamp"]
                df.columns = col_names[:len(df.columns)]
                
                # Extract numeric columns only
                df_numeric = df.select_dtypes(include=["number"])
                
                # Store statistics
                file_key = os.path.basename(filepath)
                data[file_type][file_key] = {
                    "count": df_numeric.count().to_dict(),
                    "mean": df_numeric.mean().to_dict(),
                }
    
    return data




def compare_statistics_in_root(root_dir: str) -> None:
    """Compare count and mean statistics across all accel/gyro files in a root directory."""
    if not os.path.isdir(root_dir):
        print(f"Directory does not exist: {root_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"COMPARISON TABLE: {root_dir}")
    print(f"{'='*80}")
    
    # Collect data: {file_type: {"filename": {"column": {"count": ..., "mean": ...}}}}
    data = {"accel": {}, "gyro": {}}
    
    # Walk through subdirectories
    for subdir in sorted(os.listdir(root_dir)):
        subpath = os.path.join(root_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        
        patterns = [("*accel.csv", "accel"), ("*gyro.csv", "gyro")]
        for pat, file_type in patterns:
            for filepath in glob.glob(os.path.join(subpath, pat)):
                try:
                    df = pd.read_csv(filepath, header=None)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    continue
                
                # Assign column names
                if file_type == "accel":
                    col_names = ["accel_x", "accel_y", "accel_z", "timestamp"]
                else:
                    col_names = ["gyro_x", "gyro_y", "gyro_z", "timestamp"]
                df.columns = col_names[:len(df.columns)]
                
                # Extract numeric columns only
                df_numeric = df.select_dtypes(include=["number"])
                
                # Store statistics
                file_key = os.path.basename(filepath)
                data[file_type][file_key] = {
                    "count": df_numeric.count().to_dict(),
                    "mean": df_numeric.mean().to_dict(),
                }
    
    # Display comparison tables
    for file_type in ["accel", "gyro"]:
        if not data[file_type]:
            print(f"\nNo {file_type} files found.")
            continue
        
        print(f"\n--- {file_type.upper()} FILES ---")
        
        # Get all unique column names (without prefix)
        all_columns = set()
        for filename, stats in data[file_type].items():
            all_columns.update(stats["count"].keys())
        
        # Display a separate table for each column
        for col in sorted(all_columns):
            print(f"\n{col.upper()}:")
            comparison_data = []
            for filename, stats in sorted(data[file_type].items()):
                if col in stats["count"]:
                    row = {
                        "File": filename,
                        f"{col}_count": int(stats["count"][col]),
                        f"{col}_mean": round(stats["mean"][col], 4),
                    }
                    comparison_data.append(row)
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                print(comparison_df.to_string(index=False))


def plot_left_vs_right_comparison(left_data, right_data):
    """Plot bar charts comparing mean values per file for each axis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows (accel/gyro), 3 columns (x,y,z)
    fig.suptitle('Mean Values per File: Left vs Right Directories', fontsize=16)
    
    axes = axes.flatten()
    plot_idx = 0
    
    for file_type in ["accel", "gyro"]:
        for axis in ["x", "y", "z"]:
            col_name = f"{file_type}_{axis}"
            
            # Collect all files and their means
            files_data = []
            
            # Add Left files
            for filename, stats in sorted(left_data[file_type].items()):
                files_data.append({
                    "file": f"{filename} (L)",
                    "mean": stats["mean"].get(col_name, 0),
                    "color": "blue"
                })
            
            # Add Right files
            for filename, stats in sorted(right_data[file_type].items()):
                files_data.append({
                    "file": f"{filename} (R)",
                    "mean": stats["mean"].get(col_name, 0),
                    "color": "red"
                })
            
            if not files_data:
                continue
            
            # Bar plot
            ax = axes[plot_idx]
            file_names = [d["file"] for d in files_data]
            means = [d["mean"] for d in files_data]
            colors = [d["color"] for d in files_data]
            
            ax.bar(file_names, means, color=colors, alpha=0.7)
            ax.set_title(f'{col_name.upper()} Mean per File')
            ax.set_ylabel('Mean Value')
            ax.set_xlabel('File')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('left_vs_right_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'left_vs_right_comparison.png'")
    plt.close()


def calculate_sampling_frequencies(csv_path, video_path):
    """Calculate sampling frequencies for accelerometer/gyroscope from CSV and video from video file."""
    sensor_freq = None
    video_fps = None
    
    # Calculate sensor sampling frequency from CSV timestamps
    try:
        df = pd.read_csv(csv_path, header=None)
        if df.empty or df.shape[1] < 4:
            print(f"CSV file {csv_path} does not have enough columns for timestamp.")
        else:
            # Assign column names based on file type
            if "accel" in csv_path:
                col_names = ["accel_x", "accel_y", "accel_z", "timestamp"]
            elif "gyro" in csv_path:
                col_names = ["gyro_x", "gyro_y", "gyro_z", "timestamp"]
            else:
                col_names = [f"col_{i}" for i in range(len(df.columns))]
            
            df.columns = col_names[:len(df.columns)]
            
            if "timestamp" in df.columns:
                timestamps = df["timestamp"].dropna()
                if len(timestamps) > 1:
                    diffs = timestamps.diff().dropna()
                    avg_diff = diffs.mean()
                    if avg_diff > 0:
                        sensor_freq = 1.0 / avg_diff
                    else:
                        print(f"Invalid timestamp differences in {csv_path}")
                else:
                    print(f"Not enough timestamps in {csv_path}")
            else:
                print(f"No timestamp column found in {csv_path}")
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
    
    # Get video frame rate
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        else:
            print(f"Could not open video file {video_path}")
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
    
    return sensor_freq, video_fps


def plot_sampling_frequencies(sensor_freq, video_fps, title="Sampling Frequencies Comparison"):
    """Plot a comparison of sensor and video sampling frequencies."""
    if sensor_freq is None and video_fps is None:
        print("No data to plot.")
        return
    
    labels = []
    values = []
    colors = []
    
    if sensor_freq is not None:
        labels.append("Sensor (Hz)")
        values.append(sensor_freq)
        colors.append("blue")
    
    if video_fps is not None:
        labels.append("Video (FPS)")
        values.append(video_fps)
        colors.append("green")
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=colors, alpha=0.7)
    plt.title(title)
    plt.ylabel("Frequency (Hz)")
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + max(values) * 0.01, f"{v:.2f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sampling_frequencies.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'sampling_frequencies.png'")
    plt.close()


def plot_all_frequencies(freq_data):
    """Plot sampling frequencies for all files in one figure."""
    if not freq_data["sensor"]:
        print("No frequency data to plot.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(freq_data["labels"]))
    
    ax.bar([i - 0.2 for i in x], freq_data["sensor"], width=0.4, label='Sensor (Hz)', color='blue', alpha=0.7)
    ax.bar([i + 0.2 for i in x], freq_data["video"], width=0.4, label='Video (FPS)', color='green', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(freq_data["labels"], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Sampling Frequencies: Sensor vs Video')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('sampling_frequencies_all.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'sampling_frequencies_all.png'")
    plt.close()


def walk_and_analyze(root_dirs):
    """Walk through given root directories and analyze accel/gyro files."""
    for root in root_dirs:
        print(f"\n== Processing directory: {root} ==")
        if not os.path.isdir(root):
            print(f"Directory does not exist: {root}")
            continue
        # iterate subdirectories
        for subdir in sorted(os.listdir(root)):
            subpath = os.path.join(root, subdir)
            if not os.path.isdir(subpath):
                continue
            print(f"\n-- Subdirectory: {subpath} --")
            # find accel and gyro csvs
            patterns = ["*accel.csv", "*gyro.csv"]
            for pat in patterns:
                for filepath in glob.glob(os.path.join(subpath, pat)):
                    process_file(filepath, pat)


def main():
    # default directories relative to this script
    base = os.path.dirname(os.path.abspath(__file__))
    dirs = [os.path.join(base, "Left"), os.path.join(base, "Right")]
    
    # Individual file analysis
    walk_and_analyze(dirs)
    
    # Collect data for comparison
    left_data = collect_data_for_root(dirs[0])
    right_data = collect_data_for_root(dirs[1])
    
    # Comparison analysis for each root directory
    print(f"\n\n{'#'*80}")
    print("COMPARISON TABLES FOR EACH ROOT DIRECTORY")
    print(f"{'#'*80}")
    for root_dir in dirs:
        compare_statistics_in_root(root_dir)
    
    # Plot comparison
    print(f"\n\n{'#'*80}")
    print("PLOTTING LEFT VS RIGHT COMPARISON")
    print(f"{'#'*80}")
    plot_left_vs_right_comparison(left_data, right_data)
    
    # # Calculate and plot sampling frequencies
    # print(f"\n\n{'#'*80}")
    # print("CALCULATING AND PLOTTING SAMPLING FREQUENCIES")
    # print(f"{'#'*80}")
    
    # # Find video files
    # video_files = {}
    # for root in dirs:
    #     for file in glob.glob(os.path.join(root, "**", "*.mp4"), recursive=True):
    #         video_key = os.path.basename(file).replace('.mp4', '')
    #         video_files[video_key] = file
    
    # # Calculate frequencies for each CSV
    # freq_data = {"sensor": [], "video": [], "labels": []}
    # for root in dirs:
    #     for subdir in sorted(os.listdir(root)):
    #         subpath = os.path.join(root, subdir)
    #         if not os.path.isdir(subpath):
    #             continue
    #         for pat in ["*accel.csv", "*gyro.csv"]:
    #             for csv_path in glob.glob(os.path.join(subpath, pat)):
    #                 # Find corresponding video (remove accel/gyro and .csv)
    #                 video_key = os.path.basename(csv_path).replace('.csv', '').replace('accel', '').replace('gyro', '').strip('_')
    #                 video_path = video_files.get(video_key)
    #                 if video_path:
    #                     sensor_freq, video_fps = calculate_sampling_frequencies(csv_path, video_path)
    #                     if sensor_freq is not None:
    #                         freq_data["sensor"].append(sensor_freq)
    #                         freq_data["video"].append(video_fps or 0)
    #                         freq_data["labels"].append(os.path.basename(csv_path))
    
    # plot_all_frequencies(freq_data)


if __name__ == "__main__":
    main()
