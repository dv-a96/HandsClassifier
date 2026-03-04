import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

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
    """Plot bar charts comparing mean values between Left and Right directories for each axis."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows (accel/gyro), 4 columns (x,y,z,timestamp)
    fig.suptitle('Comparison of Mean Values: Left vs Right Directories', fontsize=16)
    
    axes = axes.flatten()
    plot_idx = 0
    
    for file_type in ["accel", "gyro"]:
        for axis in ["x", "y", "z", "timestamp"]:
            col_name = f"{file_type}_{axis}"
            
            # Collect means for Left and Right
            left_means = [stats["mean"].get(col_name, 0) for stats in left_data[file_type].values()]
            right_means = [stats["mean"].get(col_name, 0) for stats in right_data[file_type].values()]
            
            if not left_means and not right_means:
                continue
            
            # Calculate average means
            left_avg = sum(left_means) / len(left_means) if left_means else 0
            right_avg = sum(right_means) / len(right_means) if right_means else 0
            
            # Bar plot
            ax = axes[plot_idx]
            ax.bar(['Left', 'Right'], [left_avg, right_avg], color=['blue', 'red'])
            ax.set_title(f'{col_name.upper()} Mean')
            ax.set_ylabel('Mean Value')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('left_vs_right_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'left_vs_right_comparison.png'")
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


if __name__ == "__main__":
    main()
