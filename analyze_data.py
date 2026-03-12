import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
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


def _load_sensor_csv(csv_path: str, header: bool = None) -> pd.DataFrame:
    """Helper to read a sensor CSV and return a dataframe with proper columns and converted time.

    The CSV is expected to contain three axes and a timestamp column representing
    inter-sample deltas (nanoseconds).
    """
    df = pd.read_csv(csv_path, header=header)
    if df.empty or df.shape[1] < 4:
        raise ValueError(f"CSV file {csv_path} does not have enough columns.")

    # Determine column names based on filename
    if "accel" in csv_path.lower() and  header is None:
        col_names = ["accel_x", "accel_y", "accel_z", "timestamp"]
    elif "gyro" in csv_path.lower() and header is None:
        col_names = ["gyro_x", "gyro_y", "gyro_z", "timestamp"]
    elif header is not None:
        col_names = df.columns.tolist()

    df.columns = col_names[:len(df.columns)]
    if "timestamp" not in df.columns:
        raise ValueError(f"No timestamp column found in {csv_path}")

    return df


def plot_raw_data(csv_path: str, save_path: str = None) -> plt.Figure:
    """Create a plot of the different axes over time from a CSV file.

    The figure object is returned so the caller can display, modify, or save it.
    If ``save_path`` is provided the plot will also be written to that path.

    Args:
        csv_path: Path to the CSV file (accel or gyro).
        save_path: Optional path to save the plot.
    Returns:
        A matplotlib Figure containing the drawn data.
    """
    try:
        df = _load_sensor_csv(csv_path)
        # Convert timestamp deltas -> cumulative seconds
        df["timestamp"] = df["timestamp"].cumsum() / 1e9
    except Exception as e:
        print(e)
        return None

    # determine labels
    if "accel" in csv_path.lower():
        title = "Accelerometer Axes Over Time"
        ylabel = "Acceleration"
    else:
        title = "Gyroscope Axes Over Time"
        ylabel = "Angular Velocity"

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['red', 'green', 'blue']
    axes = ['x', 'y', 'z']

    for i, axis in enumerate(axes):
        axis_col = f"{df.columns[0].split('_')[0]}_{axis}"
        if axis_col in df.columns:
            ax.plot(df["timestamp"], df[axis_col], color=colors[i], label=f'{axis.upper()}-axis', linewidth=1)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved as '{save_path}'")
    return fig


def compare_left_right_raw(left_dir: str, right_dir: str, max_files: int = 5, save_accel_path: str = None, save_gyro_path: str = None) -> dict:
    """Generate side-by-side raw plots for both accel and gyro data.

    Returns a dictionary with keys ``'accel'`` and ``'gyro'`` containing the
    corresponding Figure objects.  Separate save paths may be provided for each
    type.
    """
    figs = {}
    figs['accel'] = plot_side_by_side_raw(left_dir, right_dir, 'accel', max_files, save_accel_path)
    figs['gyro'] = plot_side_by_side_raw(left_dir, right_dir, 'gyro', max_files, save_gyro_path)
    return figs


def plot_side_by_side_raw(left_dir: str, right_dir: str, file_type: str, max_files: int = 5, save_path: str = None) -> plt.Figure:
    """Plot raw sensor data for files from left and right directories side by side.

    One figure is produced containing ``max_files`` rows and two columns.  The
    left column shows the first ``max_files`` matching files from ``left_dir``
    and the right column shows the first ``max_files`` matching files from
    ``right_dir``.  Only files containing ``file_type`` ("accel" or "gyro") in
    their name are considered.

    Args:
        left_dir: Directory containing left-hand CSV files.
        right_dir: Directory containing right-hand CSV files.
        file_type: "accel" or "gyro" to select the appropriate files.
        max_files: Number of files to plot from each side.
        save_path: Optional path to save the resulting figure.
    Returns:
        matplotlib Figure object with the subplots.
    """
    if file_type not in ("accel", "gyro"):
        raise ValueError("file_type must be 'accel' or 'gyro'")

    pattern = f"**/*{file_type}.csv"
    left_files = sorted(glob.glob(os.path.join(left_dir, pattern), recursive=True))[:max_files]
    right_files = sorted(glob.glob(os.path.join(right_dir, pattern), recursive=True))[:max_files]

    rows = max(len(left_files), len(right_files))
    fig, axes = plt.subplots(rows, 2, figsize=(12, rows * 2), squeeze=False)

    for i in range(rows):
        for side, files in enumerate((left_files, right_files)):
            ax = axes[i][side]
            if i < len(files):
                try:
                    df = _load_sensor_csv(files[i])
                except Exception as e:
                    ax.text(0.5, 0.5, str(e), ha='center', va='center')
                    ax.set_axis_off()
                    continue

                colors = ['red', 'green', 'blue']
                axes_names = ['x', 'y', 'z']
                for j, axis in enumerate(axes_names):
                    col = f"{file_type}_{axis}"
                    if col in df.columns:
                        ax.plot(df['timestamp'], df[col], color=colors[j], linewidth=1)
                ax.set_title(os.path.basename(files[i]))
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Acceleration' if file_type == 'accel' else 'Angular Velocity')
                ax.grid(True, alpha=0.3)
            else:
                ax.set_axis_off()
    # Left column title
    axes[0, 0].annotate('Left Hand Data', xy=(0.5, 1.3), xycoords='axes fraction',
                    ha='center', va='center', fontsize=16, fontweight='bold')

    # Right column title
    axes[0, 1].annotate('Right Hand Data', xy=(0.5, 1.3), xycoords='axes fraction',
                    ha='center', va='center', fontsize=16, fontweight='bold')
    # Set global legend
    custom_lines = [Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='green', lw=2), Line2D([0], [0], color='blue', lw=2)]
    fig.legend(custom_lines, ['X', 'Y', 'Z'], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved as '{save_path}'")
    return fig

def plot_hand_axis_raw(hand_dir:str, axis: str, file_type: str, max_files: int = 5, save_path: str = None) -> plt.Figure:
    """Plot raw data for a specific hand and axis across multiple files. Plot with dual smoothing layers:
    1. Raw Data: Faint background to show noise levels.
    2. Moving Average: Trend line for general behavior.
    3. Savitzky-Golay: Smooth line that preserves peaks and signal features.

    Args:
        hand_dir: Directory containing the hand's CSV files (Left or Right).
        axis: The axis to plot ('x', 'y', or 'z').
        file_type: "accel" or "gyro" to select the appropriate files.
        max_files: Maximum number of files to plot.
        save_path: Optional path to save the figure.
    """

    # Validation
    if axis not in ('x', 'y', 'z'):
        raise ValueError("axis must be 'x', 'y', or 'z'")
    if file_type not in ("accel", "gyro"):
        raise ValueError("file_type must be 'accel' or 'gyro'")

    y_limits = {
        "accel": {
            "x": (-2, 3),
            "y": (7.5, 11),
            "z": (-2, 5)
        },
        "gyro": {
            "x": (-1, 1),
            "y": (-1, 1),
            "z": (-0.5, 0.9)
        }
    }
    current_ylim = y_limits[file_type][axis]

    # Search for files
    pattern = f"**/*{file_type}.csv"
    files = sorted(glob.glob(os.path.join(hand_dir, pattern), recursive=True))[:max_files]
    
    if not files:
        print(f"No {file_type} files found in {hand_dir}")
        return None

    # Setup the figure and axes
    fig, axes = plt.subplots(len(files), 1, figsize=(14, 4 * len(files)), squeeze=False)
    colors = {"x": "red", "y": "green", "z": "blue"}
    main_color = colors[axis]

    for i, fpath in enumerate(files):
        ax = axes[i, 0]
        try:
            # Assuming _load_sensor_csv is defined elsewhere in your script
            df = _load_sensor_csv(fpath)
            col = f"{file_type}_{axis}"
            
            if col not in df.columns:
                ax.set_title(f"Column {col} missing in {os.path.basename(fpath)}")
                continue

            # Process time axis (converting nanoseconds to seconds)
            df["time_sec"] = df["timestamp"].cumsum() / 1e9
            raw_data = df[col].values
            time_axis = df["time_sec"].values

            # --- Smoothing Calculations ---
            
            # 1. Simple Moving Average (SMA)
            ma_window = 100
            ma_data = df[col].rolling(window=ma_window, center=True).mean()
            
            # 2. Savitzky-Golay Filter
            # window_length must be odd. Adjust 51 based on your sampling rate (Hz)
            sg_window = min(201, len(df) // 2 * 2 - 1) 
            if sg_window > 3:
                savgol_data = savgol_filter(raw_data, sg_window, polyorder=2)
            else:
                savgol_data = raw_data

            # --- Layered Plotting ---
            
            # Layer 1: Raw noisy data (Very faint)
            ax.plot(time_axis, raw_data, color=main_color, alpha=0.6, 
                    label='Raw (Noisy)', linewidth=0.8)
            
            # Layer 2: Moving Average (Black dashed line for trend)
            ax.plot(time_axis, ma_data, color='black', alpha=0.8, 
                    label=f'Moving Avg (w={ma_window})')
            
            # Layer 3: Savitzky-Golay (Bold solid line for filtered signal)
            ax.plot(time_axis, savgol_data, color='Magenta', alpha=0.8, 
                    label='Savitzky-Golay')

            # Individual Subplot Styling
            ax.set_ylim(current_ylim)
            ax.set_title(f"File: {os.path.basename(fpath)}", loc='left', fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper right', fontsize='small', ncol=3, frameon=True)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error processing file: {e}", ha='center', va='center')

    # --- Global Figure Styling ---
    unit = 'm/s²' if file_type == 'accel' else 'rad/s'
    
    # Global Title
    fig.suptitle(f'SENSOR ANALYSIS | Hand: {hand_dir.upper()} | Axis: {axis.upper()} | Mode: {file_type.upper()}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Global X-label (Only on the bottom plot to save space)
    axes[-1, 0].set_xlabel('Time (seconds)', fontsize=12)
    
    # Global Y-label positioned in the middle of the figure
    fig.text(0.01, 0.5, f'{file_type.capitalize()} Magnitude ({unit})', 
             va='center', rotation='vertical', fontsize=12, fontweight='bold')

    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Enhanced plot saved to: {save_path}")
    
    return fig


def plot_hand_stats_bars(hand_dir: str, file_type: str, stat_name: str, max_files: int = 5, save_path: str = None) -> plt.Figure:
    """
    Generate bar plots for a specific statistical metric across multiple files.
    Each file gets its own subplot showing X, Y, and Z axis values for the chosen statistic.

    Args:
        hand_dir: Directory containing the CSV files (e.g., 'Left' or 'Right').
        file_type: "accel" or "gyro" to select the sensor data.
        stat_name: The statistic to calculate (mean, std, variance, min, max, 
                   delta_min_max, count_negative, count_positive).
        max_files: Maximum number of files to process and plot.
        save_path: Optional path to save the generated figure.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    
    # 1. Define and validate supported statistics
    supported_stats = [
        'mean', 'std', 'variance', 'min', 'max', 'median',
        'delta_min_max', 'count_negative', 'count_positive'
    ]
    if stat_name not in supported_stats:
        raise ValueError(f"stat_name must be one of {supported_stats}")

    # 2. Search for files matching the pattern
    pattern = f"**/*{file_type}.csv"
    files = sorted(glob.glob(os.path.join(hand_dir, pattern), recursive=True))[:max_files]
    
    if not files:
        print(f"Warning: No files found in directory '{hand_dir}' matching type '{file_type}'")
        return None

    # 3. Initialize the figure (One subplot per file)
    fig, axes = plt.subplots(len(files), 1, figsize=(10, len(files) * 3.5), squeeze=False)
    
    # Define axis mapping and corresponding colors
    axes_names = ['x', 'y', 'z']
    axis_colors = ['red', 'green', 'blue'] # X=Red, Y=Green, Z=Blue

    # 4. Process each file
    for i, fpath in enumerate(files):
        try:
            # Assuming _load_sensor_csv is defined in your environment
            df = _load_sensor_csv(fpath)
            ax = axes[i, 0]
            
            stat_values = []
            for axis in axes_names:
                col = f"{file_type}_{axis}"
                if col in df.columns:
                    series = df[col]
                    
                    # Logic for calculating the specific statistic
                    if stat_name == 'mean': val = series.mean()
                    elif stat_name == 'std': val = series.std()
                    elif stat_name == 'variance': val = series.var()
                    elif stat_name == 'min': val = series.min()
                    elif stat_name == 'max': val = series.max()
                    elif stat_name == 'median': val = series.median()
                    elif stat_name == 'delta_min_max': val = series.max() - series.min()
                    elif stat_name == 'count_negative': val = (series < 0).sum()
                    elif stat_name == 'count_positive': val = (series > 0).sum()
                    
                    stat_values.append(val)
                else:
                    stat_values.append(0) # Default for missing columns

            # 5. Create the Bar Plot
            bars = ax.bar(axes_names, stat_values, color=axis_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
            
            # Add value labels on top of each bar for clarity
            label_format = '%d' if 'count' in stat_name else '%.3f'
            ax.bar_label(bars, fmt=label_format, padding=3, fontsize=10, fontweight='bold')
            
            # Subplot styling
            ax.set_title(f"File: {os.path.basename(fpath)}", fontsize=12, loc='left')
            ax.set_ylabel(stat_name.replace('_', ' ').title())
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
            # Add a reference line at 0 for stats that can be negative
            ax.axhline(0, color='black', linewidth=0.8)

        except Exception as e:
            # Display error message within the subplot if file loading fails
            axes[i, 0].text(0.5, 0.5, f"Error processing file: {e}", 
                           ha='center', va='center', color='red')

    # 6. Global figure styling
    main_title = f"{hand_dir.upper()} Hand Data - {file_type.upper()} {stat_name.replace('_', ' ').title()}"
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=1.02)
    
    fig.tight_layout()
    
    # 7. Save figure if a path is provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Statistics plot successfully saved to: {save_path}")
        
    return fig


def plot_axis_timeseries(left_dir: str, right_dir: str, axis: str, max_files: int = 5, save_path: str = None) -> plt.Figure:
    """Plot timeseries for a specific axis from accel files in separate subplots.

    Creates two subplots side by side: left subplot shows all left-hand files,
    right subplot shows all right-hand files. All files on the same subplot share
    the same timeline, each with a unique color.

    Args:
        left_dir: Directory containing left-hand accel CSV files.
        right_dir: Directory containing right-hand accel CSV files.
        axis: The axis to plot ('x', 'y', or 'z').
        max_files: Maximum number of files to plot from each side.
        save_path: Optional path to save the figure.
    Returns:
        matplotlib Figure object.
    """
    if axis not in ('x', 'y', 'z'):
        raise ValueError("axis must be 'x', 'y', or 'z'")

    pattern = "**/*accel.csv"
    left_files = sorted(glob.glob(os.path.join(left_dir, pattern), recursive=True))[:max_files]
    right_files = sorted(glob.glob(os.path.join(right_dir, pattern), recursive=True))[:max_files]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Colors for files
    colors = ['red', 'orange', 'yellow', 'pink', 'purple', 'blue', 'green', 'cyan', 'magenta', 'brown']

    # Plot left files
    ax = axes[0]
    for i, fpath in enumerate(left_files):
        try:
            df = _load_sensor_csv(fpath)
            col = f"accel_{axis}"
            if col in df.columns:
                # Convert timestamp deltas to cumulative seconds
                df["timestamp"] = df["timestamp"].cumsum() / 1e9
                
                label = os.path.basename(fpath)
                ax.plot(df['timestamp'], df[col], color=colors[i % len(colors)],
                        linewidth=1, label=label)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Acceleration')
    ax.set_title(f'Left Hand - {axis.upper()}-axis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot right files
    ax = axes[1]
    for i, fpath in enumerate(right_files):
        try:
            df = _load_sensor_csv(fpath)
            col = f"accel_{axis}"
            if col in df.columns:
                # Convert timestamp deltas to cumulative seconds
                df["timestamp"] = df["timestamp"].cumsum() / 1e9
                
                label = os.path.basename(fpath)
                ax.plot(df['timestamp'], df[col], color=colors[i % len(colors)],
                        linewidth=1, label=label)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Acceleration')
    ax.set_title(f'Right Hand - {axis.upper()}-axis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Acceleration {axis.upper()}-axis Timeseries Comparison', fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved as '{save_path}'")
    return fig




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
    # # default directories relative to this script
    # base = os.path.dirname(os.path.abspath(__file__))
    # dirs = [os.path.join(base, "Left"), os.path.join(base, "Right")]
    
    # # Individual file analysis
    # walk_and_analyze(dirs)
    
    # # Collect data for comparison
    # left_data = collect_data_for_root(dirs[0])
    # right_data = collect_data_for_root(dirs[1])
    
    # # Comparison analysis for each root directory
    # print(f"\n\n{'#'*80}")
    # print("COMPARISON TABLES FOR EACH ROOT DIRECTORY")
    # print(f"{'#'*80}")
    # for root_dir in dirs:
    #     compare_statistics_in_root(root_dir)
    
    # # Plot comparison
    # print(f"\n\n{'#'*80}")
    # print("PLOTTING LEFT VS RIGHT COMPARISON")
    # print(f"{'#'*80}")
    # plot_left_vs_right_comparison(left_data, right_data)
    
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
    
    
    for hand in ['Left', 'Right']:
        for file_type in ['accel', 'gyro']:
            for stat in ['mean', 'std', 'variance', 'min', 'max', 'median', 'delta_min_max', 'count_negative', 'count_positive']:
                save_path = f"Figures/Statistics/{hand}_{file_type}_{stat}.png"
                plot_hand_stats_bars(hand_dir=hand, file_type=file_type, stat_name=stat, max_files=5, save_path=save_path)
                

if __name__ == "__main__":
    main()
