import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
                    if "res" in files[i].lower():
                        header=0
                    else:
                        header=None
                    df = _load_sensor_csv(files[i], header=header)
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
                ax.set_xlabel('Time (ns)')
                ax.set_ylabel('Acceleration(m/s^2)' if file_type == 'accel' else 'Angular Velocity(rad/s)')
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


def _compute_smoothing(df: pd.DataFrame, file_type: str, ma_window: int = 100, sg_window: int = 201) -> pd.DataFrame:
    """Add smoothing columns (moving average + Savitzky-Golay) to a sensor dataframe.

    The dataframe is expected to contain a "timestamp" column (nanosecond deltas) and
    axis columns like "accel_x" / "gyro_x".
    """

    prefix = "accel" if file_type == "accel" else "gyro"
    for axis in ["x", "y", "z"]:

        df[f"{axis}_ma"] = df[axis].rolling(window=ma_window, center=True).mean()

        # Savitzky-Golay smoothing (window must be odd)
        sg_window_adj = min(sg_window, len(df) // 2 * 2 - 1)
        if sg_window_adj < 3:
            df[f"{axis}_sg"] = df[axis]
        else:
            df[f"{axis}_sg"] = savgol_filter(df[axis].values, sg_window_adj, polyorder=2)

    return df


def smooth_and_save_hand_data(hand_dir: str, save_dir: str, file_type: str, max_files: int = 5) -> None:
    """Process files in hand_dir, compute smoothing for all axes, and save each smoothed DF as a separate CSV in save_dir.

    For each file, saves a CSV named 'basename_smoothed.csv' containing raw data, moving average, and Savitzky-Golay for all axes.
    """
    pattern = f"**/*{file_type}.csv"
    files = sorted(glob.glob(os.path.join(hand_dir, pattern), recursive=True))[:max_files]
    if not files:
        print(f"No {file_type} files found in {hand_dir}")
        return

    os.makedirs(save_dir, exist_ok=True)

    for fpath in files:
        df = _load_sensor_csv(fpath, header=0)  # Assuming smoothed files have headers
        df = df.copy()
        df = _compute_smoothing(df, file_type=file_type)
        
        basename = f"smoothed_{os.path.basename(fpath)}"
        out_path = os.path.join(save_dir, basename)
        df.to_csv(out_path, index=False)
        print(f"Smoothed data saved to: {out_path}")


def plot_axis_data(file_path: str, axis: str, file_type: str, raw: bool = False, save_path: str = None, ax=None) -> plt.Figure:
    """Plot the axis data of a given file on a specific axis or a new figure."""
    try:
        df = _load_sensor_csv(file_path, header=0 if not raw else None)
        # המרת זמן - לוגיקה מאוחדת
        time_col = "timestamp" if not raw else df.columns[-1]
        df["time_sec"] = df[time_col].cumsum() / 1e9
        filename = os.path.basename(file_path)
        if not raw:
            if filename.startswith("res") or filename.startswith("cl"):
                col_name = f"{axis}"
            elif filename.startswith("smoothed"):
                col_name = f"{axis}_sg"
        else:
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            col_name = df.columns[axis_map[axis]]

        if col_name not in df.columns:
            print(f"Column {col_name} not found in {file_path}")
            return None

        # אם לא קיבלנו AX, ניצור figure חדש
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.get_figure()

        ax.plot(df["time_sec"], df[col_name], color='blue', label=f'{axis.upper()}-axis', linewidth=1)
        ax.set_xlabel('Time (ns)')
        ylabel = 'Acceleration(m/s^2)' if file_type == 'accel' else 'Angular Velocity(rad/s)'
        ax.set_ylabel(ylabel)
        ax.set_title(f"{os.path.basename(file_path)}", loc='left', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path and fig:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    except Exception as e:
        print(f"Error in plot_axis_data: {e}")
        return None

def plot_hand_data(hand_dir: str, axis: str, file_type: str, max_files: int = 5, raw: bool = False, save_path: str = None) -> plt.Figure:
    """Plot data for a specific hand and axis across multiple files using plot_axis_data."""
    pattern = f"**/*{file_type}.csv"
    files = sorted(glob.glob(os.path.join(hand_dir, pattern), recursive=True))[:max_files]
    
    if not files:
        print(f"No {file_type} files found in {hand_dir}")
        return None

    # יצירת הקנבס עם מספר subplots כמספר הקבצים
    fig, axes = plt.subplots(len(files), 1, figsize=(14, 4 * len(files)), squeeze=False)
    
    for i, fpath in enumerate(files):
        # קריאה לפונקציה המקורית ושליחת ה-AX הספציפי
        plot_axis_data(file_path=fpath, axis=axis, file_type=file_type, raw=raw, ax=axes[i, 0])

    fig.suptitle(f'Hand: {os.path.basename(hand_dir).upper()} | Axis: {axis.upper()} | Mode: {file_type.upper()}', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Combined plot saved as '{save_path}'")
        
    return fig


def plot_hand_axis_raw(hand_dir:str, axis: str, file_type: str, max_files: int = 5, raw: str = False, save_path: str = None) -> plt.Figure:
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
    header = None if raw else 0  # If raw=True, we expect no header. If raw=False, we expect smoothed files with headers.
    for i, fpath in enumerate(files):
        ax = axes[i, 0]
        try:
            df = _load_sensor_csv(fpath, header=header)
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


def plot_hand_stats_bars(hand_dir: str, file_type: str, stat_name: str, max_files: int = 5, smooth=False, save_path: str = None) -> plt.Figure:
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
        'delta_min_max', 'count_negative', 'count_positive', 'intensity', 'skewness',
        'argmax', 'argmin', 'zcr'
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
    axes_names = ['x', 'y', 'z'] if smooth==False else ['x_ma', 'y_ma', 'z_ma']  # Use smoothed column names if needed
    axis_colors = ['red', 'green', 'blue'] # X=Red, Y=Green, Z=Blue

    # 4. Process each file
    for i, fpath in enumerate(files):
        try:
            # Assuming _load_sensor_csv is defined in your environment
            header=None if smooth==False else 0  # If smooth=False, we expect no header. If smooth=True, we expect smoothed files with headers.
            df = _load_sensor_csv(fpath, header=header)
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
                    elif stat_name == 'intensity': val = (series**2).mean()  # Average intensity (mean of squares)
                    elif stat_name == 'skewness': val = series.skew()
                    elif stat_name == 'argmax': val = series.idxmax()/len(series)  # Normalized index of max value
                    elif stat_name == 'argmin': val = series.idxmin()/len(series)  # Normalized index of min value
                    elif stat_name == 'zcr': val = ((series[:-1].values * series[1:].values) < 0).sum() / (len(series) - 1)  # Zero-crossing rate
                    
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


def create_stats_dfs(root_dir: str, save_dir: str) -> None:
    """Create statistics DataFrames for Left and Right hands from smoothed data.
    
    Each DF contains: filename, axis, mean, std, variance, min, max, median, delta_min_max, count_negative, count_positive
    Saves as left_stats.csv and right_stats.csv in save_dir.
    """
    stats_list = ['mean', 'std', 'variance', 'min', 'max', 'median', 'delta_min_max', 'count_negative', 'count_positive', 'intensity', 'skewness',
                  'argmax', 'argmin', 'zcr']
    
    for hand in ['Left', 'Right']:
        hand_dir = f'{root_dir}/{hand}'
        if not os.path.isdir(hand_dir):
            print(f"Directory {hand_dir} not found")
            continue
        for pattern in ["**/*accel.csv", "**/*gyro.csv"]:
            data = []
            files = sorted(glob.glob(os.path.join(hand_dir, pattern), recursive=True))
            if root_dir == 'Smoothed':
                axes = ['x_sg', 'y_sg', 'z_sg']
            else:
                axes = ['x', 'y', 'z']
            for fpath in files:
                try:
                    df = pd.read_csv(fpath)
                    filename = os.path.basename(fpath)
                    file_type = 'accel' if 'accel' in filename else 'gyro'
                    
                    for axis in axes:
                        col = f"{file_type}_{axis}"
                        if col in df.columns:
                            series = df[col]
                            row = {'filename': filename, 'axis': axis}
                            for stat in stats_list:
                                if stat == 'mean':
                                    row[stat] = series.mean()
                                elif stat == 'std':
                                    row[stat] = series.std()
                                elif stat == 'variance':
                                    row[stat] = series.var()
                                elif stat == 'min':
                                    row[stat] = series.min()
                                elif stat == 'max':
                                    row[stat] = series.max()
                                elif stat == 'median':
                                    row[stat] = series.median()
                                elif stat == 'delta_min_max':
                                    row[stat] = series.max() - series.min()
                                elif stat == 'count_negative':
                                    row[stat] = (series < 0).sum()/len(series)  # Count negative per second
                                elif stat == 'count_positive':
                                    row[stat] = (series > 0).sum()/len(series)  # Count positve per second
                                elif stat == 'intensity':
                                    row[stat] = (series**2).mean()  # Average intensity (mean of squares)
                                elif stat == 'skewness':
                                    row[stat] = series.skew()
                                elif stat == 'argmax':
                                    row[stat] = series.idxmax()/len(series)  # Normalized index of max value
                                elif stat == 'argmin':
                                    row[stat] = series.idxmin()/len(series)  # Normalized index of min value
                                elif stat == 'zcr':
                                    row[stat] = ((series[:-1].values * series[1:].values) < 0).sum() / (len(series) - 1)  # Zero-crossing rate
                            data.append(row)
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")
            
            if data:
                stats_df = pd.DataFrame(data)
                out_path = os.path.join(save_dir, f"{hand.lower()}_{file_type}_stats.csv")
                stats_df.to_csv(out_path, index=False)
                print(f"Stats DF saved to: {out_path}")



def plot_stats_outliers(stats_csv_path, axis_name='z_sg', save_path=None):
    """
    Identifies and visualizes outliers for each statistical metric using the IQR method.
    Annotates outlier points with their respective filenames for easy debugging.
    """
    df = pd.read_csv(stats_csv_path)
    # Filter data for the specific axis (e.g., 'z_sg')
    df_axis = df[df['axis'] == axis_name].copy()
    
    if df_axis.empty:
        print(f"No data for axis {axis_name}")
        return

    # List of metrics to evaluate for outliers
    metrics = ['mean', 'std', 'variance', 'min', 'max', 'median', 'delta_min_max', 'count_negative', 'count_positive', 'intensity', 'skewness',
               'argmax', 'argmin', 'zcr']
    cols = 3
    rows = (len(metrics) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        series = df_axis[metric].dropna()
        if series.empty:
            ax.set_visible(False)
            continue
            
        # Calculate Interquartile Range (IQR) for outlier detection
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        # Standard Tukey's fences: 1.5 * IQR
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        
        # Plot the boxplot - fixed tick_labels to avoid Matplotlib warnings
        bp = ax.boxplot(series, patch_artist=True, tick_labels=[metric])
        bp['boxes'][0].set(facecolor='lightgreen', alpha=0.5)
        
        # Identify files that fall outside the calculated bounds
        outliers = df_axis[(df_axis[metric] < lower_bound) | (df_axis[metric] > upper_bound)]
        
        # Annotate each outlier point with its original filename
        for _, row in outliers.iterrows():
            ax.annotate(row['filename'], xy=(1, row[metric]), xytext=(15, 0),
                        textcoords='offset points', fontsize=7, color='darkred',
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.2))
        
        ax.set_title(f'Metric: {metric}', fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # --- Legend Construction ---
    # Create visual proxy elements for the plot legend
    legend_elements = [
        mpatches.Patch(facecolor='lightgreen', alpha=0.5, edgecolor='black', label='IQR Range (25%-75%)'),
        Line2D([0], [0], color='orange', lw=2, label='Median Value'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markeredgecolor='black', label='Outlier Point'),
        Line2D([0], [0], color='red', marker='>', markersize=8, label='File Label (Outlier Name)', linestyle='none', alpha=0.6)
    ]
    
    # Position the global legend outside the subplots
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.98, 0.98), 
               ncol=1, fontsize=10, frameon=True, shadow=True, title="Legend")

    fig.suptitle(f'Detailed Outlier Analysis | Axis: {axis_name} | File: {os.path.basename(stats_csv_path)}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Subplots saved to: {save_path}")
    else:
        plt.show()

def create_global_summary(stats_dir: str, out_path: str = None) -> pd.DataFrame:
    """
    Aggregates individual statistics files into a single global summary table.
    Categorizes data by Hand (Left/Right), Sensor (Accel/Gyro), and Axis.
    """
    all_data = []
    # Find all individual stats CSV files
    files = glob.glob(os.path.join(stats_dir, "*_stats.csv"))
    
    for fpath in files:
        df = pd.read_csv(fpath)
        filename = os.path.basename(fpath)
        # Parse metadata (hand side and sensor type) from the filename
        parts = filename.split('_')
        df['hand'] = parts[0].capitalize()
        df['sensor'] = parts[1].capitalize()
        all_data.append(df)
    
    if not all_data:
        print("No statistics files found to summarize.")
        return None

    # Merge all individual dataframes into one large master dataframe
    master_df = pd.concat(all_data, ignore_index=True)

    # 1. Define groups and target columns for aggregation
    group_cols = ['hand', 'sensor', 'axis']
    # Select all numeric columns for calculation, excluding grouping and filename columns
    stat_cols = [c for c in master_df.columns if c not in group_cols + ['filename']]

    # 2. Map metrics using an aggregation dictionary
    # This ensures safe calculation of Mean and Std for every feature
    agg_dict = {col: ['mean', 'std'] for col in stat_cols}
    
    # 3. Execute Groupby and Aggregate
    # Groups data by Hand, Sensor, and Axis to find general patterns
    summary = master_df.groupby(group_cols).agg(agg_dict)

    # 4. Flatten MultiIndex column names for cleaner CSV output
    # Example: ('intensity', 'mean') becomes 'intensity_avg'
    summary.columns = [f"{col[0]}_{'avg' if col[1]=='mean' else 'std_dev'}" for col in summary.columns]
    summary = summary.reset_index()

    if out_path:
        # Ensure the output directory exists and save the final summary
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f"Global summary successfully saved to: {out_path}")
        
    return summary


def plot_hand_summery_comparison(summary_df: pd.DataFrame, sensor_type: str, metric: str, save_path: str=None):
    """
    Creates a bar plot comparing Left vs Right hand for a specific sensor and metric.
    Uses error bars to represent the standard deviation of the metric.
    """
    # 1. Filter data for the requested sensor type
    sensor_df = summary_df[summary_df['sensor'] == sensor_type.capitalize()]
    
    if sensor_df.empty:
        print(f"No data found for sensor: {sensor_type}")
        return

    axes_names = sensor_df['axis'].unique()
    hands = ['Left', 'Right']
    
    # Define relevant column names for mean and standard deviation
    avg_col = f"{metric}_avg"
    std_col = f"{metric}_std_dev"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(axes_names))  # Positions for the axes (x, y, z)
    width = 0.35  # Width of the bars
    
    # 2. Plot bars for each hand
    for i, hand in enumerate(hands):
        hand_data = sensor_df[sensor_df['hand'] == hand]
        # Ensure data is sorted according to the axis order
        hand_data = hand_data.set_index('axis').loc[axes_names]
        
        means = hand_data[avg_col]
        stds = hand_data[std_col]
        
        ax.bar(x + (i * width) - width/2, means, width, 
               yerr=stds, label=hand, capsize=5, alpha=0.8)

    # 3. Chart styling and formatting
    ax.set_xlabel('Axis')
    ax.set_ylabel(f'Value ({metric})')
    ax.set_title(f'Comparison of {metric.capitalize()} by Hand ({sensor_type})')
    ax.set_xticks(x)
    ax.set_xticklabels(axes_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()




def plot_comprehensive_hand_comparison(summary_df, sensor_type='Accel', save_path=None):
    """
    Generates a grid of bar plots comparing Left vs Right hand performance 
    across all available statistical metrics.
    
    Parameters:
    - summary_df (pd.DataFrame): The aggregated global summary dataframe.
    - sensor_type (str): 'Accel' or 'Gyro'.
    - save_path (str): Full file path to save the image. If None, the plot is displayed.
    """
    
    # 1. Filter data by the specific sensor type
    sensor_df = summary_df[summary_df['sensor'] == sensor_type.capitalize()]
    if sensor_df.empty:
        print(f"No data found for {sensor_type}")
        return

    # 2. Automatically identify all statistical metrics based on the '_avg' suffix
    metrics = [col.replace('_avg', '') for col in summary_df.columns if col.endswith('_avg')]
    
    # 3. Calculate grid dimensions (3 columns per row)
    num_metrics = len(metrics)
    n_cols = 3
    n_rows = (num_metrics + n_cols - 1) // n_cols
    
    # Create the figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()
    
    axes_names = sorted(sensor_df['axis'].unique())
    hands = ['Left', 'Right']
    colors = ['#1f77b4', '#ff7f0e'] # Standard Blue and Orange
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        avg_col = f"{metric}_avg"
        std_col = f"{metric}_std_dev"
        
        x = np.arange(len(axes_names))
        width = 0.35
        
        for j, hand in enumerate(hands):
            # Isolate data for each hand and ensure consistent axis ordering
            hand_data = sensor_df[sensor_df['hand'] == hand].set_index('axis').loc[axes_names]
            
            # Plot bars with error bars representing the standard deviation
            ax.bar(x + (j * width) - width/2, hand_data[avg_col], width,
                   yerr=hand_data[std_col], label=hand if i == 0 else "", 
                   capsize=4, color=colors[j], alpha=0.8)
        
        # Subplot formatting
        ax.set_title(f'{metric.replace("_", " ").capitalize()}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(axes_names)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        if i % n_cols == 0:
            ax.set_ylabel('Aggregated Value')

    # Add a global legend for the entire figure
    fig.legend(['Left Hand', 'Right Hand'], loc='upper center', bbox_to_anchor=(0.5, 1.02), 
               ncol=2, fontsize=16, frameon=True, shadow=True)

    # Hide any unused axis slots in the grid
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Comprehensive Feature Comparison: Left vs Right ({sensor_type})', 
                 fontsize=20, y=1.05, fontweight='black')
    
    plt.tight_layout()

    # 4. Save or Show logic
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive plot saved successfully to: {save_path}")
    else:
        plt.show()



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

    create_stats_dfs('Smoothed', 'Smoothed/Stats')
    create_global_summary('Smoothed/Stats', 'Smoothed/global_summery.csv')
    plot_comprehensive_hand_comparison(pd.read_csv('Smoothed/global_summery.csv'), 'gyro', 'Smoothed/gyro_stats_summery.png')
    plot_comprehensive_hand_comparison(pd.read_csv('Smoothed/global_summery.csv'), 'accel', 'Smoothed/accel_stats_summery.png')    
    # for hand in ['Left', 'Right']:
    #     for file_type in ['accel', 'gyro']:
    #         for stat in ['mean', 'std', 'variance', 'min', 'max', 'median', 'delta_min_max', 'count_negative', 'count_positive']:
    #             save_path = f"Figures/Smooth/Statistics/{hand}_{file_type}_{stat}.png"
    #             plot_hand_stats_bars(hand_dir=f'Smoothed/{hand}', file_type=file_type, stat_name=stat, max_files=5, smooth=True, save_path=save_path)


    
    # create_stats_dfs('Smoothed', 'Smoothed/Stats')
    # create_global_summary('Smoothed/Stats', 'Smoothed/global_summery.csv')


    # df = pd.read_csv('Smoothed/global_summery.csv')
    # plot_comprehensive_hand_comparison(df, 'gyro', 'Smoothed/gyro_stas_summery.png')
    # plot_comprehensive_hand_comparison(df, 'accel', 'Smoothed/accel_stas_summery.png')

    # for hand in ['Left', 'Right']:
    #     for file_type in ['accel', 'gyro']:
    #         for axis in ['x', 'y', 'z']:
    #             plot_stats_outliers(f'Smoothed/Stats/{hand.lower()}_{file_type}_stats.csv', f'{axis}_sg', f'Smoothed/{hand.lower()}_{file_type}_{axis}_sg_outliers.png')

# if __name__ == "__main__":
#     main()
