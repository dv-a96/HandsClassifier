import os
import glob
import pandas as pd

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
    walk_and_analyze(dirs)


if __name__ == "__main__":
    main()
