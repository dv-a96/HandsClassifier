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
    עוברת על Left ואז Right, מוסיפה טור הפרשי זמן ושומרת את הקובץ.
    """
    hands = ['Left', 'Right']
    processed_stats = []

    for hand in hands:
        hand_dir = os.path.join(base_path, hand)
        pattern = f"**/*{file_type}.csv"
        files = sorted(glob.glob(os.path.join(hand_dir, pattern), recursive=True))
        
        for fpath in files:
            if "resampled" in fpath.lower():
                header = 0  # קבצי Resampled כבר כוללים כותרת
            else:
                header = None
            df = _load_sensor_csv(fpath, header=header)
            
            # חישוב ההפרשים (בננו-שניות, או ביחידות של הטור המקורי)
            # אם timestamp הוא זמן מוחלט - זה ייתן את ה-Sampling Interval
            # אם timestamp הוא כבר הפרש - זה ייתן את ה-Jitter
            df['ts_diff'] = pd.to_numeric(df['timestamp'], errors='coerce').diff()
            
            
            # איסוף סטטיסטיקה לגרף
            valid_diffs = df['ts_diff'].dropna()
            processed_stats.append({
                'filename': os.path.basename(fpath),
                'hand': hand,
                'min': valid_diffs.min(),
                'max': valid_diffs.max(),
                'mean': valid_diffs.mean(),
                'all_diffs': valid_diffs.values # נשמור את הכל לטובת ה"דרך החכמה"
            })
            
    return processed_stats


def plot_sampling_consistency(stats_list: list, save_path: str=None):
    """
    מציגה את טווח ההפרשים לכל קובץ.
    """
    filenames = [s['filename'] for s in stats_list]
    all_data = [s['all_diffs'] for s in stats_list]
    colors = ['blue' if s['hand'] == 'Left' else 'orange' for s in stats_list]

    plt.figure(figsize=(15, 7))
    
    # הדרך החכמה: Boxplot
    # מראה חציון, רבעונים ו-Outliers (נקודות חריגות שמעידות על איבוד דגימות)
    bp = plt.boxplot(all_data, tick_labels=filenames, patch_artist=True)
    
    # צביעה לפי יד
    for i in range(len(bp['boxes'])):
            # צביעת גוף הקופסה (שכרגע לא רואים כי היא שטוחה)
            bp['boxes'][i].set_facecolor(colors[i])
            bp['boxes'][i].set_alpha(0.5)
            
            # --- התיקון כאן: צביעת קו החציון בצבע של היד ---
            # זה יגרום לקווים של יד שמאל להפוך לכחולים!
            bp['medians'][i].set_color(colors[i]) 
            bp['medians'][i].set_linewidth(2)

    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Timestamp (ns)')
    plt.title('Sampling Consistency Check (Boxplot per File)\nBlue: Left Hand | Orange: Right Hand')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_sampling_rate_histograms(stats_list, bins=50, save_path=None):
    """
    Generates a histogram for each file in the stats_list to visualize sampling consistency.
    Includes vertical lines for median and quartiles, and a summary box for min/max/uniques.
    
    Parameters:
    - stats_list (list): List of dictionaries containing 'filename', 'all_diffs', 'hand', etc.
    - bins (int): Number of bins for the histogram.
    - save_path (str): Full path to save the resulting plot. If None, displays the plot.
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
        # Using xaxis_transform to place text at the top of the plot area
        trans = ax.get_xaxis_transform() 
        ax.text(median, 0.95, f' Med: {median:,.0f}', color='red', transform=trans, fontweight='bold', ha='left')
        ax.text(q1, 0.88, f' Q1: {q1:,.0f}', color='green', transform=trans, ha='right')
        ax.text(q3, 0.88, f' Q3: {q3:,.0f}', color='green', transform=trans, ha='left')
        
        # 5. Create Summary Statistics Box
        stats_text = (f"Min: {d_min:,.0f} ns\n"
                      f"Max: {d_max:,.0f} ns\n"
                      f"Unique Diffs: {unique_counts}")
        
        # Place the text box in the upper right corner of the individual subplot
        ax.text(0.95, 0.75, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        # Set plot metadata
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.set_title(f"File: {s['filename']} ({s['hand']})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time Difference (ns)")
        ax.set_yscale('log')  # Log scale for better visibility of outliers
        ax.set_ylabel("Log Frequency")
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Hide any unused subplots in the grid
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    
    # Save or Show logic
    if save_path:
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Histogram plot saved successfully to: {save_path}")
    else:
        plt.show()

def resample_and_interpolate_file(df, target_interval_ns=2_000_000):
    """
    מבצעת יישור (Alignment) של הנתונים לרשת זמן קבועה וביצוע אינטרפולציה.
    """
    # 1. הבטחת טיפוס נתונים מספרי וניקוי כפילויות זמן
    df = df.copy()
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    # אם יש שתי דגימות על אותה ננו-שנייה, ניקח את הממוצע שלהן
    df = df.groupby('timestamp').mean().reset_index()
    
    # 2. קביעת האינדקס כזמן
    df = df.set_index('timestamp')
    
    # 3. יצירת ציר הזמן ה"מושלם"
    # מתחילים מהדגימה הראשונה וקופצים ב-2ms עד האחרונה
    start_time = df.index.min()
    end_time = df.index.max()
    perfect_grid = np.arange(start_time, end_time, target_interval_ns)
    
    # 4. ביצוע ה-Resampling:
    # אנחנו משלבים את האינדקס הקיים עם האינדקס החדש כדי לא לאבד מידע בזמן החישוב
    combined_index = np.unique(np.concatenate([df.index, perfect_grid]))
    df_aligned = df.reindex(combined_index)
    
    # 5. אינטרפולציה לינארית למילוי כל החורים (גם הברחנים הגבוהים וגם הנמוכים)
    df_interpolated = df_aligned.interpolate(method='linear')
    
    # 6. חיתוך רק של הנקודות שיושבות בדיוק על הרשת שלנו
    df_final = df_interpolated.loc[perfect_grid]
    
    return df_final.reset_index()


def apply_butterworth_highpass(data, cutoff_hz, fs, order=4):
    # 1. חישוב תדר נייקוויסט (Nyquist Frequency)
    # זהו התדר המקסימלי שניתן למדוד, והוא תמיד מחצית מקצב הדגימה.
    nyq = 0.5 * fs
    
    # 2. נרמול תדר החיתוך (Normalized Cutoff)
    # פונקציות העיבוד הדיגיטלי מצפות לערך בין 0 ל-1, 
    # כאשר 1 מייצג את תדר נייקוויסט.
    normal_cutoff = cutoff_hz / nyq
    
    # 3. תכנון המסנן (Design)
    # הפונקציה butter מחזירה שני מערכים של מקדמים: b ו-a.
    # המקדמים האלו הם בעצם ה"נוסחה המתמטית" של הפילטר.
    # btype='high' אומר למחשב שאנחנו רוצים High-pass (להעביר גבוהים, לחסום נמוכים).
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    # 4. הפעלת המסנן (Execution)
    # filtfilt (ולא lfilter) עוברת על הנתונים קדימה ואז אחורה.
    # זה מבטיח שלא יהיה עיכוב (Phase Shift) בגרף - התנועה תישאר בדיוק באותו זמן.
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data



def apply_highpass_to_all_files(root_dir: str, save_dir: str):
    for hand in ['Left', 'Right']:
        for file_type in ['accel', 'gyro']:
            pattern = f"**/*{file_type}.csv"
            files = sorted(glob.glob(os.path.join(root_dir, f'{hand}', pattern), recursive=True))
            for file in files:
                df = _load_sensor_csv(file, header=0)  # Assuming resampled files have headers
                if file_type == 'accel':
                    cutoff = 0.6  # עלייה קלה לניקוי גרביטציה טוב יותר
                    order = 4
                    cols_to_filter = ['accel_x', 'accel_y', 'accel_z'] # חובה את כל הצירים!
                else:
                    cutoff = 0.1  # מעולה לניקוי Bias
                    order = 2
                    cols_to_filter = ['gyro_x', 'gyro_y', 'gyro_z']
                
                for col in cols_to_filter:
                        data_col = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        df[col] = apply_butterworth_highpass(data_col, cutoff_hz=cutoff, fs=500, order=order)
                df.to_csv(f'{save_dir}/{hand}/{file.split("/")[-1]}', index=False)

apply_highpass_to_all_files(root_dir='Resampled', save_dir='Clean')
# plot_sampling_rate_histograms(stats_list=add_timestamp_diff_column(base_path='./', file_type='accel'), bins=50, save_path='Figures/Raw/sampling_consistency_accel_histograms.png')