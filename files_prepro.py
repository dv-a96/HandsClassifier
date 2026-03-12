import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
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


# for hand in ['Left', 'Right']:
#     for file_type in ['accel', 'gyro']:
#         pattern = f"**/*{file_type}.csv"
#         files = sorted(glob.glob(os.path.join(f'{hand}', pattern), recursive=True))
#         for file in files:
#             file_name = file.split('/')[-1]
#             raw_df = _load_sensor_csv(file)
#             inter_df = resample_and_interpolate_file(raw_df, target_interval_ns=2_000_000)
#             inter_df.to_csv(f'Resampled/{hand}/res_{file_name}', index=False)


plot_sampling_consistency(add_timestamp_diff_column('.', 'accel'), save_path='Figures/sampling_consistency_raw_accel.png')
plot_sampling_consistency(add_timestamp_diff_column('Resampled', 'accel'), save_path='Figures/sampling_consistency_resampled_accel.png')