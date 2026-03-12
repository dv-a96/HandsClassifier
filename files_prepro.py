import pandas as pd
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
            df = _load_sensor_csv(fpath)
            
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


def plot_sampling_consistency(stats_list):
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
    plt.show()

    # 1. עיבוד והוספת הטור
stats = add_timestamp_diff_column('./', 'accel')

# 2. הצגת הגרף
plot_sampling_consistency(stats)