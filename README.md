# HandsClassifier
This project aims to detect handedness (which hand is holding the device) during smartphone video recording. The classification is performed using Machine Learning models trained on accelerometer and gyroscope data.


## Data Acquizition
The dataset was collected using the OpenCamera Sensors application [^1] [^2], which enables synchronized recording of IMU sensor data alongside video footage.
To ensure consistency across the dataset, a standardized movement protocol was followed for both left and right-hand recordings:

**Initial Position:** The device is held with a straight arm directly in front of the face.

**Movement:** The device is brought toward the nose and then extended back to the initial position.

**Variations:** Data was captured in both sitting and standing postures.

For every recorded session, the following files were generated:

`vidname.mp4:` The raw video file.

`vidname_accel.csv:` Accelerometer data (X, Y, Z axes) with corresponding timestamps.

`vidname_gyro.csv:` Gyroscope data (X, Y, Z axes) with corresponding timestamps.

Each CSV file consists of four columns: three representing the spatial axes ($X, Y, Z$) and one for the high-precision timestamps used for synchronization.


## Files Cleaning
### Sampling inconsistency 

### Resampling and Interpolation 

### Smoothing
many under data analysis 
## Data Analysis
The `analyze_data.py` script provides comprehensive statistical analysis of the IMU sensor data collected during the study.

### Comparative Analysis
The script conducts a comparison of sensor data between Left and Right hand recordings by:
1. Collecting statistics from all files within each directory (Left/Right subdirectories)
2. Grouping data by file type (accelerometer/gyroscope) and axis (X, Y, Z)
3. Displaying comparison tables showing count and mean values for each file

This allows for identification of differences in sensor behavior between left-handed and right-handed recordings.

### Visualization
The script generates a comprehensive visualization saved as `left_vs_right_comparison.png`, which includes 6 subplots (3 for accelerometer and 3 for gyroscope) for all the axis of the sensors. Each subplot contain the mean value over the spesific axes for all the files:

![left vs right sensors values plot](/left_vs_right_comparison.png)

This visualization helps identify patterns and differences in accelerometer and gyroscope readings between left and right-handed device usage.

___
[^1] A. Akhmetyanov, A. Kornilova, M. Faizullin, D. Pozo and G. Ferrer, "Sub-millisecond Video Synchronization of Multiple Android Smartphones," 2021 IEEE Sensors, 2021, pp. 1-4

[^2] Faizullin, M.; Kornilova, A.; Akhmetyanov, A.; Ferrer, G. Twist-n-Sync: Software Clock Synchronization with Microseconds Accuracy Using MEMS-Gyroscopes. Sensors 2021, 21, 68
