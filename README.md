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

To verify that there are no inconsistencies in the sampling rate of a file (caused by system load or other technical issues) we calculate a `diff` column for every file, represents the differance between the sampling time of each point. We plot the histogram of this column to see the disterbution of the values of the time differance between sampling - the sampling rate. If the sampling rate is consistant we expect to see one value (or at least narrow disterbution with no large differance betwwen values). But when we ploted the histograms of the files we saw that almost in every files there are some points with sampling rate around 4-5ms where the common range of the sampling is around 2 ms. This indicate that in most of the files there's at least one point that the device "missed a point" during the sampling.

![Sampling rate's histogram]()
### Resampling and Interpolation 

### Smoothing
many under data analysis

### Remove gravity acceleration

In accelerometer data, the measured acceleration includes both the device's motion and the constant gravitational acceleration. Since the videos were recorded in selfie mode (device held vertically), the gravity component primarily affects the Y-axis, appearing as a constant offset around 9.8 m/s² (depending on orientation).

To isolate the actual motion-induced acceleration, we applied a low-frequency Butterworth high-pass filter to remove the gravity component while preserving the dynamic motion signals. For the gyroscope data, which measures rotational velocity, no such correction was needed as gravity does not affect angular measurements.

### Sensor Data Preprocessing: Gravity and Bias Removal

This module handles the cleaning and normalization of raw inertial sensor data (Accelerometer and Gyroscope) to ensure high-fidelity motion analysis. The primary goal is to isolate the Linear Acceleration of the hand and remove the Sensor Bias from the rotational data.
1. Accelerometer: Gravity Removal (High-Pass Filtering)

Raw accelerometer data contains a constant component of approximately $9.8 \, m/s^2$ due to Earth's gravity. When the device is tilted, this gravity vector is distributed across the X, Y, and Z axes, masking the actual motion of the user.Method: A 4th-order Butterworth High-Pass Filter is applied to all three axes.Logic: Since gravity is a DC component (0 Hz) or changes very slowly during orientation shifts, the high-pass filter blocks frequencies below the cutoff while allowing rapid human movements to pass.Parameters: * Cutoff Frequency: $0.6 \, Hz$ (Optimized to remove gravity without attenuating slow intentional movements).Result: The output is Linear Acceleration, centered around $0 \, m/s^2$ when the device is at rest.

2. Gyroscope: Bias and Drift Removal

Gyroscope sensors often suffer from a "Static Bias"—a non-zero reading even when the device is perfectly still. Over time, this bias can "drift" due to thermal changes or electrical noise. If left uncorrected, these errors accumulate during integration, leading to massive inaccuracies in angular displacement.Method: A 2nd-order Butterworth High-Pass Filter.Logic: We employ a dynamic removal strategy. By setting a very low cutoff frequency, the filter continuously identifies the "average" offset (the Bias) and subtracts it from the signal in real-time.Parameters:Cutoff Frequency: $0.1 \, Hz$ (Designed to eliminate slow-moving sensor drift while preserving the integrity of rotational velocity).Result: A zero-mean rotational signal, significantly reducing "integration drift."

3. Implementation Details

Zero-Phase Distortion: All filters are implemented using scipy.signal.filtfilt. This performs a forward-backward pass, ensuring that the filtered signal has zero phase-shift, keeping the sensor data perfectly synchronized with the original timestamps.Sampling Consistency: All processing is performed at a fixed sampling rate of $500 \, Hz$ (following the resampling stage) to maintain a stable Nyquist frequency for the digital filters.

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
