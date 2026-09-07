# HandsClassifier

This project investigates the detection of handedness — i.e., which hand is holding the device — during smartphone video recording. Classification is performed using machine learning models trained on synchronized accelerometer and gyroscope data.

## Data Acquisition

The dataset was collected using the OpenCamera Sensors application [^1] [^2], which enables synchronized recording of IMU sensor data alongside video footage.

To ensure consistency across the dataset, a standardized movement protocol was followed for both left-hand and right-hand recordings:

- **Initial position:** The device is held with a straight arm directly in front of the face.
- **Movement:** The device is brought toward the nose and then returned to the initial position.

Participants were instructed to pause briefly after the recording began, upon reaching the nose, and again at the end of the movement.

The dataset comprises recordings from ___ participants (men and women), with an average age of ___.

For each recorded session, the following files were generated:

| File | Description |
|---|---|
| `vidname.mp4` | The raw video file. |
| `vidname_accel.csv` | Accelerometer data (X, Y, Z axes) with corresponding timestamps. |
| `vidname_gyro.csv` | Gyroscope data (X, Y, Z axes) with corresponding timestamps. |

Each CSV file consists of four columns: three representing the spatial axes ($X$, $Y$, $Z$) and one for the timestamps.

## Preprocessing

### Sampling Consistency

To verify the absence of sampling-rate inconsistencies (caused by system load or other technical issues), a `diff` column was computed for every file, representing the time difference between consecutive samples. The distribution of this column — effectively the sampling interval — was visualized via histograms. Under a perfectly consistent sampling rate, this distribution should concentrate around a single value, or at least a narrow range with no substantial outliers. However, histogram inspection revealed that in nearly every file, a subset of samples exhibited intervals of approximately 4–5 ms, compared to the nominal ~2 ms interval. This indicates that most files contain at least one point at which the device "missed" a sample during acquisition.

![Accelerometer sampling-interval distributions](figures/accel_sampling_consistency_histograms.png)
![Gyroscope sampling-interval distributions](figures/gyro_sampling_consistency_histograms.png)

To correct these inconsistencies and establish a uniform temporal grid for subsequent analysis, the data were resampled to a fixed rate of 500 Hz (i.e., a constant 2 ms interval in the `diff` column). Where the device "missed" a point, linear interpolation was used to estimate the missing values from their surrounding samples.

### Gravity and Static Bias Removal

In accelerometer data, the measured acceleration reflects both the device's motion and the constant gravitational acceleration. Since recordings were captured in selfie mode (device held vertically), the gravity component primarily affects the Y-axis, appearing as a constant offset of approximately 9.8 m/s² (depending on orientation).

To isolate motion-induced acceleration, a low-frequency Butterworth high-pass filter was applied to remove the gravitational component while preserving the dynamic motion signals. For gyroscope data, which measures rotational velocity, no equivalent correction was required, as gravity does not affect angular measurements.

Gyroscope data, however, are subject to static bias — a non-zero reading that persists even when the device is completely stationary. Over time, this bias is subject to drift driven by thermal fluctuations and electrical noise. Left uncompensated, these residual offsets accumulate during the integration process and may result in significant errors in the computed angular displacement.

To mitigate this, a second-order Butterworth high-pass filter was implemented to dynamically isolate and remove the bias. Using a very low cutoff frequency of 0.1 Hz, the filter identifies the slow-moving sensor drift as a DC component and subtracts it from the signal in real time. This approach preserves the integrity of the true rotational velocity while ensuring a zero-mean signal, thereby significantly reducing integration drift and improving the overall accuracy of the orientation data.

### Smoothing

Despite the removal of gravitational components and static bias, the signals continued to exhibit high-frequency artifacts characterized by abrupt, sharp peaks. These rapid, millisecond-scale fluctuations are physically inconsistent with intentional human motion and are more likely attributable to electronic noise or sensor jitter. Two smoothing techniques were evaluated for their removal: the moving-average filter and the Savitzky–Golay filter.

While the moving-average filter effectively reduced noise, it tended to "smear" the signal, causing a loss of peak information and shifting the temporal alignment of the motion. The Savitzky–Golay filter was therefore selected, as it uses local polynomial regression to smooth the data while better preserving the original shape and amplitude of the signal's peaks — suppressing noise without compromising the dynamic characteristics of the hand gestures.

## Feature Extraction

For each sample, 82 features were extracted, organized into the following three categories:

**1. Statistical Time-Domain Features**
- **Mean:** The average value of the signal.
- **Variance:** Measures the spread of the data points.
- **Min / Max:** The extreme values reached during the movement.
- **Median:** The middle value of the signal.
- **Delta Min-Max:** The total range of the signal ($Max - Min$).
- **Skewness:** Measures the asymmetry of the signal distribution around its mean.
- **Intensity:** Represents the overall magnitude of the movement, calculated as the sum of absolute values.
- **ZCR (Zero Crossing Rate):** The rate at which the signal changes sign. A high ZCR often indicates rapid oscillations.

**2. Count and Index Features**
- **Count Positive / Negative:** The number of samples with values above or below zero.
- **Argmax / Argmin:** The temporal indices (time steps) at which the signal reaches its maximum and minimum values.

**3. Correlation Features**

These features capture the relationships between different axes, sensors, and predefined gesture patterns in order to identify complex movement signatures.

- *Cross-Sensor Correlation*
  - **Gyro–Accel Correlation** (`gyro_accel_corr`): Measures the linear relationship between the gyroscope (Y-axis) and the accelerometer (X-axis).
  - **Gyro–Gyro Correlation** (`gyro_gyro_corr`): Measures the relationship between the Y- and X-axes of the gyroscope to capture rotational patterns.
- *Template Matching (Leave-One-Out)*

  To improve classification accuracy, the system compares each signal against dynamic templates for the "left" and "right" gestures using the Pearson correlation coefficient:
  - **Correlation with Right Template:** Measures how closely the current signal matches the average right-hand gesture profile.
  - **Correlation with Left Template:** Measures how closely the current signal matches the average left-hand gesture profile.
  - **Leave-One-Out Adjustment:** During training, templates are dynamically recomputed to exclude the current sample, ensuring the correlation score is not biased by the sample's own data.

## Feature Selection

To optimize the model and reduce dimensionality, redundant features — specifically those with a pairwise correlation coefficient higher than 0.9 — were removed. High correlation suggests that one feature can be largely predicted from another.

For each pair of highly correlated features, retention was determined by comparing each feature's correlation with the target label (left or right hand); the feature with the lower correlation to the label was removed, ensuring that the most informative predictors were retained.

Through this process, [number] redundant features were eliminated, resulting in a final set of [number] features.

## Random Forest Classifier

---
[^1] A. Akhmetyanov, A. Kornilova, M. Faizullin, D. Pozo and G. Ferrer, "Sub-millisecond Video Synchronization of Multiple Android Smartphones," 2021 IEEE Sensors, 2021, pp. 1-4

[^2] Faizullin, M.; Kornilova, A.; Akhmetyanov, A.; Ferrer, G. Twist-n-Sync: Software Clock Synchronization with Microseconds Accuracy Using MEMS-Gyroscopes. Sensors 2021, 21, 68
