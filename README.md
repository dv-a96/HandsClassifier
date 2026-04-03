# HandsClassifier
This project aims to detect handedness (which hand is holding the device) during smartphone video recording. The classification is performed using Machine Learning models trained on accelerometer and gyroscope data.


## Data Acquizition
The dataset was collected using the OpenCamera Sensors application [^1] [^2], which enables synchronized recording of IMU sensor data alongside video footage.
To ensure consistency across the dataset, a standardized movement protocol was followed for both left and right-hand recordings:

**Initial Position:** The device is held with a straight arm directly in front of the face.

**Movement:** The device is brought toward the nose and then extended back to the initial position.
The participents were guided to wait a moment after the recording started, when they reached next to the nose and in the end of the movment.

The data collected from __ different participents, men and wemen, with avarge age of __.

For every recorded session, the following files were generated:

`vidname.mp4:` The raw video file.

`vidname_accel.csv:` Accelerometer data (X, Y, Z axes) with corresponding timestamps.

`vidname_gyro.csv:` Gyroscope data (X, Y, Z axes) with corresponding timestamps.

Each CSV file consists of four columns: three representing the spatial axes ($X, Y, Z$) and one for the timestamps.

## Pre processing
### Sampling consistency 

To verify that there are no inconsistencies in the sampling rate of a file (caused by system load or other technical issues) we calculate a `diff` column for every file, represents the differance between the sampling time of each point. We plot the histogram of this column to see the disterbution of the values of the time differance between sampling - the sampling rate. If the sampling rate is consistant we expect to see one value (or at least narrow disterbution with no large differance betwwen values). But when we ploted the histograms of the files we saw that almost in every files there are some points with sampling rate around 4-5ms where the common range of the sampling is around 2 ms. This indicate that in most of the files there's at least one point that the device "missed a point" during the sampling.

![Sampling rate's histogram]()

To address these inconsistencies and ensure a uniform temporal grid for further analysis, we performed a resampling of the data to a fixed sampling rate of 500 Hz (a constant value of 2 ms in the `diff` column). In cases where the device "missed" a point we employed linear interpolation to estimate the missing values based on there seroundings points.


### Remove gravity acceleration and static bias

In accelerometer data, the measured acceleration includes both the device's motion and the constant gravitational acceleration. Since the videos were recorded in selfie mode (device held vertically), the gravity component primarily affects the Y-axis, appearing as a constant offset around 9.8 m/s² (depending on orientation).

To isolate the actual motion-induced acceleration, we applied a low-frequency Butterworth high-pass filter to remove the gravity component while preserving the dynamic motion signals. For the gyroscope data, which measures rotational velocity, no such correction was needed as gravity does not affect angular measurements.

In gyroscope data, sensors frequently exhibit a "Static Bias"—a non-zero reading that persists even when the device is completely stationary. Over time, this bias is subject to "drift" driven by thermal fluctuations and electrical noise. If left uncompensated, these residual offsets accumulate during the integration process, might resulting in significant errors in calculated angular displacement.

To mitigate this, we implemented a second-order Butterworth high-pass filter designed to dynamically isolate and remove the bias. By utilizing a very low cutoff frequency of 0.1 Hz, the filter effectively identifies the slow-moving sensor drift as a DC component and subtracts it from the signal in real-time. This approach preserves the integrity of the actual rotational velocity while ensuring a zero-mean signal, thereby significantly reducing integration drift and enhancing the overall accuracy of the orientation data.

### Smoothing

Despite removing the gravitational components and static bias, the signals still exhibited high-frequency artifacts characterized by abrupt, sharp peaks. These rapid fluctuations—occurring on a millisecond scale—are physically inconsistent with intentional human motion and likely stem from electronic noise or sensor jitter. To eliminate these artifacts, we evaluated two smoothing techniques: Moving Average and the Savitzky-Golay filter.

While the Moving Average filter effectively reduced noise, it tended to "smear" the signal, causing a loss of important peak information and shifting the temporal alignment of the motion. In contrast, the Savitzky-Golay filter was selected because it uses local polynomial regression to smooth the data while better preserving the original shape and height of the signal's peaks. This allowed us to suppress the noise without compromising the dynamic characteristics of the hand gestures.

## Features Extracion
Two types of features were extracted:
1. statistical features
2. correlation features
   
### statistical features
### correlation features
## Features Selection

## Random Forest Classifier

___
[^1] A. Akhmetyanov, A. Kornilova, M. Faizullin, D. Pozo and G. Ferrer, "Sub-millisecond Video Synchronization of Multiple Android Smartphones," 2021 IEEE Sensors, 2021, pp. 1-4

[^2] Faizullin, M.; Kornilova, A.; Akhmetyanov, A.; Ferrer, G. Twist-n-Sync: Software Clock Synchronization with Microseconds Accuracy Using MEMS-Gyroscopes. Sensors 2021, 21, 68
