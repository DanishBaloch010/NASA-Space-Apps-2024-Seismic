import numpy as np
import pandas as pd
from obspy import read, Stream, Trace
from obspy.signal.invsim import cosine_taper
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt, welch, spectrogram
from scipy import signal
from matplotlib import cm

# File paths
file_name = "xa.s15.00.mhz.1973-08-10HR00_evid00126"
originalTestData = f"C:/Users/AyreB12/OneDrive - Berkhamsted Schools Group/Desktop/space_apps_2024_seismic_detection/data/lunar/test/data/S15_GradeA/{file_name}"
csv_file = f'{originalTestData}.csv'
mseed_file = f'{originalTestData}.mseed'

# Read the miniSEED file
st = read(mseed_file)
data_cat = pd.read_csv(csv_file)
print(data_cat)

# Define min and max frequency for filtering
minfreq = 0.5
maxfreq = 2.0

# Apply bandpass filter
st_filt = st.copy()
st_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
tr_filt = st_filt.traces[0].copy()
tr_times_filt = tr_filt.times()
tr_data_filt = tr_filt.data

# Calculate the spectrogram using scipy's signal.spectrogram function
f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)

# Find the indices for the desired frequency range
freq_idx = np.where((f >= minfreq) & (f <= maxfreq))[0]

# Average power over the frequency range for each time point
avg_power = np.mean(sxx[freq_idx, :], axis=0)

# Weighted average frequency (power-weighted) for each time point
weighted_freq = np.sum(f[freq_idx, np.newaxis] * sxx[freq_idx, :], axis=0) / np.sum(sxx[freq_idx, :], axis=0)

# Initialize the dominant frequency array and set a threshold for detecting power spikes
dominant_freq = np.zeros(sxx.shape[1])
threshold = 0.2 * np.max(avg_power)

# Detect dominant frequency for each time point above the threshold
for i in range(sxx.shape[1]):
    if avg_power[i] >= threshold:
        max_power_idx = np.argmax(sxx[:, i])
        dominant_freq[i] = f[max_power_idx]
    else:
        dominant_freq[i] = 0

# Adjusting velocity to align with spectrogram time points
n = len(tr_data_filt) // len(t)
velocities_for_avg_power = np.mean(tr_data_filt[:n*len(t)].reshape(-1, n), axis=1)

# Prompt the user to enter the predicted event time
predicted_time = float(input("Enter the model's predicted event time (in seconds): "))

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(12, 18))

# Original trace plot
axs[0].plot(tr_times_filt, tr_data_filt, label='Filtered Seismic Data')
axs[0].axvline(predicted_time, color='r', linestyle='--', label='Predicted Event Time')
axs[0].set_ylabel('Velocity (m/s)')
axs[0].set_xlabel('Time (s)')
axs[0].legend()
axs[0].set_title('Filtered Seismic Trace')

# Spectrogram
vals = axs[1].pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
axs[1].axvline(predicted_time, color='r', linestyle='--', label='Predicted Event Time')
axs[1].set_xlim([min(t), max(t)])
axs[1].set_xlabel('Relative Time (s)')
axs[1].set_ylabel('Frequency (Hz)')
cbar = plt.colorbar(vals, ax=axs[1], orientation='horizontal')
cbar.set_label('Power ((m/s)^2/sqrt(Hz))')
axs[1].set_title('Spectrogram')

# Average power and weighted frequency over time
axs[2].plot(t, avg_power, label='Average Power', color='b')
axs[2].axvline(predicted_time, color='r', linestyle='--', label='Predicted Event Time')
axs[2].set_ylabel('Average Power ((m/s)^2/Hz)', color='b')
ax2 = axs[2].twinx()
ax2.plot(t, weighted_freq, label='Weighted Frequency', color='r')
ax2.axvline(predicted_time, color='r', linestyle='--')
ax2.set_ylabel('Weighted Frequency (Hz)', color='r')
axs[2].set_xlabel('Relative Time (s)')
axs[2].set_title('Average Power and Weighted Frequency Over Time')

# Dominant frequency over time
axs[3].plot(t, dominant_freq, label='Dominant Frequency', color='g')
axs[3].axvline(predicted_time, color='r', linestyle='--', label='Predicted Event Time')
axs[3].set_ylabel('Dominant Frequency (Hz)', color='g')
axs[3].set_xlabel('Relative Time (s)')
axs[3].set_title('Dominant Frequency Over Time')
axs[3].legend()

fig.tight_layout()
plt.show()

# Bollinger Bands Calculation and Visualization
velocity = velocities_for_avg_power
window = 240
num_std_dev = 3.5

moving_average = pd.Series(velocity).rolling(window=window, min_periods=1).mean()
rolling_std = pd.Series(velocity).rolling(window=window, min_periods=1).std()

upper_band = moving_average + (num_std_dev * rolling_std)
lower_band = moving_average - (num_std_dev * rolling_std)

if len(velocity) > 0:
    initial_mean = velocity[0]
    upper_band.iloc[0] = initial_mean + (num_std_dev * rolling_std.iloc[0]) if not np.isnan(rolling_std.iloc[0]) else initial_mean
    lower_band.iloc[0] = initial_mean - (num_std_dev * rolling_std.iloc[0]) if not np.isnan(rolling_std.iloc[0]) else -initial_mean

# Bollinger Bands Plot
plt.figure(figsize=(10, 6))
plt.plot(tr_times_filt, velocity, label='Velocity', color='blue')
plt.axvline(predicted_time, color='r', linestyle='--', label='Predicted Event Time')
plt.plot(tr_times_filt, moving_average, label='Moving Average', color='green')
plt.plot(tr_times_filt, upper_band, label='Upper Band', color='red')
plt.plot(tr_times_filt, lower_band, label='Lower Band', color='red')
plt.fill_between(tr_times_filt, lower_band, upper_band, color='grey', alpha=0.2)
plt.title('Bollinger Bands with Initial Average Adjustment')
plt.xlabel('Relative Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend(loc='best')
plt.show()
