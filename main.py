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

cat_directory = 'C:/Users/bruno/OneDrive/Desktop/NASA Space Apps 2024 Seismic/space_apps_2024_seismic_detection/space_apps_2024_seismic_detection/data/lunar/training/catalogs/'
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
data_cat = pd.read_csv(cat_file)
print(data_cat)


row = data_cat.iloc[7]
relative_seconds = float(row['time_rel(sec)'])
print(relative_seconds)

test_filename = row.filename
print(test_filename)


data_directory = 'C:/Users/bruno/OneDrive/Desktop/NASA Space Apps 2024 Seismic/space_apps_2024_seismic_detection/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
csv_file = f'{data_directory}/{test_filename}.csv'
mseed_file = f'{data_directory}/{test_filename}.mseed'
st = read(mseed_file)

data_cat = pd.read_csv(csv_file)
print(data_cat)



# Min-Max Normalization to [-1, 1]
data_min = data_cat['velocity(m/s)'].min()
data_max = data_cat['velocity(m/s)'].max()

data_cat['velocity_normalized'] = 2 * (data_cat['velocity(m/s)'] - data_min) / (data_max - data_min) - 1


# Assuming data_cat is your DataFrame with normalized velocity data
# Define your parameters for the bandpass filter
lowcut = 0.1  # Low cutoff frequency in Hz
highcut = 10.0  # High cutoff frequency in Hz
sampling_rate = 100.0  # Sampling frequency in Hz (adjust based on your data)


# Define the arrival time (replace this with your actual arrival time)
arrival_time = relative_seconds  # Ensure this variable is defined in your context



# Set the minimum frequency
minfreq = 0.5
maxfreq = 1.0
# Going to create a separate trace for the filter data
st_filt = st.copy()
st_filt.filter('bandpass',freqmin=minfreq,freqmax=maxfreq)
tr_filt = st_filt.traces[0].copy()
tr_times_filt = tr_filt.times()
tr_data_filt = tr_filt.data
 # To better see the patterns, we will create a spectrogram using the scipy␣function
# It requires the sampling rate, which we can get from the miniseed header as␣shown a few cells above
from scipy import signal
from matplotlib import cm
f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)



# Plot the time series and spectrogram
fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(2, 1, 1)
# Plot trace
ax.plot(tr_times_filt,tr_data_filt)
# Mark detection
ax.axvline(x = arrival_time, color='red',label='Detection')
ax.legend(loc='upper left')
# Make the plot pretty
ax.set_xlim([min(tr_times_filt),max(tr_times_filt)])
ax.set_ylabel('Velocity (m/s)')
ax.set_xlabel('Time (s)')
ax2 = plt.subplot(2, 1, 2)
vals = ax2.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
ax2.set_xlim([min(tr_times_filt),max(tr_times_filt)])
ax2.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
ax2.axvline(x=arrival_time, c='red')
cbar = plt.colorbar(vals, orientation='horizontal')
cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')
ax.legend()
plt.show()
