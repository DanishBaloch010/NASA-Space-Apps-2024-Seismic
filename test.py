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

cat_directory = 'C:/Users/AyreB12/OneDrive - Berkhamsted Schools Group/Desktop/space_apps_2024_seismic_detection/data/lunar/training/catalogs/'
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
data_cat = pd.read_csv(cat_file)
print(data_cat)


row = data_cat.iloc[22]
relative_seconds = float(row['time_rel(sec)'])
print(relative_seconds)

test_filename = row.filename
print(test_filename)


data_directory = 'C:/Users/AyreB12/OneDrive - Berkhamsted Schools Group/Desktop/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
csv_file = f'{data_directory}/{test_filename}.csv'
mseed_file = f'{data_directory}/{test_filename}.mseed'
st = read(mseed_file)

data_cat = pd.read_csv(csv_file)
print(data_cat)

minfreq = 0.5
maxfreq = 2.0
# Going to create a separate trace for the filter data
st_filt = st.copy()
st_filt.filter('bandpass',freqmin=minfreq,freqmax=maxfreq)
tr_filt = st_filt.traces[0].copy()
tr_times_filt = tr_filt.times()
tr_data_filt = tr_filt.data
 # To better see the patterns, we will create a spectrogram using the scipy␣function
# It requires the sampling rate, which we can get from the miniseed header as␣shown a few cells above

f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)

print(f)
print(len(f))
print("\n\n-----------------------------------------------------------------------------\n\n")
print(t)
print(len(t))
print("\n\n-----------------------------------------------------------------------------\n\n")
print(sxx)
print(np.shape(sxx))

# # Plot the time series and spectrogram
# fig = plt.figure(figsize=(10, 10))
# ax = plt.subplot(2, 1, 1)
# # Plot trace
# ax.plot(tr_times_filt,tr_data_filt)

# # Make the plot pretty
# ax.set_xlim([min(tr_times_filt),max(tr_times_filt)])
# ax.set_ylabel('Velocity (m/s)')
# ax.set_xlabel('Time (s)')
# ax2 = plt.subplot(2, 1, 2)
# vals = ax2.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
# ax2.set_xlim([min(tr_times_filt),max(tr_times_filt)])
# ax2.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
# ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
# cbar = plt.colorbar(vals, orientation='horizontal')
# cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')
# ax.legend()
# plt.show()

# Find the indices for the desired frequency range
freq_idx = np.where((f >= minfreq) & (f <= maxfreq))[0]

# Average power over the frequency range for each time point
avg_power = np.mean(sxx[freq_idx, :], axis=0)

# Weighted average frequency (power-weighted) for each time point
weighted_freq = np.sum(f[freq_idx, np.newaxis] * sxx[freq_idx, :], axis=0) / np.sum(sxx[freq_idx, :], axis=0)


# Set the threshold for detecting power spikes
threshold = 0.05 * np.max(avg_power)  # or any other value you see fit

# Initialize the dominant frequency array
dominant_freq = np.zeros(sxx.shape[1])  

# Loop through each time point
for i in range(sxx.shape[1]):
    if avg_power[i] >= threshold:  # Check if average power is above the threshold
        # Find the index of the maximum power for each time point
        max_power_idx = np.argmax(sxx[:, i])
        # Get the corresponding frequency
        dominant_freq[i] = f[max_power_idx]
    else:
        dominant_freq[i] = 0  # Set to zero if no spike



# Lengths of the different components
print(f'Length of Relative Time (t): {len(t)}')
print(f'Length of Velocity (tr_data_filt): {len(tr_data_filt)}')
print(f'Length of Average Power (avg_power): {len(avg_power)}')
print(f'Length of Weighted Frequency (weighted_freq): {len(weighted_freq)}')
print(f'Length of Dominant Frequency (dominant_freq): {len(dominant_freq)}')

n = len(tr_data_filt) // len(t)
velocities_for_avg_power = np.mean(tr_data_filt[:n*len(t)].reshape(-1, n), axis=1)


# Now, check lengths again to ensure consistency
print(f'Length of Adjusted Velocities: {len(velocities_for_avg_power)}')

# Check lengths again to ensure consistency
print(f'Length of Adjusted Relative Time (t_avg_power): {len(t)}')
print(f'Length of Interpolated Velocities: {len(velocities_for_avg_power)}')

# Now, create the DataFrame to store the results
results_df = pd.DataFrame({
    'Relative Time (s)': t,  # Use the adjusted time array
    'Velocity (m/s)': velocities_for_avg_power,
    'Average Power ((m/s)^2/Hz)': avg_power,
    'Weighted Frequency (Hz)': weighted_freq,
    'Dominant Frequency (Hz)': dominant_freq[:len(avg_power)],  # Adjust to match avg_power length
})

# Print the results DataFrame
print(results_df)


# Plotting
fig, axs = plt.subplots(4, 1, figsize=(12, 18))

# Original trace plot
axs[0].plot(tr_times_filt, tr_data_filt, label='Filtered Seismic Data')
axs[0].set_ylabel('Velocity (m/s)')
axs[0].set_xlabel('Time (s)')
axs[0].legend()
axs[0].set_title('Filtered Seismic Trace')


# Spectrogram
vals = axs[1].pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
axs[1].set_xlim([min(t), max(t)])
axs[1].set_xlabel('Relative Time (s)')
axs[1].set_ylabel('Frequency (Hz)')
cbar = plt.colorbar(vals, ax=axs[1], orientation='horizontal')
cbar.set_label('Power ((m/s)^2/sqrt(Hz))')
axs[1].set_title('Spectrogram')

# Average power and weighted frequency over time
axs[2].plot(t, avg_power, label='Average Power', color='b')
axs[2].set_ylabel('Average Power ((m/s)^2/Hz)', color='b')
ax2 = axs[2].twinx()
ax2.plot(t, weighted_freq, label='Weighted Frequency', color='r')
ax2.set_ylabel('Weighted Frequency (Hz)', color='r')
axs[2].set_xlabel('Relative Time (s)')
axs[2].set_title('Average Power and Weighted Frequency Over Time')

# Dominant frequency over time
axs[3].plot(t, dominant_freq, label='Dominant Frequency', color='g')
axs[3].set_ylabel('Dominant Frequency (Hz)', color='g')
axs[3].set_xlabel('Relative Time (s)')
axs[3].set_title('Dominant Frequency Over Time')
axs[3].legend()

fig.tight_layout()
plt.show()


# Specify the file path where you want to save the CSV file
output_csv_file = 'seismic_results.csv'

# Save the DataFrame to CSV
results_df.to_csv(output_csv_file, index=False)

# Confirm the file has been saved
print(f'The results have been saved to {output_csv_file}')