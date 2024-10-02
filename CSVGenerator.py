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


for index in range(76):

    cat_directory = 'C:/Users/bruno/OneDrive/Desktop/NASA Space Apps 2024 Seismic/space_apps_2024_seismic_detection/space_apps_2024_seismic_detection/data/lunar/training/catalogs/'
    cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
    data_cat = pd.read_csv(cat_file)
    # print(data_cat)


    row = data_cat.iloc[index]
    relative_seconds = float(row['time_rel(sec)'])
    # print(relative_seconds)

    test_filename = row.filename
    # print(test_filename)


    data_directory = 'C:/Users/bruno/OneDrive/Desktop/NASA Space Apps 2024 Seismic/space_apps_2024_seismic_detection/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
    csv_file = f'{data_directory}/{test_filename}.csv'
    mseed_file = f'{data_directory}/{test_filename}.mseed'
    st = read(mseed_file)

    data_cat = pd.read_csv(csv_file)
    # print(data_cat)

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

    # print(f)
    # print(len(f))
    # print("\n\n-----------------------------------------------------------------------------\n\n")
    # print(t)
    # print(len(t))
    # print("\n\n-----------------------------------------------------------------------------\n\n")
    # print(sxx)
    # print(np.shape(sxx))

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
    threshold = 0.2 * np.max(avg_power)  # or any other value you see fit

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



    # # Lengths of the different components
    # print(f'Length of Relative Time (t): {len(t)}')
    # print(f'Length of Velocity (tr_data_filt): {len(tr_data_filt)}')
    # print(f'Length of Average Power (avg_power): {len(avg_power)}')
    # print(f'Length of Weighted Frequency (weighted_freq): {len(weighted_freq)}')
    # print(f'Length of Dominant Frequency (dominant_freq): {len(dominant_freq)}')

    n = len(tr_data_filt) // len(t)
    velocities_for_avg_power = np.mean(tr_data_filt[:n*len(t)].reshape(-1, n), axis=1)


    # # Now, check lengths again to ensure consistency
    # print(f'Length of Adjusted Velocities: {len(velocities_for_avg_power)}')

    # # Check lengths again to ensure consistency
    # print(f'Length of Adjusted Relative Time (t_avg_power): {len(t)}')
    # print(f'Length of Interpolated Velocities: {len(velocities_for_avg_power)}')


    arrival_time_rel = row['time_rel(sec)']


    # # Plotting
    # fig, axs = plt.subplots(4, 1, figsize=(12, 18))

    # # Original trace plot
    # axs[0].plot(tr_times_filt, tr_data_filt, label='Filtered Seismic Data')
    # axs[0].set_ylabel('Velocity (m/s)')
    # axs[0].set_xlabel('Time (s)')
    # axs[0].legend()
    # axs[0].set_title('Filtered Seismic Trace')

    # # Plot where the arrival time is
    # arrival_line = axs[0].axvline(x=arrival_time_rel, c='red', label='Rel. Arrival')
    # axs[0].legend(handles=[arrival_line])


    # # Spectrogram
    # vals = axs[1].pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
    # axs[1].set_xlim([min(t), max(t)])
    # axs[1].set_xlabel('Relative Time (s)')
    # axs[1].set_ylabel('Frequency (Hz)')
    # cbar = plt.colorbar(vals, ax=axs[1], orientation='horizontal')
    # cbar.set_label('Power ((m/s)^2/sqrt(Hz))')
    # axs[1].set_title('Spectrogram')

    # arrival_line = axs[1].axvline(x=arrival_time_rel, c='red', label='Rel. Arrival')
    # axs[1].legend(handles=[arrival_line])

    # # Average power and weighted frequency over time
    # axs[2].plot(t, avg_power, label='Average Power', color='b')
    # axs[2].set_ylabel('Average Power ((m/s)^2/Hz)', color='b')
    # ax2 = axs[2].twinx()
    # ax2.plot(t, weighted_freq, label='Weighted Frequency', color='r')
    # ax2.set_ylabel('Weighted Frequency (Hz)', color='r')
    # axs[2].set_xlabel('Relative Time (s)')
    # axs[2].set_title('Average Power and Weighted Frequency Over Time')

    # arrival_line = axs[2].axvline(x=arrival_time_rel, c='red', label='Rel. Arrival')
    # axs[2].legend(handles=[arrival_line])

    # # Dominant frequency over time
    # axs[3].plot(t, dominant_freq, label='Dominant Frequency', color='g')
    # axs[3].set_ylabel('Dominant Frequency (Hz)', color='g')
    # axs[3].set_xlabel('Relative Time (s)')
    # axs[3].set_title('Dominant Frequency Over Time')
    # axs[3].legend()
    # arrival_line = axs[3].axvline(x=arrival_time_rel, c='red', label='Rel. Arrival')
    # axs[3].legend(handles=[arrival_line])

    # fig.tight_layout()
    # plt.show()


    velocity = velocities_for_avg_power

    # Parameters for the two-band approach
    window = 240  # Look-back period for moving average (can be adjusted)
    num_std_dev = 3.5  # Number of standard deviations for band calculation

    # 1. Compute the moving average over the dataset
    moving_average = pd.Series(velocity).rolling(window=window, min_periods=1).mean()

    # 2. Compute the standard deviation over the dataset
    rolling_std = pd.Series(velocity).rolling(window=window, min_periods=1).std()

    # 3. Calculate upper and lower bands based on the moving average and standard deviation
    upper_band = moving_average + (num_std_dev * rolling_std)
    lower_band = moving_average - (num_std_dev * rolling_std)

    # 4. Assign values to upper and lower bands for the first row explicitly
    if len(velocity) > 0:
        initial_mean = velocity[0]  # Use the first velocity value
        upper_band.iloc[0] = initial_mean + (num_std_dev * rolling_std.iloc[0]) if not np.isnan(rolling_std.iloc[0]) else initial_mean
        lower_band.iloc[0] = initial_mean - (num_std_dev * rolling_std.iloc[0]) if not np.isnan(rolling_std.iloc[0]) else -initial_mean


    # Normalization to the range of -1 to 1
    def normalize_minus_one_to_one(series):
        min_val = series.min()
        max_val = series.max()
        return 2 * ((series - min_val) / (max_val - min_val)) - 1

    # Normalization to the range of 0 to 1
    def normalize_zero_to_one(series):
        min_val = series.min()
        max_val = series.max()
        return (series - min_val) / (max_val - min_val)

    # Normalization to the range of -1 to 0
    def normalize_minus_one_to_zero(series):
        min_val = series.min()
        max_val = series.max()
        return (series - max_val) / (min_val - max_val)

    # Now, create the DataFrame to store the results
    results_df = pd.DataFrame({
        'Relative Time (s)': t,  # Use the adjusted time array
        'Velocity (m/s)': normalize_minus_one_to_one(velocities_for_avg_power),
        'Average Power ((m/s)^2/Hz)': normalize_zero_to_one(avg_power),
        'Weighted Frequency (Hz)': normalize_zero_to_one(weighted_freq),
        'Dominant Frequency (Hz)': normalize_zero_to_one(dominant_freq[:len(avg_power)]),  # Adjust to match avg_power length
        'Velocity Upper Band': normalize_zero_to_one(upper_band),
        'Velocity Lower Band': normalize_minus_one_to_zero(lower_band),
        'Moving Average': normalize_minus_one_to_one(moving_average)
    })


    # # Visualize the results
    # plt.figure(figsize=(10, 6))
    # plt.plot(results_df['Relative Time (s)'], velocity, label='Velocity', color='blue')
    # plt.plot(results_df['Relative Time (s)'], moving_average, label='Moving Average', color='green')
    # plt.plot(results_df['Relative Time (s)'], upper_band, label='Upper Band', color='red')
    # plt.plot(results_df['Relative Time (s)'], lower_band, label='Lower Band', color='red')
    # plt.fill_between(results_df['Relative Time (s)'], lower_band, upper_band, color='grey', alpha=0.2)
    # plt.title('Bollinger Bands with Initial Average Adjustment')
    # plt.xlabel('Relative Time (s)')
    # plt.ylabel('Velocity (m/s)')
    # plt.legend(loc='best')
    # plt.show()

    # print(test_filename)
    event_number = test_filename[-5:] 
    # print(event_number)

    # Print the results DataFrame
    # print(results_df)

    # Save the DataFrame to CSV
    output_csv_file = f'DataCSV/{event_number}_seismic_results.csv'
    results_df.to_csv(output_csv_file, index=False)

    # Confirm the file has been saved
    # print(f'The results have been saved to {output_csv_file}')