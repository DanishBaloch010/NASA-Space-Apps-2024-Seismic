import numpy as np
import pandas as pd
import os
from scipy import signal
from matplotlib import cm

# Directories
data_directory = 'C:/Users/bruno/OneDrive/Desktop/NASA Space Apps 2024 Seismic/space_apps_2024_seismic_detection/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
output_directory = 'TestingDataCSVs/'

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process each CSV file in the data directory
for filename in os.listdir(data_directory):
    if filename.endswith('.csv'):
        # Load the CSV data
        csv_file = os.path.join(data_directory, filename)
        data_cat = pd.read_csv(csv_file)

        # Assuming 'time' is the time column and the other necessary columns exist
        tr_data_filt = data_cat['velocity(m/s)'].values  # Modify this as needed
        arrival_time_rel = data_cat['time_rel(sec)'].iloc[0]  # Modify based on actual structure

        # Spectrogram parameters
        minfreq = 0.5
        maxfreq = 2.0
        
        # Spectrogram calculation
        f, t, sxx = signal.spectrogram(tr_data_filt, fs=100)  # Assuming fs (sampling frequency) is known; adjust as needed

        # Frequency analysis
        freq_idx = np.where((f >= minfreq) & (f <= maxfreq))[0]
        avg_power = np.mean(sxx[freq_idx, :], axis=0)
        weighted_freq = np.sum(f[freq_idx, np.newaxis] * sxx[freq_idx, :], axis=0) / np.sum(sxx[freq_idx, :], axis=0)

        # Threshold for detecting power spikes
        threshold = 0.2 * np.max(avg_power)
        dominant_freq = np.zeros(sxx.shape[1])

        # Identify dominant frequency
        for i in range(sxx.shape[1]):
            if avg_power[i] >= threshold:
                max_power_idx = np.argmax(sxx[:, i])
                dominant_freq[i] = f[max_power_idx]
            else:
                dominant_freq[i] = 0

        # Calculate velocities for average power
        n = len(tr_data_filt) // len(t)
        velocities_for_avg_power = np.mean(tr_data_filt[:n*len(t)].reshape(-1, n), axis=1)

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

        # Normalization functions
        def normalize_minus_one_to_one(series):
            min_val = series.min()
            max_val = series.max()
            return 2 * ((series - min_val) / (max_val - min_val)) - 1

        def normalize_zero_to_one(series):
            min_val = series.min()
            max_val = series.max()
            return (series - min_val) / (max_val - min_val)

        def normalize_minus_one_to_zero(series):
            min_val = series.min()
            max_val = series.max()
            return (series - max_val) / (min_val - max_val)

        # Create results DataFrame
        results_df = pd.DataFrame({
            'Relative Time (s)': t,
            'Velocity (m/s)': normalize_minus_one_to_one(velocities_for_avg_power),
            'Average Power ((m/s)^2/Hz)': normalize_zero_to_one(avg_power),
            'Weighted Frequency (Hz)': normalize_zero_to_one(weighted_freq),
            'Dominant Frequency (Hz)': normalize_zero_to_one(dominant_freq[:len(avg_power)]),
            'Velocity Upper Band': normalize_zero_to_one(upper_band),
            'Velocity Lower Band': normalize_minus_one_to_zero(lower_band),
            'Moving Average': normalize_minus_one_to_one(moving_average)
        })

        # Save the DataFrame to CSV
        output_csv_file = os.path.join(output_directory, f'{filename[:-4]}_seismic_results.csv')
        results_df.to_csv(output_csv_file, index=False)

        # Confirm the file has been saved
        print(f'The results have been saved to {output_csv_file}')