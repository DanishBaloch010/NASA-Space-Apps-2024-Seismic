import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

cat_directory = 'C:/Users/bruno/OneDrive/Desktop/NASA Space Apps 2024 Seismic/space_apps_2024_seismic_detection/space_apps_2024_seismic_detection/data/lunar/training/catalogs/'
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
cat = pd.read_csv(cat_file)
print(cat)


row = cat.iloc[4]
relative_seconds = float(row['time_rel(sec)'])
print(relative_seconds)

test_filename = row.filename
print(test_filename)


data_directory = 'C:/Users/bruno/OneDrive/Desktop/NASA Space Apps 2024 Seismic/space_apps_2024_seismic_detection/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
csv_file = f'{data_directory}/{test_filename}.csv'
data_cat = pd.read_csv(csv_file)
print(data_cat)


# Read in time steps and velocities
csv_times = np.array(data_cat['time_rel(sec)'].tolist())
csv_data = np.array(data_cat['velocity(m/s)'].tolist())
# Plot the trace!
fig,ax = plt.subplots(1,1,figsize=(10,3))
ax.plot(csv_times,csv_data)
# Make the plot pretty
ax.set_xlim([min(csv_times),max(csv_times)])
ax.set_ylabel('Velocity (m/s)')
ax.set_xlabel('Time (s)')
ax.set_title(f'{test_filename}', fontweight='bold')
# Plot where the arrival time is
arrival_line = ax.axvline(x=relative_seconds, c='red', label='Rel. Arrival')
ax.legend(handles=[arrival_line])
plt.show()
