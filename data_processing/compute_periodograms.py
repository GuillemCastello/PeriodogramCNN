from create_times_array import create_tdeltas_array
from astropy.timeseries import LombScargle
import numpy as np
import itertools as itools
import glob
import h5py
import sys
import multiprocessing
from tqdm import tqdm
from scipy.signal import windows

# Define the frequency range
print('Frequency range defined')
frequency = np.linspace(1/(1440*60), 1/120, 2000, dtype=np.float32)

# Define the path to the original data
year = sys.argv[1]
month = sys.argv[2]
day = sys.argv[3]
njobs = sys.argv[4]

# Define the path to the original data
original_data_path = f'./{day}/updated/'
print(f'Data is in directory {original_data_path} \n')

# Read the data from the HDF5 file
print('Loading data from HDF5 file')
f = h5py.File(original_data_path+f'{day}_data_modified.h5', mode='r')
# Load the time series data
print('Keys in HDF5 file:', list(f.keys()))
time_series = np.array(f['time_series'], dtype=np.int32)
tdeltas = np.array(f['tdeltas'], dtype=np.int32)
f.close()


print('Time series shape:', time_series.shape)
print('Normalizing time series and applying Hann window to time series')
hann_window = windows.hann(time_series.shape[0])
hann_window = hann_window[:, None, None]

# Apply the Hann window to the time series
# Define the x and y ranges


#temporal_sequence = time_series #- np.mean(time_series, axis=0)) / np.mean(time_series, axis=0)
temporal_sequence = time_series * hann_window

time_series = None  # Free memory
# Define the periodograms array
print('Defining periodograms array')
periodograms = np.zeros(shape=(len(frequency), temporal_sequence.shape[1], temporal_sequence.shape[2]), dtype=np.float32)

xrange = range(2048)
yrange = range(2048)
indices_list = list(itools.product(xrange, yrange))
inside_indices_list = list(filter(lambda x: np.sqrt((x[0]-1024)**2 + (x[1]-1024)**2) <= 830, indices_list))

# Define a function to compute the periodogram for a given index
def compute_periodogram(index, times=tdeltas, data=temporal_sequence, frequency=frequency):
    x, y = inside_indices_list[index]
    #print(f'Computing periodogram for {x}_{y}')
    # Compute the periodogram
    psd = LombScargle(times, data[:, x, y], normalization='standard').power(frequency)
    psd = np.array(psd, dtype=np.float32)
    return psd

print('Creating a pool of worker processes, to parallelize periodogram calculations')
# Create a pool of worker processes
with multiprocessing.Pool(int(njobs)) as pool:
    # Compute the periodograms in parallel
    results = list(tqdm(pool.imap(compute_periodogram, range(len(inside_indices_list))), total=len(inside_indices_list)))

# Convert the results to a numpy array
results = np.array(results)

# Fill the periodograms array
for i, (x, y) in enumerate(inside_indices_list):
    periodograms[:, x, y] = results[i]

# Save the periodograms to a file
print(f'Saving periodograms to HDF5 file, with name:{day}_periodograms.h5')
f1 = h5py.File(original_data_path+f'{day}_periodograms.h5', mode='w')
f1.create_dataset('periodograms', data=periodograms)
f1.close()
