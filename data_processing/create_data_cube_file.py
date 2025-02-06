import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
import numpy as np
import glob
import sys
from multiprocessing import Pool
from tqdm import tqdm
import time as t
import h5py
from derotate import derotate
from create_times_array import create_tdeltas_array


##############################################################
#CREATING THE DATA CUBE / DEROTATING IMAGES OF THE TIME SERIES
##############################################################
print('Derotating images')
print('Using updated files')
year = sys.argv[1]
month = sys.argv[2]
day = sys.argv[3]
n_threads = sys.argv[4]

directory_of_data = f'/home/guillem/{day}/updated/'
files_updated = sorted(glob.glob(directory_of_data+'*.fits'))

change_telescope_indices = []
for num, updated_file in enumerate(files_updated[1:]):
    if updated_file[-15] != files_updated[num][-15]:
        change_telescope_indices.append(num+1)

drot_time_start = t.perf_counter()
# Function to fill a single slice from the data cube, (image derotated)


print(f'Parallelizing the derotation process with {n_threads} threads, if this makes the computer crash lower n_threads number')
files_index = range(len(files_updated))

with Pool(int(n_threads)) as p:
    result = list(tqdm(p.imap(derotate, files_updated), total=len(files_updated)))


print('Creating data cube')
dim_data_cube = 2048
data_cube = np.zeros(shape=(len(files_updated), dim_data_cube, dim_data_cube), dtype=np.int16)
print(f'Data cube has a shape of {np.shape(data_cube)}')

for i, drot_slice in enumerate(result):
    data_cube[i] = drot_slice

drot_time_end = t.perf_counter()
drot_time_elapsed = drot_time_end - drot_time_start
print(f'Time elapsed for derotating and creating the data_cube: {(drot_time_elapsed/60):.1f} min \n')

tdeltas = np.array(create_tdeltas_array(files_updated), dtype=np.int32)


data_cube_h5_path = f'{directory_of_data}{day}_data.h5'
print(f'Saving data_cube in a .h5 file in {data_cube_h5_path} \n')
saving_h5_time_start = t.perf_counter()
with h5py.File(data_cube_h5_path, 'w') as f:
    power_map = f.create_dataset('time_series', data=data_cube)
    tdeltas = f.create_dataset('tdeltas', data=tdeltas)
saving_h5_time_end = t.perf_counter()
saving_h5_time_elapsed = saving_h5_time_end - saving_h5_time_start
print(f'h5 data file saved in {(saving_h5_time_elapsed/60):.1f} s \n')
print('Data file creation finished')