import numpy as np
import glob
from tqdm import tqdm
from multiprocessing import Pool
import itertools
import time as t
import os
from file_size_filter import *
from create_list_files_day import *
from limb_darkening import remove_limb_darkening
from adjust_intensity import remove_out_of_disk #, median_adjust_intensity
from fits_utils import open_fits_file, open_fits_fz_file, save_fits_file
import sys


#################
# DAY OF INTEREST
#################
year = sys.argv[1]
month = sys.argv[2]
day = sys.argv[3]
n_proc = sys.argv[4]
print(f'Day of calculations: {day} \n')


################################################################
# DIRECTORY WHERE ALL .FITS.FZ FILES ARE STORED FOR A SINGLE DAY
################################################################
directory_of_data = f'/home/guillem/{day}/'
print(f'Data is in directory {directory_of_data} \n')
if 'updated' not in os.listdir(directory_of_data):
    os.makedirs(directory_of_data+'updated')


########################################################
# LIST OF ALL USEFUL .FTIS.FZ FILE PATHS FOR A GIVEN DAY
########################################################
print('Creating file list and applying filters')
files = sorted(glob.glob(directory_of_data+'*.fits.fz'))
files = filter1(files)
obs_windows_telescopes = observation_windows_telescopes(files)
obs_windows_times = observation_windows_times(obs_windows_telescopes)
obs_windows_times_telescopes = observation_windows_times_telescopes(obs_windows_times, files)
obs_windows_times_telescopes = add_sharpness_entry(obs_windows_times_telescopes)
obs_windows_times_telescopes = filter_density(obs_windows_times_telescopes)
obs_windows_times_telescopes = filter_sharpness(obs_windows_times_telescopes)
final_files_list = final_list_of_files(obs_windows_times_telescopes)
print(f'Total of {len(final_files_list)} files to be preprocessed \n')
print('Files filtered \n')


#####################
# LIMB DARKENING GRID
#####################
data_axis_len = 2048 #dimension of images (2048x2048)
xcenter = ycenter = data_axis_len/2 #position of the center
xarr = np.arange(0, data_axis_len, 1) # X direction (pix) array
yarr = np.arange(0, data_axis_len, 1) # Y direction (pix) array 
xx, yy = np.meshgrid(xarr, yarr) # XY plane array
grid = np.sqrt((xx-xcenter)**2 + (yy-ycenter)**2) #grid for limb darkening correction


####################
# Data preprocessing
####################

def preprocess_data(args):
    file = args
    updated_filename = f'{directory_of_data}updated/{file[-24:-8]}_updated.fits'
    header, data = open_fits_fz_file(file)
    data = remove_limb_darkening(data, grid)
    #data = median_adjust_intensity(data, reference_data, inside_indices_list)
    save_fits_file(updated_filename, header, data)
    return f'{file[-24:-8]}_updated.fits created'


# CASE 0:
updated_filename0 = f'{directory_of_data}updated/{final_files_list[0][-24:-8]}_updated.fits'
ref_header, ref_data = open_fits_fz_file(final_files_list[0])
ref_data = remove_limb_darkening(ref_data, grid)
save_fits_file(updated_filename0, ref_header, ref_data)

print('Prepocessing data, removing limb darkening, adjusting intensity, etc ')
preprocessing_time_start = t.perf_counter()
preprocess_iter = final_files_list[1:]

with Pool(int(n_proc)) as preprocess_pool:
    preprocess_pool = list(tqdm(preprocess_pool.imap(preprocess_data, preprocess_iter), total=len(final_files_list)-1))

preprocessing_time_end = t.perf_counter()
prepocessing_time_elapsed = preprocessing_time_end - preprocessing_time_start
print(f'Preprocessing took in total: {prepocessing_time_elapsed/60:.1f} min \n')