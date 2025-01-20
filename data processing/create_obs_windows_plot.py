import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import matplotlib.patches as mpatches
fontsize = 25
plt.rc('font', size=fontsize)  

"""#################
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
obs_windows_telescopes = observation_windows_telescopes(files, plotting=True)"""

def create_tdeltas_array(file_list):
    FMT = '%H%M%S'
    tdeltas = [0]
    if file_list[0][-12:-5] == 'updated':
        time_str_ref = file_list[0][-21:-15]
        for fits_file in file_list[1:]:
            time_str = fits_file[-21:-15]
            tdeltas.append( timedelta.total_seconds(datetime.strptime(time_str, FMT) - datetime.strptime(time_str_ref, FMT)) )
        print('tdeltas array calculated')
    else:
        time_str_ref = file_list[0][-16:-10]
        for fits_file in file_list[1:]:
            time_str = fits_file[-16:-10]
            tdeltas.append( timedelta.total_seconds(datetime.strptime(time_str, FMT) - datetime.strptime(time_str_ref, FMT)) )
        print('tdeltas array calculated')
    return np.array(tdeltas, dtype=np.int32)


file_path_original = '/home/guillem/20241022/updated/20241022_data.h5'
file_path_modified = '/home/guillem/20241022/updated/20241022_data_modified.h5'

with h5py.File(file_path_modified) as f:
    print(f.keys())
    data_modified = np.array(f['time_series'][:])

tdeltas_modified = create_tdeltas_array(sorted(glob.glob('/home/guillem/20241022/updated/'+'*.fits')))

with h5py.File(file_path_original) as f:
    print(f.keys())
    data_original = np.array(f['time_series'][:])

tdeltas_original = create_tdeltas_array(sorted(glob.glob('/home/guillem/20241022/updated/'+'*.fits')+ ['/home/guillem/20241022/updated/' + file[-29:] for file in glob.glob('/home/guillem/20241022/updated/delete/'+'*.fits')]))

print('Computing telescope change instants')
telescopes = {'L': 'Learmonth', 'U':'Udaipur', 'T':'El Teide', 'C':'Cerro Tololo', 'B':'Big Bear', 'M':'Manua Loa'}
change_telescope_indices_original = []
files_original = sorted(glob.glob('/home/guillem/20241022/updated/'+'*.fits') + ['/home/guillem/20241022/updated/' + file[-29:] for file in glob.glob('/home/guillem/20241022/updated/delete/'+'*.fits')])
for num, updated_file in enumerate(files_original[1:]):
    if updated_file[-15] != files_original[num][-15]:
        change_telescope_indices_original.append(num+1)

    

files_updated = sorted(glob.glob('/home/guillem/20241022/updated/'+'*.fits'))
change_telescope_indices = []
for num, updated_file in enumerate(files_updated[1:]):
    if updated_file[-15] != files_updated[num][-15]:
        change_telescope_indices.append(num+1)


indx = 1240
indy = 1240


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))

telescopes = {'L': 'Learmonth', 'U':'Udaipur', 'T':'El Teide', 'C':'Cerro Tololo', 'B':'Big Bear', 'M':'Manua Loa'}
colors = {'L': 'red', 'U':'blue', 'T':'green', 'C':'purple', 'B':'orange', 'M':'cyan'}

telescopes_legend = {'L': 'Learmonth', 'U':'Udaipur', 'T':'El Teide', 'B':'Big Bear'}
colors_legend = {'L': 'red', 'U':'blue', 'T':'green', 'B':'orange'}
patches = [mpatches.Patch(color=color, label=telescopes_legend[label], alpha=0.5) for label, color in colors_legend.items()]

ax0 = ax[0]
ax0.plot(tdeltas_original/3600, data_original[1:, indx, indy], color='black')
ax0.fill_betweenx(x1=0, x2=tdeltas_original[change_telescope_indices_original[0]]/3600, y=[2800, 3000], color=colors[files_original[0][-15]], alpha=0.5)
for k, change in enumerate(change_telescope_indices_original):
    color = colors[files_original[change][-15]]
    ax0.axvline(tdeltas_original[change]/3600, color='red', alpha=0.5, linestyle='--')
    if k < len(change_telescope_indices_original)-1:
        ax0.fill_betweenx(x1=tdeltas_original[change_telescope_indices_original[k]]/3600, x2=tdeltas_original[change_telescope_indices_original[k+1]]/3600, y=[2800, 3000], color=color, alpha=0.5)
        #if (tdeltas_original[change_telescope_indices_original[k+1]]/3600 - tdeltas_original[change_telescope_indices_original[k]]/3600) > 5:
        #    ax0.text(x=tdeltas_original[change_telescope_indices_original[k]]/3600 + 0.8, y=2840, s=telescopes[files_original[change][-15]], color='black')
        #else:
        #    ax0.text(x=tdeltas_original[change_telescope_indices_original[k]]/3600 + 0.1, y=2840, s=telescopes[files_original[change][-15]], color='black')
    else:
        ax0.fill_betweenx(x1=tdeltas_original[change_telescope_indices_original[k]]/3600, x2=tdeltas_original[-1]/3600, y=[2800, 3000], color=color, alpha=0.5)
        #ax0.text(x=tdeltas_original[change_telescope_indices_original[k]]/3600 + 0.8, y=2840, s=telescopes[files_original[change][-15]], color='black')

ax0.annotate('(a)', xy=(0.1, 0.9), xycoords='axes fraction', fontsize=fontsize)
ax0.set_ylim(2800, 3800)
ax0.set_xlim(0, 24)
ax0.set_xticks([])
ax0.set_ylabel('Intensity (counts)')

ax1 = ax[1]
ax1.plot(tdeltas_modified/3600, data_modified[1:, indx, indy], color='black')
ax1.fill_betweenx(x1=0, x2=tdeltas_modified[change_telescope_indices[0]]/3600, y=[2800, 3000], color=colors[files_updated[0][-15]], alpha=0.5)
for k, change in enumerate(change_telescope_indices):
    color = colors[files_updated[change][-15]]
    ax1.axvline(tdeltas_modified[change]/3600, color='red', alpha=0.5, linestyle='--')

    if k < len(change_telescope_indices)-1:
        ax1.fill_betweenx(x1=tdeltas_modified[change_telescope_indices[k]]/3600, x2=tdeltas_modified[change_telescope_indices[k+1]]/3600, y=[2800, 3000], color=color, alpha=0.5)
        #if (tdeltas_modified[change_telescope_indices[k+1]]/3600 - tdeltas_modified[change_telescope_indices[k]]/3600) > 5:
            #ax1.text(x=tdeltas_modified[change_telescope_indices[k]]/3600 + 0.8, y=2840, s=telescopes[files_updated[change][-15]], color='black')
        #else:
            #ax1.text(x=tdeltas_modified[change_telescope_indices[k]]/3600 + 0.1, y=2840, s=telescopes[files_updated[change][-15]], color='black')
    else:
        ax1.fill_betweenx(x1=tdeltas_modified[change_telescope_indices[k]]/3600, x2=tdeltas_modified[-1]/3600, y=[2800, 3000], color=color, alpha=0.5)
        #ax1.text(x=tdeltas_modified[change_telescope_indices[k]]/3600 + 0.8, y=2840, s=telescopes[files_updated[change][-15]], color='black')

ax1.annotate('(b)', xy=(0.1, 0.9), xycoords='axes fraction', fontsize=fontsize)
ax1.set_ylim(2800, 3800)
ax1.set_xlim(0, 24)
ax1.set_xticks(np.arange(0, 25, 4))
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Intensity (counts)')

#ax0.legend(handles=patches, loc='center left', bbox_to_anchor=(1.02, 0.8))


# Place the legend in the separate axis, occupying the entire height
ax0.legend(handles=patches, loc='center left', bbox_to_anchor=(0.68, 0.82))
ax1.legend(handles=patches, loc='center left', bbox_to_anchor=(0.68, 0.82))

plt.tight_layout()
plt.savefig('/home/guillem/obs_windows_plot.pdf', dpi=400, bbox_inches='tight')
plt.close()