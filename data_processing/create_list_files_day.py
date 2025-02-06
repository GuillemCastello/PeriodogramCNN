import numpy as np
from fits_utils import open_fits_fz_file
from decorators import benchmark
import sys

@benchmark
def observation_windows_telescopes(file_list, plotting=False, savefig_path=None, order_files=False):
    # INPUT: LIST WITH ALL THE FILES ORDERED CHRONOLOGICALLY
 
    # OUTPUT: LIST OF LISTS WITH THE TELESCOPE OBSERVATIONS WINDOWS WITH [OBS-SITE (str), START (int), END (int), N-FILES (int)] ORDERED BY THE START HOUR
    #         IF order_files = True THIS FUNCTION WILL RETURN ALSO A LIST OF LISTS WITH THE FILE PATHS (NOT NEEDED)
    
    # THIS IS THE LIST OF POSSIBLE OBS SITES
    # ['L', 'M', 'U', 'C', 'T', 'B']
    obs_order = ['L', 'M', 'U', 'C', 'T', 'B']

    # SPLIT THE INPUT INTO TELESCOPES (WE GET A LIST OF LISTS, EACH INDEX REPRESENT A DIFFERENT TELESCOPE)
    files_obs_site_ordered = []
    missing_obs = []
    for obs_site in obs_order:
        files_obs_site = list(filter(lambda files_list: files_list[:][-10] == obs_site, file_list))
        files_obs_site_ordered.append(files_obs_site)
        if len(files_obs_site) == 0:
            missing_obs.append(obs_site)

    if len(missing_obs) > 0:
        print(f'The observatories {missing_obs} have not taken any data for this day.')
    else:
        print('All observatories have taken data for this day.')
    # HELPER FUNCTION TO RETURN A LIST OF TUPLES SORTED BY THE SECOND ELEMENT IN THE TUPLES, TO ORDER BY START OF OBSERVATION
    def Sort_Tuple(tup):
        tup.sort(key = lambda x: x[1])
        return tup

    # LIST OF LIST CONTAINING:
    #    ['OBS_SITE', START, STOP, N-FILES]
    start_end = []


    # FOR EACH OBS_SITE
    for obs_site in files_obs_site_ordered:

        # TRY-EXCEPT TO AVOID A INDEXERROR WHEN A OBS_SITE HAS NO FILES FOR A GIVEN DAY
        try:
            # USED FOR CHECKING IF ANY OBS-SITE HAS A SECOND OBSERVATION WINDOW
            cuts = 0
            for j, hour in enumerate(obs_site):
                if int(hour[-16:-10]) > 120000:
                    if cuts > 0:
                        continue
                    else:
                        cuts = j
            # IF ANY TELESCOPE HAS A SECOND OBSERVATION WINDOW APPEND 2 DIFFERENT TUPLES WITH THE RESPECTIVE OBS WINDOWS
            if (cuts > 0) and ((int(obs_site[cuts][-16:-10]) -  int(obs_site[cuts-1][-16:-10])) >= 10000 ):
                start_end.append([obs_site[0][-10], obs_site[0][-16:-10], obs_site[cuts-1][-16:-10], cuts])
                start_end.append([obs_site[0][-10], obs_site[cuts][-16:-10], obs_site[-1][-16:-10], len(obs_site)-cuts])
            else:
                start_end.append([obs_site[0][-10], obs_site[0][-16:-10], obs_site[-1][-16:-10], len(obs_site)])
        
        except IndexError:
            continue
    
    # SORT THE LIST OF TUPLES
    start_end = Sort_Tuple(start_end)

    # HELPER FUNCTION TO CHANGE FROM HHMMSS TO TOTAL NUMBER OF SECONDS
    def hhmmss_to_s(time):
        return int(time[0:2])*3600 + int(time[2:4])*60 + int(time[4:])
    
    for l in range(len(start_end)):
        start_end[l][1] = hhmmss_to_s(start_end[l][1])
        start_end[l][2] = hhmmss_to_s(start_end[l][2])

    # PLOTTING THE OBSERVATION WINDOWS FOR EACH TELESCOPE
    if plotting == True:
        import matplotlib.pyplot as plt

        plt.figure()
        lines_start = []
        lines_end = []
        for i in range(len(start_end)):
            lines_start.append(int(start_end[i][1])/3600)
            lines_end.append(int(start_end[i][2])/3600)
            if lines_end[i] <= 15:
                plt.text(lines_start[i], i+0.1, f'{round(lines_end[i]-lines_start[i], 1)} h / {start_end[i][-1]} files')
            else:
                plt.text(lines_start[i], i+0.1, f'{round(lines_end[i]-lines_start[i],1)} h / {start_end[i][-1]} files')

        plt.hlines(y=range(len(start_end)), xmin=lines_start, xmax=lines_end, colors='k')
        #plt.vlines(x=lines_start, ymin=0, ymax=len(start_end)-1, colors='r', linestyles='dashed')
        #plt.vlines(x=lines_end, ymin=0, ymax=len(start_end)-1, colors='g', linestyles='dashed')
        yticks = [telescope[0] for telescope in start_end]
        xticks = np.array([0,5,10,15,20,24])
        plt.yticks(range(len(start_end)), yticks)
        plt.xticks(xticks, map(str, xticks))
        plt.ylabel('Telescope')
        plt.xlabel('DayTime [h]')
        plt.title(str(file_list[0][-24:-20]) + '-' + str(file_list[0][-20:-18]) + '-' + str(file_list[0][-18:-16]))
        if savefig_path != None:
            plt.savefig(savefig_path)
            plt.close()
        else:
            plt.show()
            plt.close()

    # RETURN THE LIST OF FILES ORDERED PER TELESCOPE OBS WIDNOW (NOT NEEDED, BUT STILL HERE JUST IN CASE)
    if order_files == True:
        # LIST OF LIST WITH THE FILES FOR EACH TELESCOPE OBS WINDOW
        files_ordered = []
        
        # WE ITERATE THE LIST OF TUPLES WITH OBS WINDOWS
        for condition in start_end:
            # CREATE A LIST FOR EACH OBS WINDOW FOR EACH TELESCOPE AND APPEND IT TO THE MAIN LIST
            files_ordered_temp = []
            for file_list in files_obs_site_ordered:
                for file in file_list:
                    if (file[-10] == condition[0]) and (int(file[-16:-10]) >= int(condition[1])) and (int(file[-16:-10]) <= int(condition[2])):
                        files_ordered_temp.append(file)
            files_ordered.append(files_ordered_temp)        
        return start_end, files_ordered
    
    #
    return start_end

@benchmark
def observation_windows_times(obs_windows):
    # INPUT: THE LIST WITH THE TELESCOPE OBSERVATION WINDOWS
    # OUTPUT: LIST OF LIST OF TIME OBSERVATION WIDNOWS WITH THE TELESCPES OBSERVING

    # DISCTIONARY TO HELP WITH TELECOPE NAMING
    telescope_dict = {}
    for id, letter in enumerate(obs_windows):
        telescope_dict[id] = letter[0]

    # ARRAYS WITH TOTAL OBSERVING TIME (SECONDS) FOR EACH TELESCOPE OBSERVING WINDOW
    all_times = []
    for window in obs_windows:
        delta_t = np.arange(int(window[1]), int(window[2]), 1)
        all_times.append(delta_t)

    # TIME IN SEC OF THE DAY OF OBSERVATIONS

    # START TIME =-1 IS TO GET NUMBERS TO MATCH CORRECTLY
    start_time = -1
    # WE FIND THE LAST TIME THERE WAS AN OBSERVATION IN THE DAY
    end_time = int(obs_windows[0][2])
    for instant in obs_windows:
        if int(instant[1]) <= start_time:
            start_time = int(instant[1])
        if int(instant[2]) >= end_time:
            end_time = int(instant[2])
    total_time = np.arange(start_time, end_time, 1)

    # FOR EACH SECODN WE CHECK IF THERE WAS AT LEAST ONE TELESCOPE OBSERVING.
    # THE INDEX OF THE ARRAY REPRESENT THE SECOND, IF THERE ARE TELESCOPES OBSERVING WE APPEND IT TO THAT INDEX
    # HAS TO BE LIST OF LISTS, CAUSE EACH INDEX MAY HVE DIFFERENT NUMBER OF TELESCOPES OBSERVING
    telescope_windows = []
    for time in total_time:
        telescopes = []
        for telescope, times in enumerate(all_times):
            if time in times:
                telescopes.append(telescope)
        if len(telescopes) == 0:
            telescopes = ['None']
        telescope_windows.append(telescopes)

    # TIME OBSERVATION WINDOWS
    # LIST OF LIST WITH EACH TIME OBSERVATION WINDOW:
    #   [ [OBS1_START, OBS1_END -1 , [OBS1_TELESCOPES]], [OBS2_START, OBS2_END -1 , [OBS2_TELESCOPES]], ..., [OBSN_START, OBSN_END, [OBSN_TELESCOPES]] ]
    windows = []
    windows.append([0, 0, telescope_windows[0]])
    for i in range(len(telescope_windows)-1):
        if telescope_windows[i] != telescope_windows[i+1]:
            windows.append([i, i, list(map(telescope_dict.get, telescope_windows[i+1]))])
    for j in range(len(windows)):
        if j < len(windows)-1:
            windows[j][1] = windows[j+1][0] - 1
        else:
            windows[j][1] = len(total_time)

    
    return windows

@benchmark
def observation_windows_times_telescopes(obs_windows_times, file_list):
    # INPUT: LIST OF TIME OBSERVATION WINDOWS
    # OUTPUT: LIST OF TIME-TELESCOPE OBSERVATION WINDOWS (i.e: [ [[.FZ, 1NFILES, TELESCOPE1], ..., [.FZ, 4NFILES, TELESCOPE4], TIME-DURATION], [[.FZ, 1NFILES, TELESCOPE1], ..., [.FZ, 3NFILES, TELESCOPE3],  TIME-DURATION], ... ], 3 INDICES)

    def hhmmss_to_s(time):
        return int(time[0:2])*3600 + int(time[2:4])*60 + int(time[4:])

    # LIST OF TIME-TELESCOPE OBSERVATION WINDOWS
    obs_windows_times_telescopes = []

    # FOR EACH TIME OBS WINDOW
    for obs_window in obs_windows_times:
        # LIST OF LISTS WITH FILES / TELESCOPES FOR A TIME OBSERVATION WINDOW
        obs_window_files = []

        # AVOID GAPS WHERE NO TELESCOPE IS OBSERVING
        if 'None' not in obs_window[-1]:
            # FOR EACH TELESCOPE OBSERVING IN THE TIME OBSERVING WINDOW
            for telescope in obs_window[-1]:

                # LIST OF FILES FOR A GIVEN TELESCOPE
                obs_window_telescope_files = []

                # ITERATE THRUOGH THE LIST OF FILES AND CHECK IF THEY ARE FROM THE GIVEN TIME INTERVAL AND TELESCOPE
                for file in file_list:
                    obs_time_s = hhmmss_to_s(file[-16:-10])
                    file_telescope = file[-10]
                    if (obs_time_s >= obs_window[0]) and (obs_time_s <= obs_window[1]) and (file_telescope == telescope):
                        # APPEND FILE TO LIST OF FILES FOR A GIVEN TELESCOPE
                        obs_window_telescope_files.append(file)
                
                # IF AT LEAST A FILE IS APPENDED BEFORE:
                if len(obs_window_telescope_files) > 0:
                    # APPEND THE LIST OF FILES WITH THE TELECOPE NAME TO THE LIST OF LISTS
                    obs_window_files.append([obs_window_telescope_files, len(obs_window_telescope_files), telescope])
            # APPEND EACH LIST OF LIST TO THE FINAL LIST
            obs_windows_times_telescopes.append([obs_window_files, obs_window[1]-obs_window[0]+1])
    return obs_windows_times_telescopes

@benchmark
def add_sharpness_entry(obs_windows_times_telescopes):
    for window in obs_windows_times_telescopes:
        for telescope in window[0]:
            sharp = 0
            for file in telescope[0]:
                header, _ = open_fits_fz_file(file)
                sharp += header['SHARPNSS']
            sharp = sharp/(len(telescope[0]))
            telescope.append(sharp)
    return obs_windows_times_telescopes

@benchmark
def filter_density(obs_windows_times_telescopes):
    for window in obs_windows_times_telescopes:
        if len(window[0]) > 1: 
            for id, telescope in enumerate(window[0]): 
                if (telescope[1]/(window[1]/60) > 1.1) or (telescope[1]/(window[1]/60) < 0.9):
                    window[0].pop(id)
    return obs_windows_times_telescopes

@benchmark
def filter_sharpness(obs_windows_times_telescopes):
    for window in obs_windows_times_telescopes:
        if len(window[0]) > 1:
            while len(window[0]) > 1:
                sharps=[]
                for telescope in window[0]:
                    sharps.append(telescope[3])
                window[0].pop(np.argmin(sharps))
    return obs_windows_times_telescopes

@benchmark
def final_list_of_files(obs_windows_times_telescopes):
    final_files = []
    for window in obs_windows_times_telescopes:
        # Try except to catch time intervals where not a single telescope has been observing
        # we don't care about it just go to the next
        try:
            for file in window[0][0][0]:
                final_files.append(file)
        except IndexError:
            continue
    return final_files