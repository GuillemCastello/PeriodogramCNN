from mapping.plot_map import *        
from mapping.transform_map import *   
from mapping.fits2map import * 
from decorators import benchmark

# FUNCTION THAT DEROTATES A MAP TO A GIVEN TIME, DEFAULT TO 12:00:00
def derotate(file):
    # INPUTS:
    # file: path to the file we want to derotate
    # OUTPUTS:
    # data array of the derotated map
    map = fits2map(file)
    time_str = map['time'][:10] + ' 12:00:00.000'
    derotated_map = drot_map(map, time=time_str, same_center=True)
    map = None
    derotated_map_corr = np.array(derotated_map['data'], dtype=np.int16)
    derotated_map = None
    time_str = None
    return derotated_map_corr
   