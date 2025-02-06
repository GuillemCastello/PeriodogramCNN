import numpy as np
from decorators import benchmark

#@benchmark
def remove_limb_darkening(data, grid, u2=0.88, v2=-0.23, radius=900):
    #######################################
    # data : numpy array to remove limb darkening from
    # u2 & v2: obtained from Cox, A.N.: 2000, Allen’s astrophysical quantities, 355
    # Other methods can be applied but this is used broadly
    # radius: for our original images the radius is 900 (obtained from header)
    ########################################

    # Normalize to get values between 0 and 1
    # 0 -> center / 1 -> border / basically tells us the angle radially
    grid = grid/radius

    # values from outside the sun go to zero
    out = np.where(grid>1) 
    grid[out]=0

    # Angle mu for limb darkening / 1->center / 0->border
    mu = np.cos(grid) 
    
    # Correct the data
    limb_darkening = 1 - u2 - v2 + u2*mu + v2*mu**2
    corrected_data = data/limb_darkening
    return corrected_data